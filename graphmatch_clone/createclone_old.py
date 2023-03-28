import os
import itertools
import pycparser
import torch
from torch_geometric.data import Data
from pycparser import c_parser
from pycparser.c_ast import Node
parser = c_parser.CParser()
Node.__slots__=('id')
print(Node.__slots__)
#quit()
token_mode='value'
def get_token(node, lower=True,mode='value'):
    
    name = node.__class__.__name__
    token = name
    if mode=='typeonly':
        return token
    elif mode=='value':
        is_name = False
        if len(node.children()) == 0:
            attr_names = node.attr_names
            if attr_names:
                if 'names' in attr_names:
                    token = node.names[0]
                elif 'name' in attr_names:
                    token = node.name
                    is_name = True
                else:
                    token = node.value
            else:
                token = name
        else:
            if name == 'TypeDecl':
                token = node.declname
            if node.attr_names:
                attr_names = node.attr_names
                if 'op' in attr_names:
                    if node.op[0] == 'p':
                        token = node.op[1:]
                    else:
                        token = node.op
        if token is None:
            token = name
        if lower and is_name:
            token = token.lower()
        return token

def appendtokens(tree,tokenlist):
    tokenlist.append(get_token(tree,mode=token_mode))
    for (child_name, child) in tree.children():
        appendtokens(child,tokenlist)

def getnodes(tree,nodelist):
    nodelist.append(tree)
    for (child_name, child) in tree.children():
        getnodes(child,nodelist)

def getedges(tree,src,tgt,nodedict):
    for (child_name, child) in tree.children():
        src.append(nodedict[tree])
        tgt.append(nodedict[child])
        src.append(nodedict[child])
        tgt.append(nodedict[tree])
        getedges(child,src,tgt,nodedict)

def getnodeandedge(tree,indexlist,vocabdict,src,tgt,nodedict):
    token=get_token(tree,mode=token_mode)
    indexlist.append([vocabdict[token]])
    for (child_name, child) in tree.children():
        src.append(nodedict[tree])
        tgt.append(nodedict[child])
        src.append(nodedict[child])
        tgt.append(nodedict[tree])
        getnodeandedge(child,indexlist,vocabdict,src,tgt,nodedict)

class Queue():
    def __init__(self):
        self.__list = list()

    def isEmpty(self):
        return self.__list == []

    def push(self, data):
        self.__list.append(data)
    
    def pop(self):
        if self.isEmpty():
            return False
        return self.__list.pop(0)
def traverse(node,index):
    queue = Queue()
    queue.push(node)
    result = []
    while not queue.isEmpty():
        node = queue.pop()
        result.append(get_token(node,mode=token_mode))
        result.append(index)
        index+=1
        for (child_name, child) in node.children():
            #print(get_token(child),index)
            queue.push(child)
    return result

def createast():
    paths=[]
    asts=[]
    alltokens=[]
    dirname = 'sourcecode/'
    for i in range(1,16):
        for rt, dirs, files in os.walk(dirname+str(i)):
            count=0
            for file in files:
                programfile=open(os.path.join(rt,file))
                programtext=programfile.read()
                programtext=programtext.replace('\r','')
                programast=parser.parse(programtext)
                appendtokens(programast,alltokens)
                '''nodelist=[]
                getnodes(programast,nodelist)
                #print(nodelist)
                nodedict=dict(zip(nodelist,range(len(nodelist))))
                print(len(nodedict))
                edgesrc=[]
                edgetgt=[]
                getedges(programast,edgesrc,edgetgt,nodedict)
                edge_index=[edgesrc,edgetgt]
                print(len(edgesrc))
                print(edge_index)
                quit()'''
                programfile.close()
                programpath=os.path.join(rt,file)
                print(programpath)
                paths.append(programpath)
                asts.append(programast)
    astdict=dict(zip(paths,asts))
    #print(astdict)
    print(len(astdict))
    alltokens=list(set(alltokens))
    vocablen=len(alltokens)
    tokenids=range(vocablen)
    vocabdict=dict(zip(alltokens,tokenids))
    print(vocablen)
    return astdict,vocablen,vocabdict

def creategmndata(id,astdict,vocablen,vocabdict):
    if id=='0':
        trainfile=open('train.txt')
        validfile = open('valid.txt')
        testfile = open('test.txt')
    elif id=='13':
        trainfile = open('train13.txt')
        validfile = open('valid.txt')
        testfile = open('test.txt')
    elif id=='11':
        trainfile = open('train11.txt')
        validfile = open('valid.txt')
        testfile = open('test.txt')
    elif id=='0small':
        trainfile = open('trainsmall.txt')
        validfile = open('validsmall.txt')
        testfile = open('testsmall.txt')
    elif id == '13small':
        trainfile = open('train13small.txt')
        validfile = open('validsmall.txt')
        testfile = open('testsmall.txt')
    elif id=='11small':
        trainfile = open('train11small.txt')
        validfile = open('validsmall.txt')
        testfile = open('testsmall.txt')
    else:
        print('file not exist')
        quit()
    trainlist=trainfile.readlines()
    validlist=validfile.readlines()
    testlist=testfile.readlines()
    traindata=[]
    validdata=[]
    testdata=[]
    for line in trainlist:
        pairinfo=line.split()
        code1path=pairinfo[0].replace('\\','/')
        #print(pairinfo[0].replace('\\','/'))
        code2path = pairinfo[1].replace('\\','/')
        #print(code1path,code2path)
        ast1=astdict[code1path]
        ast2=astdict[code2path]
        nodelist=[]
        getnodes(ast1,nodelist)
        getnodes(ast2,nodelist)
        #print(len(nodelist))
        nodedict=dict(zip(nodelist,range(len(nodelist))))
        x=[]
        edgesrc=[]
        edgetgt=[]
        getnodeandedge(ast1,x,vocabdict,edgesrc,edgetgt,nodedict)
        ast1lenth=len(x)
        #print(ast1lenth)
        getnodeandedge(ast2,x,vocabdict,edgesrc,edgetgt,nodedict)
        pairlength=len(x)
        #print(pairlength)
        x=torch.tensor(x,dtype=torch.long)
        edge_index=torch.tensor([edgesrc,edgetgt],dtype=torch.long)
        #print(edge_index)
        edge2src=[]
        edge2tgt=[]
        for i in range(ast1lenth):
            for j in range(ast1lenth,pairlength):
                edge2src.append(i)
                edge2tgt.append(j)
                edge2src.append(j)
                edge2tgt.append(i)
        edge_index2=torch.tensor([edge2src,edge2tgt],dtype=torch.long)
        #print(edge_index2)
        data = Data(x=x, edge_index=edge_index,edge_index2=edge_index2)
        #print(data)
        traindata.append(data)
        #quit()
    return traindata,validdata,testdata
if __name__ == '__main__':
    xxx=0
    astdict,vocablen,vocabdict=createast()
    creategmndata('11',astdict,vocablen,vocabdict)

    