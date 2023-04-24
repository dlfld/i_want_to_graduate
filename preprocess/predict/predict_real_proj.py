from typing import List

from javalang.tree import MethodDeclaration
import models


def get_method_asts(code: str) -> List[MethodDeclaration]:
    """
        获取java文件内方法AST节点
    :param code: .java文件的code
    :return: .java文件内所有方法转换成AST的方法节点列表
    """
    method_asts = []
    # 将输入代码转换成ast树
    program_ast = javalang.parse.parse(code)
    # 遍历所有的节点
    for ast_type in program_ast.types:
        # 节点名
        node_name = type(ast_type).__name__
        # 方法应该存在于类定义中，因此需要找到类定义的标签
        if node_name == "ClassDeclaration":
            # 遍历类节点的子节点，获取到方法节点
            for body in ast_type.body:
                # 获取当前节点的节点名
                body_type = type(body).__name__
                # 如果当前节点是方法节点
                if body_type == "MethodDeclaration":
                    # 添加结果集
                    method_asts.append(body)

    return method_asts


def get_proj_method_asts(proj_dir: str) -> List[MethodDeclaration]:
    """
        扫描工程文件目录，获取工程中所有方法的AST
    :param proj_dir: 工程文件目录
    :return: 工程中所有的方法ast节点
    """
    method_ast_list = []
    # 遍历

    for rt, dirs, files in os.walk(proj_dir):
        for file in files:
            if file.endswith(".java"):
                programfile = open(os.path.join(rt, file), encoding='utf-8')
                # 读取文件
                code = programfile.read()
                # 获取方法的AST节点
                try:
                    asts = get_method_asts(code)
                    method_ast_list.extend(asts)
                except Exception as e:
                    # print(e)
                    pass
                    
                programfile.close()
    return method_ast_list


import os
import javalang
import javalang.tree
import javalang.ast
import javalang.util
from javalang.ast import Node
import torch
from anytree import AnyNode
# import treelib
from utils.dependencies.createclone_java import getedge_nextsib, getedge_flow, getedge_nextstmt, getedge_nexttoken, getedge_nextuse


def get_token(node):
    """
        获取输入node的token
    """
    token = ''
    if isinstance(node, str):
        token = node
        # print(f"node->{node}")
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    return token


def get_child(root):
    """
        获取当前节点的孩子节点
    """
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    # print(sub_item)
                    yield sub_item
            elif item:
                # print(item)
                yield item

    return list(expand(children))


def get_sequence(node):
    """
        获取所有的token 
    """
    sequence = []
    token, children = get_token(node), get_child(node)
    sequence.append(token)
    for child in children:
        res = get_sequence(child)
        sequence.extend(res)
    return sequence

def getnodes(node, nodelist):
    nodelist.append(node)
    children = get_child(node)
    for child in children:
        getnodes(child, nodelist)


def createtree(root, node, nodelist, parent=None):
    id = len(nodelist)
    token, children = get_token(node), get_child(node)
    if id == 0:
        root.token = token
        root.data = node
    else:
        newnode = AnyNode(id=id, token=token, data=node, parent=parent)
    nodelist.append(node)
    for child in children:
        if id == 0:
            createtree(root, child, nodelist, parent=root)
        else:
            createtree(root, child, nodelist, parent=newnode)

def getnodeandedge_astonly(node, nodeindexlist, vocabdict, src, tgt):
    """
        创建ast的边
        采用的是DFS的方式
    """
    token = node.token
    nodeindexlist.append([vocabdict[token]])
    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        src.append(child.id)
        tgt.append(node.id)
        getnodeandedge_astonly(child, nodeindexlist, vocabdict, src, tgt)

def getnodeandedge(node, nodeindexlist, vocabdict, src, tgt, edgetype):
    token = node.token
    nodeindexlist.append([vocabdict[token]])
    # =====================添加自链接的边   效果不是很好，先不加================================
    # src.append(node.id)
    # tgt.append(node.id)
    # edgetype.append([0])
    # =====================添加自链接的边================================
    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        edgetype.append([0])
        src.append(child.id)
        tgt.append(node.id)
        edgetype.append([0])
        getnodeandedge(child, nodeindexlist, vocabdict, src, tgt, edgetype)

def countnodes(node, ifcount, whilecount, forcount, blockcount):
    token = node.token
    if token == 'IfStatement':
        ifcount += 1
    if token == 'WhileStatement':
        whilecount += 1
    if token == 'ForStatement':
        forcount += 1
    if token == 'BlockStatement':
        blockcount += 1
    # print(ifcount, whilecount, forcount, blockcount)
    for child in node.children:
        countnodes(child, ifcount, whilecount, forcount, blockcount)

def createast(proj_dir):
    """
        遍历工程文件，获取每一个方法的AST，并对AST进行遍历，获取每一个AST的token
    """
    alltokens = []

     # 当前工程项目下所有的java方法转换成的ast
    proj_method_asts = get_proj_method_asts(proj_dir)
    # logddd.log(len(proj_method_asts))

    # 遍历每一个方法ast
    for method in proj_method_asts:
        res = get_sequence(method)
        alltokens.extend(res)
    
    # logddd.log(len(alltokens))

    ifcount = 0
    whilecount = 0
    forcount = 0
    blockcount = 0
    docount = 0
    switchcount = 0
    for token in alltokens:
        if token == 'IfStatement':
            ifcount += 1
        if token == 'WhileStatement':
            whilecount += 1
        if token == 'ForStatement':
            forcount += 1
        if token == 'BlockStatement':
            blockcount += 1
        if token == 'DoStatement':
            docount += 1
        if token == 'SwitchStatement':
            switchcount += 1

    alltokens = list(set(alltokens))
    vocabsize = len(alltokens)
    tokenids = range(vocabsize)
    vocabdict = dict(zip(alltokens, tokenids))
    return proj_method_asts,vocabsize, vocabdict

def create_separate_graph(ast_list,vocablen, vocabdict, device, mode='astonly', nextsib=False, ifedge=False,
                        whileedge=False, foredge=False, blockedge=False, nexttoken=False, nextuse=False):
    """
        创建图信息，根据ast重新构建树的结构，并在树上添加边
        并返回一条处理完成之后的数据
    """
    treelist = []
    logddd.log(len(ast_list))
    # 遍历方法列表
    for tree in ast_list:
        nodelist = []
        newtree = AnyNode(id=0, token=None, data=None)
        # 创建树
        createtree(newtree, tree, nodelist)
        # print(path)
        # print(newtree)
        x = []
        edgesrc = []
        edgetgt = []
        edge_attr = []
        if mode == 'astonly':
            getnodeandedge_astonly(newtree, x, vocabdict, edgesrc, edgetgt)
        else:
            getnodeandedge(newtree, x, vocabdict, edgesrc, edgetgt, edge_attr)
            if nextsib == True:
                # 链接下一个兄弟结点，将一个节点连接到它的下一个兄弟姐妹 (从左到右)。
                # 因为图神经网络不考虑节点的顺序，所以有必要向我们的神经网络模型提供子的顺序。
                # 尝试一下，如果不要兄弟结点呢？不考虑顺序信息。
                getedge_nextsib(newtree, vocabdict, edgesrc, edgetgt, edge_attr)
            getedge_flow(newtree, vocabdict, edgesrc, edgetgt, edge_attr, ifedge, whileedge, foredge)
            if blockedge == True:
                getedge_nextstmt(newtree, vocabdict, edgesrc, edgetgt, edge_attr)
            tokenlist = []
            if nexttoken == True:
                getedge_nexttoken(newtree, vocabdict, edgesrc, edgetgt, edge_attr, tokenlist)
            variabledict = {}
            if nextuse == True:
                getedge_nextuse(newtree, vocabdict, edgesrc, edgetgt, edge_attr, variabledict)
                edge_index = [edgesrc, edgetgt]

        astlength = len(x)
        treelist.append([x, edge_index, edge_attr])

    return treelist

def create_input_model_data(treelist):
    """
        将传入列表的元素进行两两组合，判断他们是不是克隆对
    """
    data_list = []
    for i in range(len(treelist)-1):
        for j in range(i+1,len(treelist)):
            data_list.append([treelist[i][0],treelist[j][0],treelist[i][1],treelist[j][1],treelist[i][2],treelist[j][2],i,j])
    return data_list

if __name__ == "__main__":
    import logddd
    dirname = 'kafka/'
    proj_method_asts,vocabsize, vocabdict = createast(dirname)
    logddd.log(len(proj_method_asts))
    treelist=create_separate_graph(proj_method_asts, vocabsize, vocabdict,device='cpu',mode='else',nextsib=True,ifedge=True,whileedge=True,foredge=True,blockedge=True,nexttoken=True,nextuse=True)
    logddd.log(f"len(tree) = {len(treelist)}")
    device=torch.device("cuda:0")
    model= models.GMNnet(77535, embedding_dim=100, num_layers=4, device=device).to(device)
    model.load_state_dict(torch.load('0.pt'))
    model=model.to(device)
    model.eval()


    for i in range(len(treelist)-1):
        for j in range(i+1,len(treelist)):
            data = [treelist[i][0],treelist[j][0],treelist[i][1],treelist[j][1],treelist[i][2],treelist[j][2],i,j]
    # for data in proj_data:
            x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2,index_i,index_j=data

            x1=torch.tensor(x1, dtype=torch.long, device=device)
            x2=torch.tensor(x2, dtype=torch.long, device=device)

            edge_index1=torch.tensor(edge_index1, dtype=torch.long, device=device)
            edge_index2=torch.tensor(edge_index2, dtype=torch.long, device=device)

            if edge_attr1!=None:
                edge_attr1=torch.tensor(edge_attr1, dtype=torch.long, device=device)
                edge_attr2=torch.tensor(edge_attr2, dtype=torch.long, device=device)

            data=[x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2]
            logits=model(data)
            # logits  = logits.squeeze(0)
            output = torch.sigmoid(logits)
            output = output.squeeze(0)
            
            if output[0] <= output[1]:
                logddd.log(f"找到相似 {index_i}  < -- > {index_j} --- > {output}")
            else:
                logddd.log(f" {index_i}  < -- > {index_j} 不相似 --- > {output} ")
            