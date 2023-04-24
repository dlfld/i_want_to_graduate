import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import time
from param_parser import get_args
import sys
import argparse
from tqdm import tqdm, trange
import os
import pycparser
from createclone_bcb import createast,creategmndata,createseparategraph
import models
from torch_geometric.data import Data, DataLoader
from torch.utils.tensorboard import SummaryWriter   
from early_stopping import EarlyStopping
import logddd

# 获取参数
args = get_args()
import joblib
device=torch.device('cuda:0')
if not os.path.exists("data.data"):
    # 读取数据集获取数据信息 
    astdict,vocablen,vocabdict=createast()
    # 数据预处理
    treedict=createseparategraph(astdict, vocablen, vocabdict,device,mode=args.graphmode,nextsib=args.nextsib,ifedge=args.ifedge,whileedge=args.whileedge,foredge=args.foredge,blockedge=args.blockedge,nexttoken=args.nexttoken,nextuse=args.nextuse)
    # 获取格式化数据
    traindata,validdata,testdata=creategmndata(args.data_setting,treedict,vocablen,vocabdict,device)
    train_data = {
        "traindata":traindata,
        "validdata":validdata,
        "testdata":testdata,
        "vocablen":vocablen
    }
    joblib.dump(train_data,"data.data")
else:
    train_data = joblib.load("data.data")
    traindata,validdata,testdata,vocablen=train_data["traindata"],train_data["validdata"],train_data["testdata"],train_data["vocablen"]

    

num_layers=int(args.num_layers)
model=models.GMNnet(vocablen,embedding_dim=100,num_layers=num_layers,device=device).to(device)

# 这儿进行了修改，将原来的Adam改为了AdamW
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
criterion=nn.CosineEmbeddingLoss()
criterion2=nn.MSELoss()

criterion3 = torch.nn.BCEWithLogitsLoss(weight=torch.tensor([1, 6] ,dtype=torch.long, device=device))
# weight=torch.tensor([0.8576, 0.1424] ,dtype=torch.long, device=device)
# weight=torch.tensor([0.8576, 0.1424], dtype=torch.long, device=device)
criterion4 = nn.BCELoss(weight=torch.tensor([1, 6] ,dtype=torch.long, device=device)) # 交叉熵
save_path = "./models" #当前目录下
# early_stopping = EarlyStopping()
early_stopping = EarlyStopping(patience=10, verbose=True,save_path=save_path)  # 早停
# criterion4 = torch.nn.BCE

def create_batches(data):
    batches = [data[graph:graph+args.batch_size] for graph in range(0, len(data), args.batch_size)]
    return batches

def test(dataset):
    #model.eval()
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    results=[]
    for data,label in dataset:
        label =  [0] if label == -1 else [1]

        label=torch.tensor(label, dtype=torch.float, device=device)
        label=torch.unsqueeze(label,dim=0)
        x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2=data

        x1=torch.tensor(x1, dtype=torch.long, device=device)
        x2=torch.tensor(x2, dtype=torch.long, device=device)

        edge_index1=torch.tensor(edge_index1, dtype=torch.long, device=device)
        edge_index2=torch.tensor(edge_index2, dtype=torch.long, device=device)
        if edge_attr1!=None:
            edge_attr1=torch.tensor(edge_attr1, dtype=torch.long, device=device)
            edge_attr2=torch.tensor(edge_attr2, dtype=torch.long, device=device)

        data=[x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2]

        logits=model(data)
        output = torch.sigmoid(logits)


        # output=F.cosine_similarity(prediction[0],prediction[1])
        results.append(output.item())
        # 将值改为-1 1 0
        # prediction = torch.sign(output).item()
        prediction  = output

        if prediction>args.threshold and label.item()==1:
            tp+=1
            #print('tp')
        if prediction<=args.threshold and label.item()==0:
            tn+=1
            #print('tn')
        if prediction>args.threshold and label.item()==0:
            fp+=1
            #print('fp')
        if prediction<=args.threshold and label.item()==1:
            fn+=1
            #print('fn')
    print(tp,tn,fp,fn)
    p=0.0
    r=0.0
    f1=0.0
    if tp+fp==0:
        print('precision is none')
        return
    p=tp/(tp+fp)
    if tp+fn==0:
        print('recall is none')
        return
    r=tp/(tp+fn)
    f1=2*p*r/(p+r)
    acc = (tp + tn) / len(dataset)
    print(f'precision = {p}')
    print(f'recall = {r}')
    print(f'F1={f1}')
    print(f"acc = {acc}")
    return results

# 计算邻接矩阵
def to_adjacen_matrix(nodes,edge_index):
    # 构造邻接矩阵的样式并求出当前邻接矩阵大小
    data = [item[0] for item in nodes]
    min_id = min(data)
    max_id = max(data)
    len_nodes = max_id - min_id + 1
    # 构造邻接矩阵
    A = [[0 for x in range(len_nodes)] for y in range(len_nodes)]

    # 取出边矩阵
    sources,targets = edge_index
    # 对邻接矩阵赋值    
    for index,source in enumerate(sources):
        row = source-min_id
        col = targets[index] - min_id
        A[row][col]+=1

    return A

# loss_list = []
#=======================================================early stop=======================================================

train_losses = []
train_acces = []
# 用数组保存每一轮迭代中，在测试数据上测试的损失值和精确度，也是为了通过画图展示出来。
eval_losses = []
eval_acces = []
#=======================================================early stop=======================================================
writer = SummaryWriter('log/')
epochs = trange(args.num_epochs, leave=True, desc = "Epoch")
for epoch in epochs:# without batching
    print(epoch)
    batches=create_batches(traindata)
    # batches = batches[:10]
    main_index=0.0
    # 存储一个epoch的loss
    epoch_loss = 0.0
    train_loss = 0

    model.train()

    for index, batch in tqdm(enumerate(batches), total=len(batches), desc = "Batches"):
        optimizer.zero_grad()
        # batchloss= 0wei

        # 预测正确的数量
        # num_correct = 0
        batch_losses = []
             

        for data,label in batch:
            
            label =  [[1,0]] if label == -1 else [[0,1]]
            label=torch.tensor(label, dtype=torch.float, device=device)
            x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2=data
            x1=torch.tensor(x1, dtype=torch.long, device=device)
            x2=torch.tensor(x2, dtype=torch.long, device=device)

            edge_index1=torch.tensor(edge_index1, dtype=torch.long, device=device)
            edge_index2=torch.tensor(edge_index2, dtype=torch.long, device=device)
            if edge_attr1!=None:
                edge_attr1=torch.tensor(edge_attr1, dtype=torch.long, device=device)
                edge_attr2=torch.tensor(edge_attr2, dtype=torch.long, device=device)

            # x1_a = to_adjacen_matrix(x1, edge_index1)
            # x2_a = to_adjacen_matrix(x2, edge_index2)

            # data=[x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2,x1_a,x2_a]
            data=[x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2]
            logits=model(data)
            # pred_sig = torch.sigmoid(logits)
            logddd.log(logits)
            # 计算出当前预测是否正确,如果正确就计数，作为后面计算acc的条件
        
            
            loss = criterion3(logits,label)
            
            logddd.log(loss)
            batch_losses.append(loss.item())
            loss.backward(retain_graph=True)

        # 记录整个过程中每一个batch的loss
        batch_loss = np.average(batch_losses)

        # train loss 添加
        train_losses.append(batch_loss)
        
        # loss_list.append(batch_loss)
        writer.add_scalar('loss',batch_loss, epoch*len(batches)+index)
        epochs.set_description("Epoch (Loss=%g)" % round(batch_loss,5))

        optimizer.step()

        # 早停策略判断
    
    import joblib
    torch.save(model.state_dict(), f"models/{epoch}.pt")
    # joblib.dump(model,f"models/{epoch}.model")
    model.eval()
    # 验证和计算早停 
    with torch.no_grad():
        valid_loss_list = []
        val_data = validdata[:40000]

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        p=0.0
        r=0.0
        f1=0.0

        for data,label in val_data:
            label =  [[1,0]] if label == -1 else [[0,1]]
            
            label=torch.tensor(label, dtype=torch.float, device=device)
            # label=torch.unsqueeze(label,dim=0)
            x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2=data

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
            pred = output.squeeze(0)
          
            # p = torch.exp(logits)  反对数 实际预测的时候用的

            loss = criterion4(output,label)
            # 负对数似然函数
            # loss = F.nll_loss(logits, label)
            eval_losses.append(loss.item())

            if pred[0] > pred[1]:
                prediction = 0.4
            else:
                prediction = 0.6
            if prediction>args.threshold and label[0]==[0,1]:
                tp+=1
                #print('tp')
            if prediction<=args.threshold and label[0]==[1,0]:
                tn+=1
                #print('tn')
            if prediction>args.threshold and label[0]==[1,0]:
                fp+=1
                #print('fp')
            if prediction<=args.threshold and label[0]==[0,1]:
                fn+=1

        if tp+fp==0:
            print('precision is none')
            exit(1)

        p=tp/(tp+fp)
        if tp+fn==0:
            print('recall is none')
            exit(1)

        r=tp/(tp+fn)
        f1=2*p*r/(p+r)
        acc = (tp + tn) / len(val_data)
        print(f'\nprecision = {p}')
        print(f'recall = {r}')
        print(f'F1={f1}')
        print(f"acc = {acc}")

        avg_valid_loss = np.average(eval_losses)
        print("验证集loss:{}".format(avg_valid_loss))
        early_stopping(avg_valid_loss, model)
        if early_stopping.early_stop:
            print("此时早停！")
            break



    #test(validdata)

    # devresults=test(validdata[:40000])
    # testresults=test(testdata[:40000])
    # devfile=open('gmnbcbresult/'+args.graphmode+'_dev_epoch_'+str(epoch+1),mode='w')
    # for res in devresults:
    #     devfile.write(str(res)+'\n')
    # devfile.close()
    # testresults=test(testdata[:40000])
    # resfile=open('gmnbcbresult/'+args.graphmode+'_epoch_'+str(epoch+1),mode='w')
    # for res in testresults:
    #     resfile.write(str(res)+'\n')
    # resfile.close()

    #torch.save(model,'gmnmodels/gmnbcb'+str(epoch+1))
    #for start in range(0, len(traindata), args.batch_size):
        #batch = traindata[start:start+args.batch_size]
        #epochs.set_description("Epoch (Loss=%g)" % round(loss,5))

import joblib
joblib.dump(train_losses,args.loss_name)

'''for batch in trainloder:
    batch=batch.to(device)
    print(batch)
    quit()
    time_start=time.time()
    model.forward(batch)
    time_end=time.time()
    print(time_end-time_start)
    quit()'''
