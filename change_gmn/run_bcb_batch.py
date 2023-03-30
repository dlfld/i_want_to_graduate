import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import time
import sys
import argparse
from tqdm import tqdm, trange
import pycparser
from createclone_bcb import createast,creategmndata,createseparategraph
import models
from torch_geometric.data import Data, DataLoader
from torch.utils.tensorboard import SummaryWriter  
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from scipy.linalg import fractional_matrix_power


parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=True)
parser.add_argument("--dataset", default='gcj')
parser.add_argument("--graphmode", default='astandnext')
parser.add_argument("--nextsib", default=True)
parser.add_argument("--ifedge", default=True)
parser.add_argument("--whileedge", default=True)
parser.add_argument("--foredge", default=True)
parser.add_argument("--blockedge", default=True)
parser.add_argument("--nexttoken", default=True)
parser.add_argument("--nextuse", default=True)
parser.add_argument("--data_setting", default='11')
parser.add_argument("--batch_size", default=32)
parser.add_argument("--num_layers", default=4)
parser.add_argument("--num_epochs", default=10)
parser.add_argument("--lr", default=0.001)
parser.add_argument("--threshold", default=0.5)
args = parser.parse_args()
# args = parser.parse_known_args()[0]
device=torch.device('cuda:0')


astdict,vocablen,vocabdict=createast()
treedict=createseparategraph(astdict, vocablen, vocabdict,device,mode=args.graphmode,nextsib=args.nextsib,ifedge=args.ifedge,whileedge=args.whileedge,foredge=args.foredge,blockedge=args.blockedge,nexttoken=args.nexttoken,nextuse=args.nextuse)
traindata,validdata,testdata=creategmndata(args.data_setting,treedict,vocablen,vocabdict,device)


num_layers=int(args.num_layers)
model=models.GMNnet(vocablen,embedding_dim=100,num_layers=num_layers,device=device).to(device)
# 这儿进行了修改，将原来的Adam改为了AdamW
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
criterion=nn.CosineEmbeddingLoss()
criterion2=nn.MSELoss()
criterion3 = torch.nn.BCEWithLogitsLoss()



# 将当前x矩阵转换为邻接矩阵并进行归一化
def to_adjacen_matrix(nodes,edge_index):
    # 构造邻接矩阵的样式并求出当前邻接矩阵大小
    data = [item[0] for item in nodes]
    min_id = min(data)
    max_id = max(data)
    len_nodes = max_id - min_id + 1
    # 构造邻接矩阵
    A = [[0 for x in range(len_nodes)] for y in range(len_nodes)]
    print(len(A),len(A[0]))
    # 取出边矩阵
    sources,targets = edge_index
    # 对邻接矩阵赋值    
    for index,source in enumerate(sources):
        row = source-min_id
        col = targets[index] - min_id
        A[row][col]+=1

    # 归一化所需要的D矩阵
    matrix_d = [[0 for x in range(len_nodes)] for y in range(len_nodes)]

    # 添加指向自己的边（单位阵）
    for index in range(len(A)):
        A[index][index] += 1
        """
        执行归一化需要的数据
            找到D矩阵，在邻接矩阵两边分别✖️-根号D，
            D = 对邻接矩阵每一行求和并写在对角线上
        """
        matrix_d[index][index] = sum(A[index])

    
    # 下面对这个矩阵进行归一化
    D = np.array(matrix_d)
    D = fractional_matrix_power(D, -0.5)
    A = np.array(A)
    # 
    A = np.dot(D,A)
    A = np.dot(A,D)

    return A

# x = [[1],[2],[3],[4],[5]]
# edge_index = [[1,1,2,2,3,3,4,4,5,5],[3,2,1,4,1,5,2,5,4,3]]
# res = to_adjacen_matrix(x, edge_index)
# for item in res:
#     print(item)

# 存储一对数据的类
class PairData(Data):
    def __init__(self, edge_index_s=None, x_s=None, edge_index_t=None, x_t=None,edge_attr_s=None,edge_attr_t=None,label=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t
        self.edge_attr_s = edge_attr_s
        self.edge_attr_t = edge_attr_t
        self.label = label

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

# 所有数据的data列表
data_list = []
for item in traindata:
    total_data,label = item
    x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2=total_data
    x1 = to_adjacen_matrix(x1, edge_index1)
    x2 = to_adjacen_matrix(x2, edge_index2)
    # 数据处理
    label =  [0] if label == -1 else [1]

    # 创建pairdata
    data = PairData(
        edge_index_s=edge_index1,x_s=x1,edge_attr_s=edge_attr1,
        edge_index_t=edge_index2,x_t=x2,edge_attr_t=edge_attr2,
        label=label
    )
    data_list.append(data)





loss_list = []
writer = SummaryWriter('log/')
epochs = trange(args.num_epochs, leave=True, desc = "Epoch")
# dataloader
loader = DataLoader(data_list, batch_size=args.batch_size,follow_batch=['x_s', 'x_t'])

for epoch in epochs:# without batching
    print(epoch)
    # batches=create_batches(traindata)
    totalloss=0.0
    main_index=0.0
    
    for index, batch in tqdm(enumerate(loader), total=args.batch_size, desc = "Batches"):

        batch.edge_index_s = torch.tensor(batch.edge_index_s, dtype=torch.long, device=device)
        batch.x_s = torch.tensor(batch.x_s, dtype=torch.long, device=device)
        batch.edge_index_t = torch.tensor(batch.edge_index_t, dtype=torch.long, device=device)
        batch.x_t = torch.tensor(batch.x_t, dtype=torch.long, device=device)
        batch.edge_attr_s = torch.tensor(batch.edge_attr_s, dtype=torch.long, device=device)
        batch.edge_attr_t = torch.tensor(batch.edge_attr_t, dtype=torch.long, device=device)
        batch.label = torch.tensor(batch.label, dtype=torch.long, device=device)
        
        optimizer.zero_grad()
        batchloss= 0
        print(batch)
        logits=model(batch)
        batchloss = criterion3(logits, label)  # -log(sigmoid(1.5))

        batchloss.backward(retain_graph=True)
        optimizer.step()
        loss = batchloss.item()
        loss_list.append(loss)
        writer.add_scalar('loss',loss, epoch*len(batches)+index)
        epochs.set_description("Epoch (Loss=%g)" % round(loss,5))



