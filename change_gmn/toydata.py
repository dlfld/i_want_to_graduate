import torch
from torch_geometric.data import Data, DataLoader


edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
x2 = torch.tensor([[-1], [0], [1],[2],[3]], dtype=torch.float)
edge_index2 = torch.tensor([[0, 1, 0, 2,3,2],
                           [1, 0, 2, 0,4,3]], dtype=torch.long)

data = Data(x=x, edge_index=edge_index,edge_index2=edge_index2)
data2=Data(x=x, edge_index=edge_index2,edge_index2=edge_index)
datalist=[data,data2]
loader=DataLoader(datalist, batch_size=2)
for batch in loader:
    print(batch,batch.num_graphs)