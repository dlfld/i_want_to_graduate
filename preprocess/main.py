import logddd
import torch
import models

if __name__ == "__main__":

    dirname = 'kafka/'
    proj_method_asts,vocabsize, vocabdict = createast(dirname)
    logddd.log(len(proj_method_asts))
    treelist=create_separate_graph(proj_method_asts, vocabsize, vocabdict,device='cpu',mode='else',nextsib=True,ifedge=True,whileedge=True,foredge=True,blockedge=True,nexttoken=True,nextuse=True)
    logddd.log(f"len(tree) = {len(treelist)}")
    device=torch.device("cuda:0")
    model=models.GMNnet(77535,embedding_dim=100,num_layers=4,device=device).to(device)
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
            