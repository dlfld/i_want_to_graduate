import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GatedGraphConv
from torch_geometric.utils import degree, remove_self_loops, add_self_loops, softmax, scatter
# from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.glob import GlobalAttention
import sys
import inspect
from param_parser import get_args
from torch_geometric.nn import GCNConv


is_python2 = sys.version_info[0] < 3
getargspec = inspect.getargspec if is_python2 else inspect.getfullargspec
special_args = [
    'edge_index', 'edge_index_i', 'edge_index_j', 'size', 'size_i', 'size_j'
]
__size_error_msg__ = ('All tensors which should get mapped to the same source '
                      'or target nodes must be of same size in dimension 0.')

# ==========================================================在最后加MLP，将整个计算相似度问题转换成为二分类问题===========================================================


def dense_layer(
        inp: int,
        out: int,
        activation: str,
        p: float,
        bn: bool,
        linear_first: bool,
):
    if activation == "relu":
        act_fn = nn.ReLU(inplace=True)

    layers = [nn.BatchNorm1d(out if linear_first else inp)] if bn else []
    if p != 0:
        layers.append(nn.Dropout(p))  # type: ignore[arg-type]
    lin = [nn.Linear(inp, out, bias=not bn), act_fn]
    layers = lin + layers if linear_first else layers + lin
    return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(
            self,
            d_hidden,
            activation: str,
            dropout: 0,
            batchnorm: bool,
            batchnorm_last: bool,
            linear_first: bool,

    ):
        super(MLP, self).__init__()

        if not dropout:
            dropout = [0.0] * len(d_hidden)
        elif isinstance(dropout, float):
            dropout = [dropout] * len(d_hidden)

        self.mlp = nn.Sequential()
        for i in range(1, len(d_hidden)):
            self.mlp.add_module(
                "dense_layer_{}".format(i - 1),
                dense_layer(
                    d_hidden[i - 1],
                    d_hidden[i],
                    activation,
                    dropout[i - 1],
                    batchnorm and (i != len(d_hidden) - 1 or batchnorm_last),
                    linear_first,
                ),
            )

    def forward(self, X):
        return self.mlp(X)


class DeepSim(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
# ======================================================attention =============================================================================
        # self.attention = AttentionModule()
# ======================================================attention =============================================================================
        self.mlp1 = MLP(  # 定义的全连接层
            d_hidden=[200, 2 * 200, 200],  # 全连接层数
            activation='relu',  # 激活函数
            dropout=0,  # dropout
            batchnorm=False,
            batchnorm_last=False,
            linear_first=True,
        )

        self.mlp2 = MLP(  # 定义的全连接层
            d_hidden=[200, 2 * 200, 200],  # 全连接层数
            activation='relu',  # 激活函数
            dropout=0,  # dropout
            batchnorm=False,
            batchnorm_last=False,
            linear_first=True,
        )

        # 如果添加了hist 需要将hist的长度添加上去
        self.liner = nn.Linear(200, 2)

    def forward(self, x1, x2, hist):

        # x1 = self.mlp1(x1)  # 经过全连接层特征提取x1
        # x2 = self.mlp1(x2)  # 经过全连接层特征提取x2

        x12 = torch.cat([x1, x2], 1)  # 拼接向量
        x21 = torch.cat([x2, x1], 1)
        # print(f"x12.shape = {x12.shape}")
# ======================================================添加attention ================================================================================
        # x12 = self.attention(x12)
        # x21 = self.attention(x21)
        # print(f"x12.shape = {x12.shape}")
        # print()
# ======================================================添加attention ================================================================================

        h1 = self.mlp1(x12)
        h2 = self.mlp2(x21)

        h = torch.cat((h1, h2), 0)
        # print(f"求mean之前的shape  h.shape={h.shape}")
        h = torch.mean(h, keepdim=True, dim=0)  # 合并x1x2
        # print(f"求mean之后的shape  h.shape={h.shape}")
        # logddd.log(h.shape)
        # h = torch.cat((h,hist),1)

        # logddd.log(h.shape)
        # print(h)
        # 要输入是*200
        logits = self.liner(h)

        return logits

# ==========================================================在最后加MLP，将整个计算相似度问题转换成为二分类问题===========================================================


# ==========================================================attention 层==========================================================
class AttentionModule(torch.nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """

    def __init__(self):
        super(AttentionModule, self).__init__()
        # 获取参数
        self.args = get_args()
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(200,
                                                             1).cuda())

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector.
        """
        global_context = torch.mean(torch.matmul(
            self.weight_matrix, embedding), dim=0)

        transformed_global = torch.tanh(global_context)
        sigmoid_scores = torch.sigmoid(
            torch.mm(embedding, transformed_global.view(-1, 1)))

        representation = torch.mm(sigmoid_scores, torch.t(embedding).T)
        return representation
# ==========================================================attention 层==========================================================


class GMNlayer(MessagePassing):
    def __init__(self, in_channels, out_channels, device):
        super(GMNlayer, self).__init__(aggr='add')  # "Add" aggregation.
        self.device = device
        self.out_channels = out_channels
        self.fmessage = nn.Linear(3*in_channels, out_channels)
        self.fnode = torch.nn.GRUCell(2*out_channels, out_channels, bias=True)
        self.__match_args__ = getargspec(self.match)[0][1:]
        self.__special_match_args__ = [(i, arg)
                                       for i, arg in enumerate(self.__match_args__)
                                       if arg in special_args]
        self.__match_args__ = [
            arg for arg in self.__match_args__ if arg not in special_args
        ]

    '''def propagate(self, edge_index, size=None, **kwargs):
        size = [None, None] if size is None else list(size)
        assert len(size) == 2

        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {"_i": i, "_j": j}

        message_args = []
        for arg in self.__message_args__:
            #print(arg)
            if arg[-2:] in ij.keys():
                tmp = kwargs.get(arg[:-2], None)
                if tmp is None:  # pragma: no cover
                    message_args.append(tmp)
                else:
                    idx = ij[arg[-2:]]
                    if isinstance(tmp, tuple) or isinstance(tmp, list):
                        assert len(tmp) == 2
                        if tmp[1 - idx] is not None:
                            if size[1 - idx] is None:
                                size[1 - idx] = tmp[1 - idx].size(0)
                            if size[1 - idx] != tmp[1 - idx].size(0):
                                raise ValueError(__size_error_msg__)
                        tmp = tmp[idx]

                    if size[idx] is None:
                        size[idx] = tmp.size(0)
                    if size[idx] != tmp.size(0):
                        raise ValueError(__size_error_msg__)

                    tmp = torch.index_select(tmp, 0, edge_index[idx])
                    message_args.append(tmp)
            else:
                message_args.append(kwargs.get(arg, None))

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        kwargs['edge_index'] = edge_index
        kwargs['size'] = size

        for (idx, arg) in self.__special_args__:
            if arg[-2:] in ij.keys():
                message_args.insert(idx, kwargs[arg[:-2]][ij[arg[-2:]]])
            else:
                message_args.insert(idx, kwargs[arg])

        update_args = [kwargs[arg] for arg in self.__update_args__]
        out = self.message(*message_args)
        out = scatter_(self.aggr, out, edge_index[i], dim_size=size[i])
        #print(out.size())
        out = self.update(out, *update_args)
        return out'''

    def propagate_match(self, edge_index, size=None, **kwargs):
        size = [None, None] if size is None else list(size)
        assert len(size) == 2

        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {"_i": i, "_j": j}

        match_args = []
        # print(self.__special_match_args__)
        # print(self.__match_args__)
        # print(ij.keys())
        for arg in self.__match_args__:
            # print(arg)
            # print(arg[-2:])
            if arg[-2:] in ij.keys():
                tmp = kwargs.get(arg[:-2], None)
                if tmp is None:  # pragma: no cover
                    match_args.append(tmp)
                else:
                    idx = ij[arg[-2:]]
                    if isinstance(tmp, tuple) or isinstance(tmp, list):
                        assert len(tmp) == 2
                        if tmp[1 - idx] is not None:
                            if size[1 - idx] is None:
                                size[1 - idx] = tmp[1 - idx].size(0)
                            if size[1 - idx] != tmp[1 - idx].size(0):
                                raise ValueError(__size_error_msg__)
                        tmp = tmp[idx]

                    if size[idx] is None:
                        size[idx] = tmp.size(0)
                    if size[idx] != tmp.size(0):
                        raise ValueError(__size_error_msg__)

                    tmp = torch.index_select(tmp, 0, edge_index[idx])
                    match_args.append(tmp)
                # print(tmp)
            else:
                match_args.append(kwargs.get(arg, None))

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        kwargs['edge_index'] = edge_index
        kwargs['size'] = size

        for (idx, arg) in self.__special_match_args__:
            if arg[-2:] in ij.keys():
                match_args.insert(idx, kwargs[arg[:-2]][ij[arg[-2:]]])
            else:
                match_args.insert(idx, kwargs[arg])

        update_args = [kwargs[arg] for arg in self.__update_args__]
        # print(match_args)
        out_attn = self.match(*match_args)
        # print(out_attn.size())
        out_attn = scatter_(self.aggr, out_attn,
                            edge_index[i], dim_size=size[i])
        # print(out_attn.size())
        out_attn = self.update(out_attn, *update_args)
        # out=torch.cat([out,out_attn],dim=1)
        # print(out.size())
        return out_attn

    def forward(self, x1, x2, edge_index1, edge_index2, edge_weight1, edge_weight2, mode='train'):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # Step 1: Add self-loops to the adjacency matrix.
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Step 2: Linearly transform node feature matrix.
        # x = self.lin(x)

        # Step 3-5: Start propagating messages.
        m1 = self.propagate(edge_index1, size=(
            x1.size(0), x1.size(0)), x=x1, edge_weight=edge_weight1)
        m2 = self.propagate(edge_index2, size=(
            x2.size(0), x2.size(0)), x=x2, edge_weight=edge_weight2)

        # 让矩阵相乘
        scores = torch.mm(x1, x2.t())
        attn_1 = F.softmax(scores, dim=1)
        # print(attn_1.size())
        attn_2 = F.softmax(scores, dim=0).t()
        # print(attn_2.size())
        attnsum_1 = torch.mm(attn_1, x2)
        attnsum_2 = torch.mm(attn_2, x1)
        '''if mode!='train':
            print(attn_1)
            torch.save(attn_1,'attns/'+mode+'_attn1')
            print(attn_1.size())
            torch.save(attn_2, 'attns/' + mode + '_attn2')'''
        # print(attn_2)
        # print(attn_2.size())
        # print(attnsum_1.size())
        # print(attnsum_2.size())
        u1 = x1-attnsum_1
        u2 = x2-attnsum_2
        # u=self.propagate_match(edge_index_attn,size=(x1.size(0), x2.size(0)),x=(x1,x2))
        # print('u',u.size())
        m1 = torch.cat([m1, u1], dim=1)
        h1 = self.fnode(m1, x1)
        m2 = torch.cat([m2, u2], dim=1)
        h2 = self.fnode(m2, x2)

        return h1, h2

    def message(self, x_i, x_j, edge_index, size, edge_weight=None):
        # x_j has shape [E, out_channels]
        # Step 3: Normalize node features.
        # print(x_i.size(),x_j.size())
        if type(edge_weight) == type(None):
            edge_weight = torch.ones(x_i.size(0), x_i.size(1)).to(self.device)
            m = F.relu(self.fmessage(
                torch.cat([x_i, x_j, edge_weight], dim=1)))
        else:
            m = F.relu(self.fmessage(
                torch.cat([x_i, x_j, edge_weight], dim=1)))
        return m

    def match(self, edge_index_i, x_i, x_j, size_i):
        return
    '''def match(self, edge_index_i, x_i, x_j, size_i):
        #x_j = x_j.view(-1, 1, self.out_channels)
        #alpha = torch.dot(x_i, x_j)
        #print(edge_index_i.size())
        #print(x_i.size(),x_j.size())
        alpha=torch.sum(x_i*x_j, dim=1)
        #alpha=torch.bmm(x_i.unsqueeze(1), x_j.unsqueeze(2))
        #print(alpha.size())
        size_i=x_i.size(0)
        alpha = softmax(alpha, edge_index_i, size_i)
        #print(alpha.size())
        c = torch.ones(A, B) * 2
        v = torch.randn(A, B, C)
        print(c)
        print(v)
        print(c[:,:, None].size())
        d = c[:,:, None] * v
        return alpha[:,None]*x_j
        #return x_j* alpha.view(-1, 1, 1)
        #return (x_i-x_j)* alpha.view(-1, 1, 1)'''

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        # Step 5: Return new node embeddings.
        return aggr_out


class GMNnet(torch.nn.Module):
    def __init__(self, vocablen, embedding_dim, num_layers, device):
        super(GMNnet, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocablen, embedding_dim)
        self.edge_embed = nn.Embedding(20, embedding_dim)
        # self.gmn=nn.ModuleList([GMNlayer(embedding_dim,embedding_dim) fsor i in range(num_layers)])
        self.gmnlayer = GMNlayer(embedding_dim, embedding_dim, self.device)
        self.mlp_gate = nn.Sequential(
            nn.Linear(embedding_dim, 1), nn.Sigmoid())
        self.pool = GlobalAttention(gate_nn=self.mlp_gate)
# ======================================================添加Transformer===========================================================================
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4,batch_first=True)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
# ======================================================添加后面的神经网络===========================================================================
        self.deep_sim = DeepSim()
# ======================================================添加后面的神经网络===========================================================================
        self.args = get_args()
# ======================================================GCN==========================================================
        # self.num_features = 100
        self.convolution_1 = GCNConv(100, 256)
        self.convolution_2 = GCNConv(256, 128)
        self.convolution_3 = GCNConv(128, 100)

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Absstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_3(features, edge_index)
        return features
# ======================================================GCN==========================================================

# =====================================================Pairwise Node Comparison======================================
    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histsogram of similarity scores.
        """
        scores = torch.mm(abstract_features_1, torch.t(
            abstract_features_2)).detach()
        scores = scores.view(-1, 1)
        # torch.histc() 是一个用于计算张量中元素在各个区间内的频率的函数。
        hist = torch.histc(scores, bins=16)
        hist = hist/torch.sum(hist)
        hist = hist.view(1, -1)
        return hist
# =====================================================Pairwise Node Comparison======================================

    def forward(self, data, mode='train'):
        x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2 = data
        # 将稠密矩阵转换为邻接矩阵，然后输入到GCN中

        # x1 和x2都是点的集合，每一个点被embedding为了100d，因此x1 和 x2 的维度是 n * 100
        x1 = self.embed(x1)
        x1 = x1.squeeze(1)

        x2 = self.embed(x2)
        x2 = x2.squeeze(1)
# ======================================================添加GCN=======================================================
        # x1 = self.convolutional_pass(edge_index=edge_index1, features = x1)
        # x2 = self.convolutional_pass(edge_index=edge_index2, features = x2)

# ======================================================添加GCN=======================================================


# ===================================================== 添加 Pairwise Node Comparison======================================
        # hist = self.calculate_histogram(x1, x2)
# ===================================================== 添加 Pairwise Node Comparison======================================


# ======================================================添加transformer encoder===========================================================================
        # x1 = x1.unsqueeze(0)
        # x2 = x2.unsqueeze(0)
        # x1 = self.transformer_encoder(x1)
        # x2 = self.transformer_encoder(x2)
        # x1 = x1.squeeze(0)
        # x2 = x2.squeeze(0)

# ======================================================添加transformer encoder===========================================================================

        if type(edge_attr1) == type(None):
            edge_weight1 = None
            edge_weight2 = None
        else:
            edge_weight1 = self.edge_embed(edge_attr1)
            edge_weight1 = edge_weight1.squeeze(1)
            edge_weight2 = self.edge_embed(edge_attr2)
            edge_weight2 = edge_weight2.squeeze(1)

        for i in range(self.num_layers):
            x1, x2 = self.gmnlayer.forward(
                x1, x2, edge_index1, edge_index2, edge_weight1, edge_weight2, mode='train')
            '''if i==self.num_layers-1:
                x1,x2=self.gmnlayer.forward(x1,x2 ,edge_index1, edge_index2,edge_weight1,edge_weight2,mode=mode)
            else:
                x1, x2 = self.gmnlayer.forward(x1, x2, edge_index1, edge_index2, edge_weight1, edge_weight2, mode='train')'''

        batch1 = torch.zeros(x1.size(0), dtype=torch.long).to(
            self.device)  # without batching
        batch2 = torch.zeros(x2.size(0), dtype=torch.long).to(self.device)
        # print(f"batch1 size = {batch1.shape}")

        hg1 = self.pool(x1, batch=batch1)
        # print(f"hg1 size = {hg1.shape}")

        hg2 = self.pool(x2, batch=batch2)

#
        # logits = self.deep_sim(hg1,hg2,hist)
        logits = self.deep_sim(hg1, hg2, None)
# F.log_softmax(logits, dim=1)
        return logits


class GGNN(torch.nn.Module):
    def __init__(self, vocablen, embedding_dim, num_layers, device):
        super(GGNN, self).__init__()
        self.device = device
        # self.num_layers=num_layers
        self.embed = nn.Embedding(vocablen, embedding_dim)
        self.edge_embed = nn.Embedding(20, embedding_dim)
        # self.gmn=nn.ModuleList([GMNlayer(embedding_dim,embedding_dim) for i in range(num_layers)])
        self.ggnnlayer = GatedGraphConv(embedding_dim, num_layers)
        self.mlp_gate = nn.Sequential(
            nn.Linear(embedding_dim, 1), nn.Sigmoid())
        self.pool = GlobalAttention(gate_nn=self.mlp_gate)

    def forward(self, data):
        x, edge_index, edge_attr = data
        x = self.embed(x)
        x = x.squeeze(1)
        if type(edge_attr) == type(None):
            edge_weight = None
        else:
            edge_weight = self.edge_embed(edge_attr)
            edge_weight = edge_weight.squeeze(1)
        x = self.ggnnlayer(x, edge_index)
        batch = torch.zeros(x.size(0), dtype=torch.long).to(self.device)
        hg = self.pool(x, batch=batch)
        return hg
