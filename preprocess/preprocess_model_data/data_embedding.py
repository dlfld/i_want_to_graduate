from typing import List, Dict, Set, Tuple, Any
import os
import joblib
from anytree import AnyNode
# import logddd
from tqdm import tqdm
from edge_index import edges


def get_tree_tokens_dfs(tree: AnyNode, token_set: Set):
    """
    根据AnyNode获取每一个节点的token
        深度优先的递归遍历当前树的每一个节点，获取每一个节点的token
    @param tree: 根节点
    @param token_set: token 列表
    @return:
    """
    token_set.add(tree.token)
    for child in tree.children:
        # token = child.token
        # token_set.add(token)
        get_tree_tokens_dfs(child, token_set)


def get_all_tokens(dataset_path: str) -> Tuple[Dict[Any, Any], int, List[Any]]:
    """
    根据输入的数据集的位置获取数据集的所有token
    @param dataset_path: 数据的路径
    @return: 词表、词表大小、所有的数据
    """
    # 所有token的列表
    all_tokens = set([])
    all_data = []
    # 测试，限制数据量
    sum = 0

    # 读取所有的数据文件到data里面
    for rt, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".data"):
                proj_data = joblib.load(os.path.join(rt, file))
                all_data.extend(proj_data)

    # 遍历所有的数据文件，获取所有的token
    for data in tqdm(all_data,desc="get_all_tokens:"):
        # 解析出两个ast和标签,从这儿取出来的ast
        ast1, ast2, label = data
        # 遍历树获取当前树的token
        get_tree_tokens_dfs(ast1, all_tokens)
        get_tree_tokens_dfs(ast2, all_tokens)
        # 测试，限制数据量
        # sum += 1
        # if sum == 200000:
        #     break

    vocab_size = len(list(all_tokens))
    token_ids = range(vocab_size)

    return dict(zip(all_tokens, token_ids)), vocab_size, all_data


def getnodeandedge_astonly(node, nodeindexlist, vocabdict, src, tgt):
    """
        在树中加入边。
    @param node: 节点
    @param nodeindexlist: 节点转换成id之后的列表
    @param vocabdict: 词表
    @param src: 边出发节点列表
    @param tgt: 边指向节点列表
    @return:
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
    """
        在树中加入边。
    @param node: 节点
    @param nodeindexlist: 节点转换成id之后的列表
    @param vocabdict: 词表
    @param src: 边出发节点列表
    @param tgt: 边指向节点列表
    @param edgetype: 边的类型
    @return:
    """

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


def getedge_nextsib(node, vocabdict, src, tgt, edgetype):
    """
        在兄弟结点之间添加边
        @param node: 节点
        @param vocabdict: 词表
        @param src: 边出发节点列表
        @param tgt: 边指向节点列表
        @param edgetype: 边的类型
        @return:
    """
    token = node.token
    for i in range(len(node.children)-1):
        src.append(node.children[i].id)
        tgt.append(node.children[i+1].id)
        edgetype.append([1])
        src.append(node.children[i+1].id)
        tgt.append(node.children[i].id)
        edgetype.append([edges['Prevsib']])
    for child in node.children:
        getedge_nextsib(child, vocabdict, src, tgt, edgetype)


def getedge_flow(node, vocabdict, src, tgt, edgetype, ifedge=False, whileedge=False, foredge=False):
    """
        添加控制流信息
        @param node: 节点
        @param vocabdict: 词表
        @param src: 边出发节点列表
        @param tgt: 边指向节点列表
        @param edgetype: 边的类型
        @return:
    """
    token = node.token
    if whileedge == True:
        if token == 'WhileStatement':
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            edgetype.append([edges['While']])
            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
            edgetype.append([edges['While']])
    if foredge == True:
        if token == 'ForStatement':
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            edgetype.append([edges['For']])
            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
            edgetype.append([edges['For']])

    if ifedge == True:
        if token == 'IfStatement':
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            edgetype.append([edges['If']])
            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
            edgetype.append([edges['If']])
            if len(node.children) == 3:
                src.append(node.children[0].id)
                tgt.append(node.children[2].id)
                edgetype.append([edges['Ifelse']])
                src.append(node.children[2].id)
                tgt.append(node.children[0].id)
                edgetype.append([edges['Ifelse']])
    for child in node.children:
        getedge_flow(child, vocabdict, src, tgt,
                     edgetype, ifedge, whileedge, foredge)


def add_graph_edge(func_ast, vocablen, vocabdict, mode='astonly', nextsib=False, ifedge=False, whileedge=False,
                   foredge=False, blockedge=False, nexttoken=False, nextuse=False):
    """
        在ast的图中按照规则加入边
    @param vocablen: 词表大小
    @param vocabdict: 词表
    @param mode:
    @param nextsib:
    @param ifedge:
    @param whileedge:
    @param foredge:
    @param blockedge:
    @param nexttoken:
    @param nextuse:
    @return: 将当前树添加边并转换成能够输入模型的表示
    """
    x = []
    edgesrc = []
    edgetgt = []
    edge_attr = []

    if mode == 'astonly':
        # 如果只是使用ast
        getnodeandedge_astonly(func_ast, x, vocabdict, edgesrc, edgetgt)
    else:
        # logddd.log("进来了")
        getnodeandedge(func_ast, x, vocabdict, edgesrc, edgetgt, edge_attr)
        if nextsib == True:
            getedge_nextsib(func_ast, vocabdict, edgesrc, edgetgt, edge_attr)

        getedge_flow(func_ast, vocabdict, edgesrc, edgetgt,
                     edge_attr, ifedge, whileedge, foredge)

    edge_index = [edgesrc, edgetgt]
    cur_tree = [x, edge_index, edge_attr]
    # 暂时只用AST
    # cur_tree = [x, edge_index, None]
    return cur_tree


def create_pair_data(all_data: List[Any], token_dict: Dict[Any, Any], vocab_size: int) -> List[Any]:
    """
     获取所有的数据，并创建数据对
     @param all_data: 所有的数据
     @param token_dict: 词表
     @param vocab_size: 词表大小
    @return: 数据组成训练数据的pair
    """
    all_pair_data = []
    for data in tqdm(all_data, desc="all_data_process:"):
        func1, func2, label = data
        # 解析树为输入embedding数据
        func1_data = add_graph_edge(func1, vocab_size, token_dict, mode='astandnext', nextsib=True,
                                    ifedge=True, whileedge=True, foredge=True, blockedge=True, nexttoken=True, nextuse=True)
        func2_data = add_graph_edge(func1, vocab_size, token_dict, mode='astandnext', nextsib=True,
                                    ifedge=True, whileedge=True, foredge=True, blockedge=True, nexttoken=True, nextuse=True)

        # 形成pair数据
        pair_data = [func1_data[0], func2_data[0], func1_data[1],
                     func2_data[1], func1_data[2], func2_data[2], label]
        all_pair_data.append(pair_data)

    return all_pair_data


def get_all_mes_datas(dataset_path: str) -> List[Any]:
    """
    读取所有数据，对数据格式进行转换并embedding
    @return: 返回所有数据经过embedding之后的pair,词表大小,词表
    """
    # 加载数据，获取token
    token_dict, vocab_size, all_data = get_all_tokens(dataset_path)

    # 创建pairdata
    all_pair_data = create_pair_data(
        all_data=all_data, token_dict=token_dict, vocab_size=vocab_size)

    # logddd.log(len(all_pair_data))
    return all_pair_data, token_dict


def load_mom_data(args):
    """
        返回MOM&MES数据,
        @return: 所有数据的列表，词表
    """
    data_file_name = "mom_data.data"
    # data_file_path = "../generate_dataset/dataset/"
    data_file_path = "../generate_dataset/total_data/"
    # 加载数据
    if not os.path.exists(data_file_name):
        all_data_list, token_dict = get_all_mes_datas(data_file_path)
        temp_data = {
            "all_data_list": all_data_list,
            "token_dict": token_dict
        }
        joblib.dump(temp_data, data_file_name)
    else:
        temp_data = joblib.load(data_file_name)
        all_data_list = temp_data["all_data_list"]
        token_dict = temp_data["token_dict"]

    print("load mom data compelete!")
    return all_data_list, token_dict


if __name__ == "__main__":
    all_pair_data, token_dict = get_all_mes_datas(
        "../generate_dataset/dataset/")
    temp_data = {
        "all_pair_data": all_pair_data,
        "token_dict": token_dict
    }
    joblib.dump(temp_data, "mom_data.data")
