from typing import List, Dict, Set, Tuple, Any
import os
import joblib
from anytree import AnyNode
import logddd
from tqdm import tqdm


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

    for rt, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".data"):
                data = joblib.load(os.path.join(rt, file))
                all_data.append(data)
                # 解析出两个ast和标签,从这儿取出来的ast
                ast1, ast2, label = data
                # 遍历树获取当前树的token
                get_tree_tokens_dfs(ast1, all_tokens)
                get_tree_tokens_dfs(ast2, all_tokens)

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
        pass

    edge_index = [edgesrc, edgetgt]
    cur_tree = [x, edge_index, edge_attr]
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
    for data in all_data:
        func1, func2, label = data
        # 解析树为输入embedding数据
        func1_data = add_graph_edge(func1, vocab_size, token_dict, mode='astonly', nextsib=False,
                                    ifedge=False, whileedge=False, foredge=False, blockedge=False, nexttoken=False, nextuse=False)
        func2_data = add_graph_edge(func1, vocab_size, token_dict, mode='astonly', nextsib=False,
                                    ifedge=False, whileedge=False, foredge=False, blockedge=False, nexttoken=False, nextuse=False)

        # 形成pair数据
        pair_data = [func1_data[0], func2_data[0], func1_data[1],
                     func2_data[1], func1_data[2], func2_data[2]]
        all_pair_data.append(pair_data)

    return all_pair_data


def get_all_mes_datas():
    # 加载数据，获取token
    token_dict, vocab_size, all_data = get_all_tokens(
        "../generate_dataset/dataset/")
    # logddd.log(token_dict)
    # 创建pairdata
    all_pair_data = create_pair_data(
        all_data=all_data, token_dict=token_dict, vocab_size=vocab_size)
