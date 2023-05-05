from typing import List, Dict, Set
import os
import joblib
from anytree import AnyNode


def get_tree_tokens_dfs(tree: AnyNode, token_set: Set):
    """
    根据AnyNode获取每一个节点的token
        深度优先的递归遍历当前树的每一个节点，获取每一个节点的token
    @param tree: 根节点
    @param token_set: token 列表
    @return:
    """

    for child in tree.children:
        token = child.token
        token_set.add(token)
        get_tree_tokens_dfs(child,token_set)



def get_all_tokens(dataset_path: str) -> Dict:
    """
    根据输入的数据集的位置获取数据集的所有token
    @param dataset_path: 数据的路径
    @return: token-id 对应的列表
    """
    # 所有token的列表
    all_tokens = set([])

    for rt, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".data"):
                data = joblib.load(os.path.join(rt, file))
                # 解析出两个ast和标签,从这儿取出来的ast
                ast1, ast2, label = data
                # 遍历树获取当前树的token
                get_tree_tokens_dfs(ast1,all_tokens)
                get_tree_tokens_dfs(ast2,all_tokens)
    
    print(len(list(all_tokens)))
    print(all_tokens)
    


if __name__ == "__main__":
    get_all_tokens("../generate_dataset/dataset/")
