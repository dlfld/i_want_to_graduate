from typing import List

import logddd
import torch
import models
from utils.ast_handle_utils import createast, create_separate_graph
import javalang
from javalang.tree import MethodDeclaration
def preprocess(dir_name:str)->List[MethodDeclaration]:
    proj_method_asts, vocabsize, vocabdict = createast(dir_name)
    logddd.log(len(proj_method_asts))
    tree_list = create_separate_graph(proj_method_asts, vocabsize, vocabdict, device='cpu', mode='else', nextsib=True,
                                     ifedge=True, whileedge=True, foredge=True, blockedge=True, nexttoken=True,
                                     nextuse=True)
    logddd.log(f"len(tree) = {len(tree_list)}")
    return tree_list


if __name__ == "__main__":
    dir_name = "projects/kafka/"
    tree_list = preprocess(dir_name)