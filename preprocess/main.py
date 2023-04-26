from typing import List

import logddd
import torch
from utils.ast_handle_utils import createast, create_separate_graph
from utils.proj_read_utils import get_proj_method_asts
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
    # dir_name = "projects/qqrobot_mirai/"
    # dir_name = "projects/class-bot/"
    # dir_name = "projects/kafka/"
    dir_name = "projects/test/"
    # dir_name = "projects/dubbo/"
    # dir_name = "projects/mom_mes/"
    tree_list = preprocess(dir_name)
    print(len(tree_list))

    # proj_method_asts = get_proj_method_asts(dir_name)
    # print(proj_method_asts)