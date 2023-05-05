"""
    我当前要做的任务是
        生成MOM的数据集
    1. 首先根据AST的规则，提取类和方法节点，将方法归类到类下面。形成一个Map其结构为：{className_funcName:methodAST}
        同时生成方法节点列表，将方法节点统一放到一个列表里面 其结构为[methodAST, ]
    2. 遍历方法节点列表中每一个方法的AST，将方法中的方法调用节点获取到。
    3. 寻找到方法调用节点具体调用的是哪一个类的哪一个方法
    4. 将方法调用节点替换为被调用方法的AST节点
    5. 某个方法调用了指定的方法，进行替换之后需要有个列表来记录下来，作为有监督训练的训练集
        其结构暂时设置为这样的
            {
                called_func_id:[call_func_ast,]
            }
"""
import sys
sys.path.append("generate_dataset/")

from gen_dataset.dump_dataset import combination_func, dump_dataset
from gen_dataset.rebuild_ast import func_call_replace
from gen_proj_msg import get_proj_method_asts_classes
from tqdm import tqdm
import logddd
if __name__ == '__main__':
    proj_name = "ktg-mes"
    proj_dir = "../projects/mom_mes/ktg-mes/"
    # proj_name = "industry4"
    # proj_dir = "../projects/mom_mes/industry4.0-mes/"
    # 获取ast列表和方法对应map
    method_ast_list, class_func_asts = get_proj_method_asts_classes(proj_dir)
    # print(class_func_asts)
    # 调用这个方法，返回方法调用数据map
    dataset_map = func_call_replace(method_ast_list, class_func_asts)

    logddd.log("len(method_ast_list) = ", len(method_ast_list))
    logddd.log("len(class_func_asts) = ", len(class_func_asts))
    logddd.log(f"len(dataset_map) = {len(dataset_map)}")

    # 获取数据匹配列表
    dataset_list = combination_func(dataset_map, class_func_asts)
    logddd.log(len(dataset_list))
    # sum = 0
    # for key in dataset_map.keys():
    #     # sum += len(dataset_map[key])
    #     print(f"func_name={key},len(key) = {len(dataset_map[key])}")
    # logddd.log(sum)

    for index in tqdm(range(len(dataset_list)),desc="saving"):
        data = dataset_list[index]
        dump_dataset(data, f"dataset/{proj_name}_{index}.data")
