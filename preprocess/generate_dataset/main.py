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
import logddd
from tqdm import tqdm
from gen_proj_msg import get_proj_method_asts_classes
from gen_dataset.rebuild_ast import func_call_replace
from gen_dataset.dump_dataset import combination_func, dump_dataset
import sys

sys.path.append("generate_dataset/")
import csv
from typing import List, Dict, Any
import numpy as np
import joblib


pre_dir = "total_data/"
def statistics_func_called(dataset_map: Dict[Any, Any], proj_name: str, types):
    """
        统计每一个被调用方法的调用次数，然后将信息写入excel中。
        @param dataset_map: 方法被调用map信息
        @param proj_name: 当前被调用方法的
        @return:
    """
    # 存储在excel中的信息，二维数组，第一列是被调用方法的类_方法名，第二列是被调用方法的调用次数
    save_list = []
    for key in dataset_map.keys():
        save_list.append([key, len(dataset_map[key])])

    save_list = sorted(save_list, key=lambda x: x[1], reverse=True)

    save_list = np.array(save_list)
    indexs = np.array([x for x in range(len(save_list))])
    # 在数据中增加索引列
    save_list = np.c_[indexs, save_list]
    # 将结果写到csv文件当中
    with open(f"{pre_dir}/{proj_name}_func_called.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        # writer.writerow(["index", "被调用方法", "被调用次数", "new class().func()", "local.func()", "OtherClass.func()","ProjClass.func()"])
        writer.writerow(["index", "被调用方法", "被调用次数"])
        # writer.writerow(types)
        writer.writerows(save_list)


def save_func_doc(java_func_doc_dict: dict,proj_name: str):
    """
     保存java方法对应的注释
    @param java_func_doc_dict: java方法名-方法注释的map
    @param proj_name: 当前工程的名字
    @return: None
    """
    with open(f"{pre_dir}/{proj_name}_doc.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index","className_funcName","doc"])
        values = [[index,key,java_func_doc_dict[key]] for index,key in enumerate(java_func_doc_dict.keys())]
        writer.writerows(values)


if __name__ == '__main__':
    # proj_name = "ktg-mes"
    # proj_dir = "../projects/mom_mes/ktg-mes/"
    proj_name = "industry4"
    proj_dir = "../projects/mom_mes/industry4.0-mes/"

    # 获取ast列表和方法对应map
    # java 方法-方法注释 字典{class_func:doc} java_func_doc_dict
    method_ast_list, class_func_asts, java_func_doc_dict = get_proj_method_asts_classes(proj_dir)

    # 保存方法注释信息
    save_func_doc(java_func_doc_dict,proj_name)

    # 调用这个方法，返回方法调用数据map,方法调用的种类
    # 第一种 new class().func()的方式，其qualifier属性为空
    # 第二种 local.func() 其qualifier属性为变量名
    # 第三种 第三方工具类 StringUtils.isEmpty() 这种，其qualifier属性为变量名为第三方类名
    # 第四种 当前项目的staticClass.func() 这种，其qualifier属性为变量名为第三方类名
    dataset_map, type1, type2, type3, type4 = func_call_replace(method_ast_list, class_func_asts)
    logddd.log("方法总共的调用次数：",len(type1),"被单独调用的次数",len(set(type1)),"被重复调用的次数",len(type1) - len(set(type1)))
    logddd.log("方法总共的调用次数：",len(type2),"被单独调用的次数",len(set(type2)),"被重复调用的次数",len(type2) - len(set(type2)))
    logddd.log("方法总共的调用次数：",len(type3),"被单独调用的次数",len(set(type3)),"被重复调用的次数",len(type3) - len(set(type3)))
    logddd.log("方法总共的调用次数：",len(type4),"被单独调用的次数",len(set(type4)),"被重复调用的次数",len(type4) - len(set(type4)))

    logddd.log("len(method_ast_list) = ", len(method_ast_list))
    logddd.log("len(class_func_asts) = ", len(class_func_asts))
    logddd.log(f"len(dataset_map) = {len(dataset_map)}")

    # 获取数据匹配列表
    dataset_list = combination_func(dataset_map, class_func_asts)
    logddd.log(len(dataset_list))
    # 将方法调用次数统计信息落地
    statistics_func_called(dataset_map, proj_name, ['', '', '', len(type1), len(type2), len(type3), len(type4)])

    # 将一个项目的数据集落地

    # 一次性落地
    # joblib.dump(dataset_list, f"{proj_name}.data")

    # 挨个挨个落地
    # for index in tqdm(range(len(dataset_list)), desc="saving"):
    #     data = dataset_list[index]
    #     dump_dataset(data, f"dataset/{proj_name}_{index}.data")

    logddd.log("DONE!")