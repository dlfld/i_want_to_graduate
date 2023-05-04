"""
这个包的作用是：
    1. 将替换好的数据进行两两配对，组成训练数据集
    2. 将组合好的数据集进行落地保存
输入的数据格式为{"className_funcName":[func_ast,]}
"""
from itertools import product
from typing import List

import joblib


def combination_func(dataset_map: dict, class_func_asts: dict) -> List[List]:
    """
        对扫描到的数据进行两两组合，组合的计算方式如下：
            1. 对于正例，从调用关系dataset_map中，依次获取到每一个被调用方法和调用方法关系
                a. 将每一个调用方法和被调用方法组合形成一条数据
                b. 对于其中的每一个调用方法列表，将列表那的数据进行两两匹配，形成多条数据。
            2. 对于负例，从调用关系dataset_map中，依次获取到每一被调用方法和调用方法关系，以及其他调用方法、关系
                将每一个被调用方法和其他方法调用（dataset_map中没有调用这个方法的方法）进行两两匹配
    @param dataset_map: 被调用方法还原之后信息的记录 {called_func:[caller_func_ast,]}
    @param class_func_asts:当前项目中所有的方法，用于检索被调用方法的ast
    @return: 数据集list
    """
    dataset_list = []
    # 对每一个相同的项进行遍历,构造正例
    for key in dataset_map.keys():
        caller_ast_list = dataset_map[key]
        # 遍历调用方法的ast，将被调用方法与调用方法进行两两组合
        for item in class_func_asts:
            # 将调用方法和被调用方法组成
            data = [class_func_asts[key], item, 1]
            dataset_list.append(data)

        # 两两组合caller_list中的ast，他们都调用了同一个方法，因此有相同的子树
        combine_list = list(product(caller_ast_list, repeat=2))
        # 结合出来没有标签，因此需要把标签加上去
        for item in combine_list:
            item = list(item) + [1]
            dataset_list.append(item)

    # 构造负例
    # 遍历调用关系map，依次获取每一个key
    for item in dataset_map.keys():
        # 调用当前方法的列表
        caller_ast_list = dataset_map[item]
        caller_ast_set = set(caller_ast_list)
        data_list = []

        # 再遍历方法调用关系map，获取到除了当前key之外的其他key对应的方法调用方法，与当前的key进行组合（因为这些方法是没有调用当前key对应的方法的）
        for key in dataset_map.keys():
            # 当前方法，当前方法列表
            if key == item:
                continue
            # 将这些调用方法添加到一个列表中,还需要满足的一点是当前方法没有在当前被调用方法的调用列表中
            data_list.extend([val for val in dataset_map[key] if val not in caller_ast_set])

        called_func = class_func_asts[item]
        # 遍历调用方法列表 生成负例列表
        for caller_func in data_list:
            # dataset_list.append()
            data = [called_func, caller_func, -1]
            dataset_list.append(data)

    return dataset_list


def dump_dataset(data: list, save_path: str):
    """
        保存数据
    @param data: 一条数据
    @param save_path: 数据保存位置
    @return:
    """
    joblib.dump(data, save_path)
