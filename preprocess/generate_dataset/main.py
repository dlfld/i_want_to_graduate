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
from generate_dataset.gen_dataset.dump_dataset import combination_func, dump_dataset
from generate_dataset.gen_dataset.rebuild_ast import func_call_replace
from generate_dataset.gen_proj_msg import get_proj_method_asts_classes


if __name__ == '__main__':
    proj_dir = "../projects/mom_mes/ktg-mes/"
    # 获取ast列表和方法对应map
    method_ast_list, class_func_asts = get_proj_method_asts_classes(proj_dir)
    # print(class_func_asts)
    # 调用这个方法，返回方法调用数据map
    dataset_map = func_call_replace(method_ast_list, class_func_asts)

    print("len(method_ast_list) = ", len(method_ast_list))
    print("len(class_func_asts) = ", len(class_func_asts))
    print(f"len(dataset_map) = {len(dataset_map)}")
    # for key in dataset_map.keys():
    #     print(f"func_name={key},len(key) = {len(dataset_map[key])}")

    # 获取数据匹配列表
    dataset_list = combination_func(dataset_map, class_func_asts)
    for index, data in enumerate(dataset_list):
        dump_dataset(data, f"dataset/{index}.data")
