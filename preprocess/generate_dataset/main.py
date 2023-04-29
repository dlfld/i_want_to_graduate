"""
    我当前要做的任务是
        生成MOM的数据集
    1. 首先根据AST的规则，提取类和方法节点，将方法归类到类下面。形成一个Map其结构为：{className:[methodAST,]}
        同时生成方法节点列表，将方法节点统一放到一个列表里面 其结构为[methodAST, ]
    2. 遍历方法节点列表中每一个方法的AST，将方法中的方法调用节点获取到。
    3. 寻找到方法调用节点具体调用的是哪一个类的哪一个方法
    4. 将方法调用节点替换为被调用方法的AST节点
    5. 某个方法调用了指定的方法，进行替换之后需要有个列表来记录下来，作为有监督训练的训练集
        其结构暂时设置为这样的
            {
                "source":"func_ast",  被调用方法的ast
                "target":[func_ast,]  调用方法的ast
            }

"""
import os
from typing import List, Dict

import javalang.parse
from javalang.tree import MethodDeclaration

from utils.proj_read_utils import get_proj_method_asts


def get_file_method_asts_classes(code: str, method_ast_list: list, class_func_asts: dict):
    """
    根据输入Java代码，解析出当前java文件中的类-方法 对应关系和方法AST列表
    类-方法对应关系结构为：
        {
            class-name+func-name：func-ast
        }
    方法ast列表对应的结构为：[func-ast]
    @param code: 一个.java文件的代码
    @param method_ast_list: 方法ast列表
    @param class_func_asts: class-func对应关系map
    """
    # 将java代码转换成ast
    code_ast = javalang.parse.parse(code)
    print(code_ast)
    exit(0)

    # 遍历所有的语法节点
    for ast_type in code_ast.types:
        # 节点名
        node_name = type(ast_type).__name__
        # 方法应该存在于类定义中，因此需要找到类定义的标签
        if node_name == "ClassDeclaration":
            # 获取到当前类的类名
            class_name = ast_type.name

            # 遍历类节点的子节点，获取到方法节点
            for body in ast_type.body:
                # 获取当前节点的节点名
                body_type = type(body).__name__
                # 如果当前节点是方法节点
                if body_type == "MethodDeclaration":
                    func_name = body.name
                    # 类-方法 字典的key
                    dict_key = f"{class_name}_{func_name}"
                    # 将结果添加到字典中
                    class_func_asts[dict_key] = body
                    # 将方法的ast节点添加到ast节点列表中
                    method_ast_list.append(body)


def get_proj_method_asts_classes(proj_dir: str) -> tuple[List[MethodDeclaration], Dict]:
    """
        根据输入的项目位置，解析出当前项目的类-方法 对应关系和方法AST列表
        类-方法对应关系结构为：
            {
                class-name+func-name：func-ast
            }
        方法ast列表对应的结构为：[func-ast]
    """
    # 方法ast列表
    method_ast_list = list([])
    # 类-方法 对应列表
    class_func_asts = dict({})

    # 遍历项目目录下所有的.java文件
    for rt, dirs, files in os.walk(proj_dir):
        for file in files:
            if file.endswith(".java"):
                java_file = open(os.path.join(rt, file), encoding='utf-8')
                # 读取文件
                code = java_file.read()
                # 获取方法的ast节点 和 类
                try:
                    get_file_method_asts_classes(code, method_ast_list, class_func_asts)
                except Exception as e:
                    pass

    return method_ast_list, class_func_asts


def func_call_replace(func_node_list: List[MethodDeclaration], class_func_asts: dict):
    """
        逐个扫描每一个方法的ast，将调用当前项目方法源码的方法调用节点替换为指定方法的ast
    @param func_node_list: 当前项目所有的方法ast
    @param class_func_asts: 当前项目所有类-方法对应列表，表示指定方法属于哪一个类
    @return: 替换之后的训练数据，其结构为
            {
                "source":"func_ast",  被调用方法的ast
                "target":[func_ast,]  调用方法的ast
            }
    在Java的ast节点中，具体方法调用节点的类型是：MethodInvocation
    在java中，方法调用的方式有如下几种：
        1. 静态类.静态方法()
            类名为：node.qualifier
            方法名为:node.member
        2. 类变量.方法()
            类变量名:node.qualifier
            方法名:node.member
            这种情况，要找到这个变量定义的地方

        3. new 类().方法()

    """
    # 遍历每一个方法
    for func in func_node_list:
        pass


if __name__ == '__main__':
    proj_dir = "../projects/mom_mes/ktg-mes/"
    method_ast_list, class_func_asts = get_proj_method_asts_classes(proj_dir)

    print(len(method_ast_list))
    print(len(class_func_asts))
