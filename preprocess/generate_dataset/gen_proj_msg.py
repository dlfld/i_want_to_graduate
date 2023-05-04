import os
from typing import List, Dict

import javalang
from javalang.tree import MethodDeclaration


def get_file_method_asts_classes(code: str, method_ast_list: list, class_func_asts: Dict):
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
                    class_func_asts[dict_key] = body
                    # 将方法的ast节点添加到ast节点列表中
                    method_ast_list.append(body)


# def get_proj_method_asts_classes(proj_dir: str) -> tuple[List[MethodDeclaration], Dict]:
def get_proj_method_asts_classes(proj_dir: str):
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