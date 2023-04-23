import os
from typing import List

import javalang
from javalang.tree import MethodDeclaration


def get_method_asts(code: str) -> List[MethodDeclaration]:
    """
        获取java文件内方法AST节点
    :param code: .java文件的code
    :return: .java文件内所有方法转换成AST的方法节点列表
    """
    method_asts = []
    # 将输入代码转换成ast树
    program_ast = javalang.parse.parse(code)
    # 遍历所有的节点
    for ast_type in program_ast.types:
        # 节点名
        node_name = type(ast_type).__name__
        # 方法应该存在于类定义中，因此需要找到类定义的标签
        if node_name == "ClassDeclaration":
            # 遍历类节点的子节点，获取到方法节点
            for body in ast_type.body:
                # 获取当前节点的节点名
                body_type = type(body).__name__
                # 如果当前节点是方法节点
                if body_type == "MethodDeclaration":
                    # 添加结果集
                    method_asts.append(body)

    return method_asts


def get_proj_method_asts(proj_dir: str) -> List[MethodDeclaration]:
    """
        扫描工程文件目录，获取工程中所有方法的AST
    :param proj_dir: 工程文件目录
    :return: 工程中所有的方法ast节点
    """
    method_ast_list = []
    # 遍历

    for rt, dirs, files in os.walk(proj_dir):
        for file in files:
            if file.endswith(".java"):
                programfile = open(os.path.join(rt, file), encoding='utf-8')
                # 读取文件
                code = programfile.read()
                # 获取方法的AST节点
                try:
                    asts = get_method_asts(code)
                    method_ast_list.extend(asts)
                except Exception as e:
                    # print(e)
                    pass
                    
                programfile.close()
    return method_ast_list


if __name__ == '__main__':
    proj_dir = "kafka"
    proj_method_asts = get_proj_method_asts(proj_dir)
    print(len(proj_method_asts))
