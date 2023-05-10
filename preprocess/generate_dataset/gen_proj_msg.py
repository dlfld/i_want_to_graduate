import os
from typing import List, Dict,Any

import javalang
from javalang.tree import MethodDeclaration
from gen_dataset.rebuild_ast import get_token, get_child
from anytree import AnyNode


def add_func_doc(func_node:MethodDeclaration,class_name:str):
    # 方法的注释
    func_doc = func_node.documentation
    # 方法名
    func_name = func_node.name
    # 类-方法 字典的key
    dict_key = f"{class_name}_{func_name}"



def get_file_method_asts_classes(code: str, method_ast_list: list, class_func_asts: Dict,java_func_doc_dict:Dict[Any,Any]):
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
    @param java_func_doc_dict: java 方法-方法注释 字典{class_func:doc}
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
                    # 方法名
                    func_name = body.name
                    # 方法的注释
                    func_doc = body.documentation
                    # 类-方法 字典的key
                    dict_key = f"{class_name}_{func_name}"

                    # 将需要加入class_func dict的ast使用anytree进行重构,在后面构建数据集的时候需要重构来统一格式
                    nodelist = []
                    new_tree = AnyNode(id=0, token=None, data=None)
                    # 重构树
                    create_tree(new_tree, body, nodelist, None, )
                    # 存储重构之后的tree
                    class_func_asts[dict_key] = new_tree
                    # 将方法的ast节点添加到ast节点列表中
                    method_ast_list.append(body)

                    if func_doc is not None:
                        # 如果当前方法的注释不为空（满足要求）则将当前的方法和对应的注释关联到一起
                        """
                            处于方法上方的注释才会被认为是方法注释，并且方法注释的格式是/*xxx*/格式的，其他格式不行
                        """
                        java_func_doc_dict[dict_key] = func_doc


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
    # java 方法-方法注释 字典{class_func:doc}
    java_func_doc_dict = dict({})

    # 遍历项目目录下所有的.java文件
    for rt, dirs, files in os.walk(proj_dir):
        for file in files:
            if file.endswith(".java"):
                java_file = open(os.path.join(rt, file), encoding='utf-8')
                # 读取文件
                code = java_file.read()
                # 获取方法的ast节点 和 类
                try:
                    get_file_method_asts_classes(code, method_ast_list, class_func_asts, java_func_doc_dict)
                except Exception as e:
                    pass

    return method_ast_list, class_func_asts, java_func_doc_dict


def create_tree(root, node, nodelist, parent):
    """
     将javalang parse出来的ast使用anytree进行重构
    @param root: 根节点
    @param node: 节点
    @param nodelist: 节点列表
    @param parent: 父节点
    @return:
    """
    id = len(nodelist)
    token, children = get_token(node), get_child(node)

    if id == 0:
        root.token = token
        root.data = node
    else:
        newnode = AnyNode(id=id, token=token, data=node, parent=parent)

    nodelist.append(node)

    for child in children:
        if id == 0:
            create_tree(root, child, nodelist, parent=root)
        else:
            create_tree(root, child, nodelist, parent=newnode)
