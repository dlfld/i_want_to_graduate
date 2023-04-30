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
from anytree import AnyNode
from javalang.ast import Node
from javalang.tree import MethodDeclaration


def get_token(node):
    token = ''
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__

    return token

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

def method_invoca_replace(node,class_func_asts: dict)-> Node:
    """
        检测当前节点是否为方法调用节点，
            如果是的话就对该节点进行替换
            如果不是的话就直接返回

    @param node: 节点
    @param class_func_asts: 方法map
    @return: 替换之后的节点
    """
    note_type = type(node).__name__
    # 提取静态方法调用
    if note_type == "MethodInvocation" and node.qualifier is not None:
        # qualifier 是调用对象、类名  member 是具体的方法名
        # 因为在这个地方只需要提取静态方法的使用，因此直接将qualifier的值当作是类名去找，找不到就不是。
        map_key = f"{node.qualifier}_{node.member}"
        called_func_ast = class_func_asts[map_key]
        return called_func_ast
    else:
        return node


# def get_child(root,children_list:list):
#     """
#         获取当前节点的所有子节点
#     @param root: ast节点
#     @param children_list: 子节点列表
#     @return: 节点的子节点
#     """
#
#     if isinstance(root, Node):
#         # 这个是直接获取当前节点属性值
#         children = root.children
#         children_list.append(root)
#     elif isinstance(root, set):
#         children = list(root)
#     else:
#         children = []
#
#     # 将当前节点所有子节点添加到列表中，
#     for item in children:
#         if isinstance(item,list):
#             for sub_node in item:
#                 get_child(sub_node,children_list)
#         elif item:
#             get_child(item,children_list)
def get_child(root):
    """
        获取指定节点的所有属性
    @param root:
    @return:
    """
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    #print(sub_item)
                    yield sub_item
            elif item:
                #print(item)
                yield item
    return list(expand(children))

def createtree(root, node, nodelist, parent, class_func_asts, replaced_func):
    """
     根据token和子结构创建一棵树,使用anytree
    @param root: 根节点
    @param node: 节点
    @param nodelist: 节点列表
    @param parent: 父节点
    @param class_func_asts: 类和方法对应的map
    @param replaced_func: 被替换的ast在class_func_asts中的key列表
    @return:
    """
    id = len(nodelist)
    token, children = get_token(node), get_child(node)
    if id==0:
        root.token=token
        root.data=node
    else:
        # 如果当前节点的类型是方法调用，那么就找到被调用的方法节点，进行替换
        print(token)
        if token == "MethodInvocation":
            node, token = replace_called_func(node, class_func_asts, token, replaced_func)

        newnode = AnyNode(id=id, token=token, data=node, parent=parent)

    nodelist.append(node)
    for child in children:
        if id == 0:
            createtree(root, child, nodelist, parent=root, class_func_asts=class_func_asts, replaced_func=replaced_func)
        else:
            createtree(root, child, nodelist, parent=newnode, class_func_asts=class_func_asts,
                       replaced_func=replaced_func)


def replace_called_func(node: Node, class_func_asts: Dict, token, replaced_func):
    """
     根据规则替换掉方法调用节点
    @param node: 方法调用节点
    @param class_func_asts: 类-方法映射列表
    @param token: node对应的token
    @return:替换之后的node
    """
    # node.qualifier不为空表示当前节点是xxx.xxx()调用方式的
    if node.qualifier is not None:
        # 类名
        class_name = node.qualifier
        # 方法名
        func_name = node.member
        class_func_key = f"{class_name}_{func_name}"
        print(class_func_key)
        if class_func_key in class_func_asts:
            called_func_ast = class_func_asts[class_func_key]
            # 表示当前调用方法是当前项目内部编写的方法
            if called_func_ast is not None:
                # 添加被替换节点的key
                print("进来了"+class_func_key)
                replaced_func.append(class_func_key)
                return called_func_ast, get_token(called_func_ast)
    return node, token


def func_call_replace(func_node_list: List[MethodDeclaration], class_func_asts: dict):
    """
    逐个扫描每一个方法的ast，将调用当前项目方法源码的方法调用节点替换为指定方法的ast
    在Java的ast节点中，具体方法调用节点的类型是：MethodInvocation
    在java中，方法调用的方式有如下几种：
        1. 静态类.静态方法()
            XXXX.xxxx() 大多数工具类的重用方法，工具类中工具类的使用会出现在一个项目中的不同地方，不同与一些专门的Service类（只会在上层的一个地方被调用）
            先识别这一类的方法调用并进行还原生成数据集
                因为这一类的方法会在不同的方法中被调用，可以对调用该方法的方法进行方法调用还原后进行两两组合形成较大数量的数据集
            类名为：node.qualifier
            方法名为:node.member
        2. 类变量.方法()
            类变量名:node.qualifier
            方法名:node.member
            这种情况，要找到这个变量定义的地方,简单的来说就是两个地方：
                1. 如果是spring项目，可能在类的属性里定义
                2. 普通项目就是使用java类对象获取的四种方式：
                    a. new class
                    b. clone
                    c. reflect
                    d. deserialization
                    他们的共同点就是，A a = xxxx 因此可以归为一类，就是LocalVariableDeclaration节点

        3. new 类().方法()

    @param func_node_list: 当前项目所有的方法ast
    @param class_func_asts: 当前项目所有类-方法对应列表，表示指定方法属于哪一个类
    @return: 替换之后的训练数据，其结构为
            {
                "source":"func_ast",  被调用方法的ast
                "target":[func_ast,]  调用方法的ast
            }



    """

    # 遍历每一个方法
    for func in func_node_list:
        nodelist = []
        newtree = AnyNode(id=0, token=None, data=None)
        replaced_func = []
        createtree(newtree, func, nodelist, None, class_func_asts, replaced_func=replaced_func)
        print(replaced_func)
        exit(0)


if __name__ == '__main__':
    proj_dir = "../projects/test/"
    method_ast_list, class_func_asts = get_proj_method_asts_classes(proj_dir)
    print(class_func_asts)
    func_call_replace(method_ast_list, class_func_asts)
    print(len(method_ast_list))
    print(len(class_func_asts))

