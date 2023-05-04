from typing import List, Dict

from anytree import AnyNode
from javalang.ast import Node
from javalang.tree import MethodDeclaration
"""
    这个包的用处是：重新构建方法的抽象语法树。
    具体工作就是将被调用方法的方法体还原到原方法中。
"""

def get_token(node):
    """
     获取当前节点的token
    @param node: 节点
    @return: token
    """
    token = ''
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__

    return token

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
                    # print(sub_item)
                    yield sub_item
            elif item:
                # print(item)
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
    @param replaced_func: 被替换的ast在class_func_asts中的key列表，也就是当前方法调用了那些方法
    @return:
    """
    id = len(nodelist)
    token, children = get_token(node), get_child(node)

    if id == 0:
        root.token = token
        root.data = node
    else:
        # print(node)
        # 如果当前节点的类型是方法调用，那么就找到被调用的方法节点，进行替换
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
     根据规则替换掉方法调用节点，并将节点替换进行记录
      调用的方式有很多种，目前只是识别了静态方法调用（这是重用出现频率最高的一种）
    @param node: 方法调用节点
    @param class_func_asts: 类-方法映射列表
    @param token: node对应的token
    @param replaced_func: 被调用方法的列表，也就是
    @return:替换之后的node
    """
    # node.qualifier不为空表示当前节点是xxx.xxx()调用方式的
    if node.qualifier is not None:
        # 类名
        class_name = node.qualifier
        # 方法名
        func_name = node.member
        class_func_key = f"{class_name}_{func_name}"
        # 判断当前调用的方法是不是当前工程中的方法
        if class_func_key in class_func_asts.keys():
            called_func_ast = class_func_asts[class_func_key]
            # 表示当前调用方法是当前项目内部编写的方法
            if called_func_ast is not None:
                # 添加被替换节点的key
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
                called_func_id:[call_func_ast,]
            }
    """

    dataset_map = dict()
    # 遍历每一个方法
    for func in func_node_list:
        nodelist = []
        new_tree = AnyNode(id=0, token=None, data=None)
        # 这个列表记录的是当前方法节点进行方法调用节点替换的时候，替换了那些方法调用进去
        replaced_func = []
        # 对当前方法的抽象语法树进行重构，在重构过程中进行方法调用节点的替换
        createtree(new_tree, func, nodelist, None, class_func_asts, replaced_func=replaced_func)
        # 一个方法可能调用了多个方法，因此，需要使用一个列表来记录被调用方法是被那些方法调用了
        for item in replaced_func:
            if item in dataset_map.keys():
                dataset_map[item].append(new_tree)
            else:
                dataset_map[item] = [new_tree]
    return dataset_map
