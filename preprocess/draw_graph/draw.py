from graphviz import Digraph
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

    # # 遍历所有的节点
    # for ast_type in program_ast.types:
    #     # 节点名
    #     node_name = type(ast_type).__name__
    #     # 方法应该存在于类定义中，因此需要找到类定义的标签
    #     if node_name == "ClassDeclaration":
    #         # 遍历类节点的子节点，获取到方法节点
    #         for body in ast_type.body:
    #             # 获取当前节点的节点名
    #             body_type = type(body).__name__
    #             # 如果当前节点是方法节点
    #             if body_type == "MethodDeclaration":
    #                 # 添加结果集
    #                 method_asts.append(body)

    return program_ast


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
                asts = get_method_asts(code)
                method_ast_list.extend(asts)

                programfile.close()
    return method_ast_list


def visit(node, nodes, pindex, g):
    name = str(type(node).__name__)

    index = len(nodes)
    nodes.append(index)
    g.node(str(index), name)
    if index != pindex:
        g.edge(str(index), str(pindex))
    for n in ast.iter_child_nodes(node):
        visit(n, nodes, index, g)


def visit_java_ast(node, nodes, pindex, g):
    index = len(nodes)
    nodes.append(index)
    print(node.__class__)
    name = str(type(node).__name__)
    g.node(str(index), name)
    if index != pindex:
        g.edge(str(index), str(pindex))


    if hasattr(node,"types"):
        for n in node.types:
            visit_java_ast(n, nodes, index, g)
    if hasattr(node,"body"):
        for n in node.body:
            visit_java_ast(n, nodes, index, g)
    if hasattr(node,"statements"):
        for n in node.statements:
            visit_java_ast(n, nodes, index, g)


if __name__ == '__main__':
    graph = Digraph(format="png")
    # tree = ast.parse(code)
    code = """
        import java.util.Arrays;
        import java.util.LinkedList;
        import java.util.List;

        public class test {
            public void testMultiParams(String... params) {
                for (String string : params) {
                    System.out.println(string);
                }
            }

            public void testGener(List<?> list) {
            }

            public void testGener1() {
                List<String> list = new LinkedList<>();

            }
        }
    """
    program_ast = javalang.parse.parse(code)
    visit_java_ast(program_ast, [], 0, graph)
    graph.render("test")