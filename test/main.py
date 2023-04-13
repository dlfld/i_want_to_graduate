# 通过这个代码可以获取到Java的语法分析树
import antlr4
from JavaLexer import JavaLexer
from JavaParser import JavaParser

# 定义Java代码
java_code = "public class Main { public static void main(String[] args) { System.out.println(\"Hello World!\"); } }"

# 创建ANTLR输入流
input_stream = antlr4.InputStream(java_code)

# 创建JavaLexer
lexer = JavaLexer(input_stream)

# 创建ANTLR词法分析器
token_stream = antlr4.CommonTokenStream(lexer)

# 创建JavaParser
parser = JavaParser(token_stream)

# 解析Java代码
tree = parser.compilationUnit()

# 输出语法分析树
print(tree.toStringTree(recog=parser))
