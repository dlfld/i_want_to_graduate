import javalang
import javalang.tree
import javalang.ast
import javalang.util
from javalang.ast import Node
def get_children(root):

    if isinstance(root, Node):
        children = root.children
        print(type(root))
        print(len(children))
        print(children)
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


if __name__ == "__main__":
    code = """
class Solution {
    public int[] twoSum(int[] nums, int target) {
        int n = nums.length;
        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (nums[i] == target - nums[j]) {
                    return new int[]{i, j};
                }
            }
        }
        return new int[0];
    }
}
    """
    # programtokens = javalang.tokenizer.tokenize(code)
    # # 这个token是每一个符号
    # # print(list(programtokens))
    # parser = javalang.parse.Parser(programtokens)
    # programast = parser.parse_member_declaration()
    # # print(programast)
    # children = get_children(programast)
    # print(len(children))
    class A:
        b = 1
        c = 2
        d = 3
    a = A()
    print(a.children)