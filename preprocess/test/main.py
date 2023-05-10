import javalang
if __name__ == '__main__':
    code = """
    class Test{

        /**
        * 提供精确的加法运算。
        *  v1 被加数
        *  v2 加数
        *  两个参数的和
        */
        public static double add(double v1, double v2)
        {
            BigDecimal b1 = new BigDecimal(Double.toString(v1));
            BigDecimal b2 = new BigDecimal(Double.toString(v2));
            return b1.add(b2).doubleValue();
        }
        public AjaxResult getInfo() throws Exception
        {
            new Server().copyTo();
            Server server = new Server();
            server.copyTo();

            return AjaxResult.success(server);
        }
    }

    """
    import javalang
    programtokens = javalang.tokenizer.tokenize(code)
    programast = javalang.parser.parse(programtokens)
    print(javalang.parse.parse(code))


    # print()
    # from anytree import AnyNode
    #
    # new_tree = AnyNode(id=1,token="token",data="aa",parent=None)
    # sub = AnyNode(id=2,token="2",data = "bb",parent=new_tree)
    # print(new_tree.children)
    # print(sub)