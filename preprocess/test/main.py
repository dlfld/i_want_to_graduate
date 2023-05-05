if __name__ == '__main__':
    code = """
    class Test{
        public AjaxResult getInfo() throws Exception
        {
            Server server = new Server();
            server.copyTo();
            return AjaxResult.success(server);
        }
    }

    """
    import javalang

    # print(javalang.parse.parse(code))
    from anytree import AnyNode

    new_tree = AnyNode(id=1,token="token",data="aa",parent=None)
    sub = AnyNode(id=2,token="2",data = "bb",parent=new_tree)
    print(new_tree.children)
    # print(sub)