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

    print(javalang.parse.parse(code))
