from createclone_bcb import createast, creategmndata, createseparategraph
import joblib


def load_bcb_data(args):
    """
        获取bcb的数据
        @return: bcb数据和词表
    """
    data_file_name = "bcb_data.data"
    data = []
    if not os.path.exists(data_file_name):
        # 读取数据集获取数据信息
        astdict, vocablen, vocabdict = createast()
        # 数据预处理
        treedict = createseparategraph(astdict, vocablen, vocabdict, mode=args.graphmode, nextsib=args.nextsib, ifedge=args.ifedge,
                                       whileedge=args.whileedge, foredge=args.foredge, blockedge=args.blockedge, nexttoken=args.nexttoken, nextuse=args.nextuse)
        # 获取格式化数据
        traindata, validdata, testdata = creategmndata(
            args.data_setting, treedict, vocablen, vocabdict)
        data.extend(traindata)
        data.extend(validdata)
        data.extend(testdata)

        joblib.dump(data, data_file_name)
    else:
        data = joblib.load(data_file_name)

    return data, vocabdict


if __name__ == '__main__':
    data, vocabdict = load_bcb_data()
    print(data)
