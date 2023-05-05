from typing import List,Dict
import os
import joblib

def get_tokens(dataset_path:str)->Dict:
    """
    根据输入的数据集的位置获取数据集的所有token
    @param dataset_path: 数据的路径
    @return: token-id 对应的列表
    """
    all_tokens = set([])
    for rt, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".data"):
                data = joblib.load(os.path.join(rt, file))
                # 解析出两个ast和标签,从这儿取出来的ast
                ast1,ast2,label = data 
                

                










if __name__ == "__main__":
    pass
