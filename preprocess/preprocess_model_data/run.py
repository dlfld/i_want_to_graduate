from sklearn.model_selection import train_test_split
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import time
import sys
import argparse
from tqdm import tqdm, trange
import pycparser
import models
# from torch_geometric.data import Data, DataLoader
from torch.utils.data import DataLoader
import os
import joblib
from torch.utils.tensorboard import SummaryWriter
from data_embedding import get_all_mes_datas
import logddd
from early_stopping import EarlyStopping
writer = SummaryWriter('log/')
save_path = "./models/"  # 当前目录下
# early_stopping = EarlyStopping()
early_stopping = EarlyStopping(
    patience=10, verbose=True, save_path=save_path)

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=True)
parser.add_argument("--dataset", default='gcj')
parser.add_argument("--graphmode", default='astandnext')
parser.add_argument("--nextsib", default=True)
parser.add_argument("--ifedge", default=True)
parser.add_argument("--whileedge", default=True)
parser.add_argument("--foredge", default=True)
parser.add_argument("--blockedge", default=True)
parser.add_argument("--nexttoken", default=True)
parser.add_argument("--nextuse", default=True)
parser.add_argument("--data_setting", default='11')
parser.add_argument("--batch_size", default=32)
parser.add_argument("--num_layers", default=4)
parser.add_argument("--num_epochs", default=100)
parser.add_argument("--lr", default=0.001)
parser.add_argument("--threshold", default=0)
args = parser.parse_args()


def create_batches(data):
    batches = [data[graph:graph+args.batch_size]
               for graph in range(0, len(data), args.batch_size)]
    return batches


def split_data(data):
    """
        划分数据，换分训练集测试集和验证集
        @return 训练集、验证集、测试集
    """
    X_train, X_validate_test, _, _ = train_test_split(
        data, data, test_size=0.2, random_state=42)
    X_validate, X_test, _, _ = train_test_split(
        X_validate_test, X_validate_test, test_size=0.5, random_state=42)
    return X_train, X_validate, X_test


if __name__ == '__main__':
    device = torch.device('cuda:0')
    data_file_name = "mom_data.data"
    data_file_path = "../generate_dataset/dataset/"
    # 加载数据
    if not os.path.exists(data_file_name):
        all_data_list, vocab_size = get_all_mes_datas(data_file_path)
        temp_data = {
            "all_data_list": all_data_list,
            "vocab_size": vocab_size
        }
        joblib.dump(temp_data, data_file_name)
    else:
        temp_data = joblib.load(data_file_name)
        all_data_list = temp_data["all_data_list"]
        vocab_size = temp_data["vocab_size"]

    # 划分数据
    train, valid, test = split_data(all_data_list)

    train_data_loader = create_batches(train)

    num_layers = int(args.num_layers)
    model = models.GMNnet(vocab_size, embedding_dim=100,
                          num_layers=num_layers, device=device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CosineEmbeddingLoss()
    criterion2 = nn.MSELoss()
    criterion3 = torch.nn.BCEWithLogitsLoss()
    criterion4 = nn.BCELoss()

    epochs = trange(args.num_epochs, leave=True, desc="Epoch")

    eval_losses = []

    for epoch in epochs:
        epoch_loss = 0
        model.train()
        for index, batch in tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc="Batches"):
            optimizer.zero_grad()
            batch_losses = []
            for data in batch:
                x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2, label = data
                label = [[1, 0]] if label == -1 else [[0, 1]]

                label = torch.tensor(label, dtype=torch.float, device=device)
                x1 = torch.tensor(x1, dtype=torch.long, device=device)
                x2 = torch.tensor(x2, dtype=torch.long, device=device)
                # logddd.log(x1.shape)
                # logddd.log(x2.shape)
                edge_index1 = torch.tensor(
                    edge_index1, dtype=torch.long, device=device)
                edge_index2 = torch.tensor(
                    edge_index2, dtype=torch.long, device=device)
                # logddd.log(edge_index1.shape)
                # logddd.log(edge_index2.shape)
                if edge_attr1 != None:
                    edge_attr1 = torch.tensor(
                        edge_attr1, dtype=torch.long, device=device)
                    edge_attr2 = torch.tensor(
                        edge_attr2, dtype=torch.long, device=device)
                # logddd.log(edge_attr1.shape)
                # logddd.log(edge_attr2.shape)
                data = [x1, x2, edge_index1,
                        edge_index2, edge_attr1, edge_attr2]

                logits = model(data)
                loss = criterion3(logits, label)
                batch_losses.append(loss.item())
                loss.backward(retain_graph=True)

            # 记录过程中每一个batch的loss
            batch_loss = np.average(batch_losses)
            writer.add_scalar('loss', batch_loss, epoch*len(batch)+index)
            epochs.set_description("Epoch (Loss=%g)" % round(batch_loss, 5))

            optimizer.step()

        model.eval()
        with torch.no_grad():
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            p = 0.0
            r = 0.0
            f1 = 0.0
            for data in valid:
                x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2, label = data
                label = [[1, 0]] if label == -1 else [[0, 1]]

                label = torch.tensor(label, dtype=torch.float, device=device)
                x1 = torch.tensor(x1, dtype=torch.long, device=device)
                x2 = torch.tensor(x2, dtype=torch.long, device=device)
                edge_index1 = torch.tensor(
                    edge_index1, dtype=torch.long, device=device)
                edge_index2 = torch.tensor(
                    edge_index2, dtype=torch.long, device=device)
                if edge_attr1 != None:
                    edge_attr1 = torch.tensor(
                        edge_attr1, dtype=torch.long, device=device)
                    edge_attr2 = torch.tensor(
                        edge_attr2, dtype=torch.long, device=device)

                train_data = [x1, x2, edge_index1,
                              edge_index2, edge_attr1, edge_attr2]
                logits = model(train_data)
                output = torch.sigmoid(logits)
                pred = output.squeeze(0)

                loss = criterion4(output, label)
                eval_losses.append(loss.item())

                if pred[0] > pred[1]:
                    prediction = 0.4
                else:
                    prediction = 0.6

                y = label.tolist()
                # print(label[0]==[0,1])
                if prediction > args.threshold and y[0] == [0, 1]:
                    tp += 1
                    # print('tp')
                if prediction <= args.threshold and y[0] == [1, 0]:
                    tn += 1
                    # print('tn')
                if prediction > args.threshold and y[0] == [1, 0]:
                    fp += 1
                    # print('fp')
                if prediction <= args.threshold and y[0] == [0, 1]:
                    fn += 1

                p = tp/(tp+fp)
                if tp+fn == 0:
                    print('recall is none')
                    exit(1)

                r = tp/(tp+fn)
                f1 = 2*p*r/(p+r)
                acc = (tp + tn) / len(valid)
                print(f'\nprecision = {p}')
                print(f'recall = {r}')
                print(f'F1={f1}')
                print(f"acc = {acc}")

                avg_valid_loss = np.average(eval_losses)
                print("验证集loss:{}".format(avg_valid_loss))
                early_stopping(avg_valid_loss, model)
                if early_stopping.early_stop:
                    print("此时早停！")
                    break
