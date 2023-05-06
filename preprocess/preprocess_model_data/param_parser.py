import pycparser
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--dataset", default='gcj')
    parser.add_argument("--graphmode", default='astandnext')
    # parser.add_argument("--graphmode", default='astonly')
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
    parser.add_argument("--num_epochs", default=200)
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--threshold", default=0.5)
    parser.add_argument("--loss_name", default="loss_data.data")
    parser.add_argument("--filters-3", default=32,
                        help="Filters (neurons) in 3rd convolution. Default is 32.")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout probability. Default is 0.5.")
    args = parser.parse_args()
    return args
