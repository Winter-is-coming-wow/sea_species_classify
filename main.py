from data_balance import balance
from data_aug import aug_gen
from data_loader import load_data
from model import efficientnet_model
from train import finetune
from test import predict
import os
import time
from gl import *

if __name__ == '__main__':
    start = time.perf_counter()
    balance()
    if not os.path.exists('/dataset/data_auged/'):
        aug_gen()
    train_dataset, test_dataset = load_data()
    model = efficientnet_model(num_classes)
    finetune(train_dataset, model_type, model)
    predict(test_dataset, model)
    end = time.perf_counter()
    d = end - start
    print("运行时间: %f s" % d)
