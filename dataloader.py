from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import scipy.io as scio
import argparse
import torch
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
import scipy.io as sio
parser = argparse.ArgumentParser()
parser.add_argument("--missing_rate", default=0.5)
# parser.add_argument("--data_seed ", default= 10)
args = parser.parse_args()
def normalize(x):
    """Normalize""" #可以替换为Z-score
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x

def get_mask(view_num, data_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data.
        随机生成不完整数据信息，用完整视图数据模拟局部视图数据

        Args:
          view_num: view number
          data_len: number of samples
          missing_rate: Defined in section 4.1 of the paper
        Returns:
          mask

    """
    missing_rate = missing_rate / view_num
    one_rate = 1.0 - missing_rate
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(data_len, 1))).toarray()
        return view_preserve
    error = 1
    if one_rate == 1:
        matrix = randint(1, 2, size=(data_len, view_num))
        return matrix
    while error >= 0.005:
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(data_len, 1))).toarray()
        one_num = view_num * data_len * one_rate - data_len
        ratio = one_num / (view_num * data_len)
        matrix_iter = (randint(0, 100, size=(data_len, view_num)) < int(ratio * 100)).astype(np.int)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * data_len)
        matrix_iter = (randint(0, 100, size=(data_len, view_num)) < int(ratio * 100)).astype(np.int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int)
        ratio = np.sum(matrix) / (view_num * data_len)
        error = abs(one_rate - ratio)

    return matrix

class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)

        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)
        # print("Noisy shaped", self.V1[0].shape)


    def __len__(self):
        return 5000

    # np.random.seed(10)
    # mask = get_mask(2, xs[0].shape[0], args.missing_rate)
    # xs[0] = torch.from_numpy((xs[0] * mask[:, 0][:, np.newaxis]).numpy().astype('float32'))
    # xs[1] = torch.from_numpy((xs[1] * mask[:, 1][:, np.newaxis]).numpy().astype('float32'))


    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        # np.random.seed(10)
        # mask = get_mask(2, 5000, args.missing_rate)
        # x1 = (x1 * mask[:, 0][:, np.newaxis]).astype(np.float32)
        # x2 = (x2 * mask[:, 0][:, np.newaxis]).astype(np.float32)
        # print("shape", torch.from_numpy(x1).shape)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

def load_data(dataset):
    if dataset == "MNIST-USPS":
        dataset = MNIST_USPS('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
        data_seed = 10
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
