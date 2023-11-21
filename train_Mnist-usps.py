import os
from Network import Network
import numpy as np
import random
from loss import Loss
from dataloader import load_data
import os
from Fuction import pretrain,contrastive_train
from metric import valid

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import argparse
import torch
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder
import scipy.io as scio
import warnings
warnings.filterwarnings("ignore")

def get_mask(view_num, data_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data.
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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def datasave():
    np.random.seed(10)
    data_0 = scio.loadmat('..//data//MNIST_USPS.mat')
    V1 = data_0['X1'].astype(np.float32)
    V2 = data_0['X2'].astype(np.float32)
    x11 = V1.reshape(5000, 784)
    x22 = V2.reshape(5000, 784)
    x1 = x11.astype(np.float32)
    x2 = x22.astype(np.float32)
    mask = get_mask(2, x1.shape[0], 0.5)  # missing_rate = 0.5
    data1 = x1 * mask[:, 0][:, np.newaxis]
    data2 = x2 * mask[:, 1][:, np.newaxis]
    data1 = data1.astype(np.float32)
    data2 = data2.astype(np.float32)
    Y = scio.loadmat('..//data//MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000, )
    xx1 = data1.reshape(5000, 28, 28)
    xx2 = data2.reshape(5000, 28, 28)
    incomplete = '..//in_mnist_0.5s.mat'
    scio.savemat(incomplete, {'xi1': xx1, 'xi2': xx2, 'x1': x1, 'x2': x2, 'y': Y, 'mask': mask})


if __name__ == '__main__':
    Dataname = 'MNIST-USPS'
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', default=Dataname)
    parser.add_argument('--batch_size', default=1000, type=int)  # 256--32--1000
    parser.add_argument("--temperature_f", default=0.5)
    parser.add_argument("--temperature_l", default=1.0)
    parser.add_argument("--learning_rate", default=0.0003)
    parser.add_argument("--weight_decay", default=0.)
    parser.add_argument("--workers", default=8)
    parser.add_argument("--mse_epochs", default=200)  # 200
    parser.add_argument("--con_epochs", default=100)  # 50
    parser.add_argument("--tune_epochs", default=100)  # 50
    parser.add_argument("--feature_dim", default=512)
    parser.add_argument("--high_feature_dim", default=256)
    parser.add_argument("--missing_rate", default=0.3)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset == "MNIST-USPS":
        args.con_epochs = 50
        seed = 10
    setup_seed(seed)
    dataset, dims, view, data_size, class_num = load_data(args.dataset)
    print(args.dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
    )
    # datasave()
    accs = []
    nmis = []
    aris = []
    if not os.path.exists('./model'):
        os.makedirs('./model')
    times = 1
    for i in range(times):
        print("ROUND:{}".format(i + 1))
        model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
        print(model)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)
        epoch = 1
        while epoch <= args.mse_epochs:  # args.mse_epochs = 200
            tot_loss = 0.
            mes = torch.nn.MSELoss()
            for batch_idx, (xs, _, _) in enumerate(data_loader):
                if args.dataset == "MNIST-USPS":
                    np.random.seed(10)
                    mask = get_mask(2, xs[0].shape[0], args.missing_rate)
                    xs[0] = torch.from_numpy((xs[0] * mask[:, 0][:, np.newaxis]).numpy().astype('float32'))
                    xs[1] = torch.from_numpy((xs[1] * mask[:, 1][:, np.newaxis]).numpy().astype('float32'))
                    for v in range(view):
                        xs[v] = xs[v].to(device)
                else:
                    for v in range(view):
                        # print(xs[v].shape)
                        xs[v] = xs[v].to(device)
                optimizer.zero_grad()
                _, _, xrs, _, xfakes, d, d_ = model(xs)
                loss_list = []
                for v in range(view):
                    loss_list.append(mes(xs[v], xrs[v]))
                generator_loss = 0
                discriminator_loss = 0
                for i in range(view):
                    generator_loss -= torch.mean(d_[i])
                    discriminator_loss = discriminator_loss - torch.mean(d[i]) + torch.mean(d_[i])
                Gan_loss = discriminator_loss + generator_loss
                loss = sum(loss_list) + Gan_loss
                loss.backward()
                optimizer.step()
                tot_loss += loss.item()
            print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
            epoch += 1
        while epoch <= args.mse_epochs + args.con_epochs:  # args.con_epochs = 50; 250
            tot_loss = 0.
            mes = torch.nn.MSELoss()
            for batch_idx, (xs, _, _) in enumerate(data_loader):
                if args.dataset == "MNIST-USPS":

                    np.random.seed(10)
                    mask = get_mask(2, xs[0].shape[0], args.missing_rate)
                    xs[0] = torch.from_numpy((xs[0] * mask[:, 0][:, np.newaxis]).numpy().astype('float32'))
                    xs[1] = torch.from_numpy((xs[1] * mask[:, 1][:, np.newaxis]).numpy().astype('float32'))
                    for v in range(view):
                        xs[v] = xs[v].to(device)
                else:
                    for v in range(view):
                        xs[v] = xs[v].to(device)
                optimizer.zero_grad()
                hs, qs, xrs, zs, xfakes, d, d_ = model(xs)
                loss_list = []
                for v in range(view):
                    for w in range(v + 1, view):
                        loss_list.append(criterion.forward_feature(hs[v], hs[w]))
                        loss_list.append(criterion.forward_label(qs[v], qs[w]))
                    loss_list.append(mes(xs[v], xrs[v]))
                generator_loss = 0
                discriminator_loss = 0
                for i in range(view):
                    generator_loss -= torch.mean(d_[i])
                    discriminator_loss = discriminator_loss - torch.mean(d[i]) + torch.mean(d_[i])
                Gan_loss = discriminator_loss + generator_loss
                loss = sum(loss_list) + Gan_loss
                loss.backward()
                optimizer.step()
                tot_loss += loss.item()
            print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
            if epoch == args.mse_epochs + args.con_epochs:  # 250
                acc, nmi, ari, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False)
                print('Saving..')
                accs.append(acc)
                nmis.append(nmi)
                aris.append(pur)
            epoch += 1


