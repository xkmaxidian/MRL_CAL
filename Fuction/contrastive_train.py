import torch
import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder
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

def contrastive_train(epoch, args, data_loader, view, device, model, optimizer, criterion):
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
            for w in range(v+1, view):
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
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))
