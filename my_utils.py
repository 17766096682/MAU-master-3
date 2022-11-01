import codecs
import csv
import glob
import math
import os

import numpy as np
import torch
from imageio import imread
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm
import torch.nn as nn


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0


up_path = {
    70: "Radar",
    35: "Wind",
    10: "Precip"
}
low_path = {
    70: "radar_",
    35: "wind_",
    10: "precip_"
}


def prep_clf(obs, pre, low=0, high=70):
    '''
    func: 计算二分类结果-混淆矩阵的四个元素
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。

    returns:
        hits, misses, falsealarms, correctnegatives
        #aliases: TP, FN, FP, TN
    '''
    # 根据阈值分类为 0, 1
    obs = (np.where(obs >= low, 1, 0)) & (np.where(obs < high, 1, 0))
    pre = (np.where(pre >= low, 1, 0)) & (np.where(pre < high, 1, 0))

    # True positive (TP)
    hits = np.sum((obs == 1) & (pre == 1)).astype(np.float64)

    # False negative (FN)
    misses = np.sum((obs == 1) & (pre == 0)).astype(np.float64)

    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (pre == 1)).astype(np.float64)

    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (pre == 0)).astype(np.float64)

    return hits, misses, falsealarms, correctnegatives


def HSS(obs, pre, low, high):
    '''
    HSS - Heidke skill score
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): pre
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: HSS value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre,
                                                           low=low, high=high)

    HSS_num = 2 * (hits * correctnegatives - misses * falsealarms)
    HSS_den = (misses ** 2 + falsealarms ** 2 + 2 * hits * correctnegatives +
               (misses + falsealarms) * (hits + correctnegatives))
    if HSS_den == 0:
        return 0.0
    return HSS_num / HSS_den


def BIAS(obs, pre, low, high):
    '''
    func: 计算Bias评分: Bias =  (hits + falsealarms)/(hits + misses)
    	  alias: (TP + FP)/(TP + FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    returns:
        dtype: float
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre,
                                                           low=low, high=high)
    if hits + misses == 0:
        return 0.0
    return (hits + falsealarms) / (hits + misses)


def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) * 2.0)

    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    fid = 0
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
        # calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def escore(image1, image2, low, high):
    hss = HSS(image1, image2, low, high)
    bias = BIAS(image1, image2, low, high)
    fid = calculate_fid(image1, image2)
    score = hss * (math.exp(-abs(1 - bias)) ** 0.2) * (math.exp(-(fid / 100)) ** 0.2)
    return score


def radar_sc(image1, image2):
    return escore(image1, image2, 30, 40) * 0.2 + escore(image1, image2, 40, 50) * 0.3 + escore(image1, image2, 50,
                                                                                                70) * 0.5


def wind_sc(image1, image2):
    return escore(image1, image2, 5, 10.8) * 0.2 + escore(image1, image2, 10.8, 17.2) * 0.3 + escore(image1, image2,
                                                                                                     17.2,
                                                                                                     35) * 0.5


def precip_sc(image1, image2):
    return escore(image1, image2, 5, 10) * 0.2 + escore(image1, image2, 10, 20) * 0.3 + escore(image1, image2,
                                                                                               20,
                                                                                               100) * 0.5


def radar(gtimage, genimage):
    return radar_sc(gtimage[-15, ...], genimage[-15, ...]) * 0.3 + radar_sc(gtimage[-10, ...],
                                                                            genimage[-10, ...]) * 0.3 + radar_sc(
        gtimage[-1], genimage[-1]) * 0.4


def wind(gtimage, genimage):
    return wind_sc(gtimage[-15, ...], genimage[-15, ...]) * 0.3 + wind_sc(gtimage[-10, ...],
                                                                          genimage[-10, ...]) * 0.3 + wind_sc(
        gtimage[-1], genimage[-1]) * 0.4


def precip(gtimage, genimage):
    image1 = np.sum(gtimage[-20:-10, ...], axis=(0))
    image2 = np.sum(genimage[-20:-10, ...], axis=(0))
    score1 = precip_sc(image1, image2)

    image1 = np.sum(gtimage[-15:-5, ...], axis=(0))
    image2 = np.sum(genimage[-15:-5, ...], axis=(0))
    score2 = precip_sc(image1, image2)

    image1 = np.sum(gtimage[-10:-1, ...], axis=(0))
    image2 = np.sum(genimage[-10:-1, ...], axis=(0))
    score3 = precip_sc(image1, image2)

    return 0.3 * score1 + 0.3 * score2 + 0.4 * score3


def batch_radar(gtimage, genimage):
    assert genimage.shape == genimage.shape, "geimage,genimage shaoe is not same"
    batch = gtimage.shape[0]
    sc_list = []
    gtimage = np.clip(gtimage, 0, 1) * 70.0
    genimage = np.clip(genimage, 0, 1) * 70.0
    for i in range(batch):
        sc_list.append(radar(gtimage=gtimage[i, ..., 0], genimage=genimage[i, ..., 0]))
    return np.mean(sc_list)


def batch_wind(gtimage, genimage):
    assert genimage.shape == genimage.shape, "geimage,genimage shaoe is not same"
    batch = gtimage.shape[0]
    sc_list = []
    gtimage = np.clip(gtimage, 0, 1) * 35.0
    genimage = np.clip(genimage, 0, 1) * 35.0
    for i in range(batch):
        sc_list.append(wind(gtimage=gtimage[i, ..., 0], genimage=genimage[i, ..., 0]))
    return np.mean(sc_list)


def batch_precip(gtimage, genimage):
    assert genimage.shape == genimage.shape, "geimage,genimage shaoe is not same"
    batch = gtimage.shape[0]
    sc_list = []
    gtimage = np.clip(gtimage, 0, 1) * 10.0
    genimage = np.clip(genimage, 0, 1) * 10.0
    for i in range(batch):
        sc_list.append(precip(gtimage=gtimage[i, ..., 0], genimage=genimage[i, ..., 0]))
    return np.mean(sc_list)


def load_csv(root, factor, directory="TestB1/", filename='TestB1.csv'):
    filename = up_path[factor] + filename
    if not os.path.exists(os.path.join(root, filename)):
        category = root + directory + up_path[factor]
        dirs = os.listdir(category)
        with open(os.path.join(root, filename), mode='w', newline='') as f:  #
            writer = csv.writer(f)
            for dir in dirs:
                images = []
                images += glob.glob(os.path.join(category, dir, '*.png'))
                writer.writerow([dir] + images)
    images = []
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:
            images.append(row)
    return images


def image_read(sinle_image):
    image = np.array(imread(sinle_image) / 255.0, np.float64)
    return image


def image_num_TestB(root, factor, directory, filename):
    pathlist = load_csv(root=root, factor=factor, directory=directory, filename=filename)
    total_avg = []
    with codecs.open('precip_sum_TestB.txt', 'w+') as data_write:
        for line in pathlist:
            images = []
            dirname = line[0]
            print(dirname)
            imagepath = line[-20:]
            for path in imagepath:
                images.append(np.sum(image_read(path)))
            avg_num = np.sum(images) / len(images)
            total_avg.append(avg_num)
            data_write.writelines(str([dirname, avg_num] + images) + '\n')
        data_write.writelines(str(np.mean(total_avg)) + '\n')


# def loadcsv(root,csvname):
#     image_path = []
#     with open(os.path.join(root,csvname)) as f:
#         reader = csv.reader(f)
#         for line in reader:
#             image_path.append(line)
#     return image_path

def loadcsv(root, factor):
    image_path = glob.glob(os.path.join(root, "TestA", up_path[factor], '*.png'))
    return image_path


def image_read_TestA(sinle_image):
    # image_name = low_path[factor] + sinle_image
    # image_path = os.path.join(root, "TestA", up_path[factor], image_name)
    image = np.array(imread(sinle_image) / 255.0, np.float64)
    return np.sum(image)


def image_num_Testa(root, csvname, factor):
    imagelist = loadcsv(root, factor)
    img_sum = []
    total_avg = []
    with codecs.open('precip_sum_TestA.txt', 'w+') as data_write:
        for i in range(len(imagelist)):
            img = image_read_TestA(imagelist[i])
            img_sum.append(img)
            if (i + 1) % 20 == 0:
                print(i + 1)
                avg = np.mean(img_sum)
                total_avg.append(avg)
                data_write.writelines(str([i + 1] + [avg]) + '\n')
                img_sum = []
        data_write.writelines(str(np.mean(total_avg)) + '\n')


# 计算results里面的最优得分
def best_score(count):
    scorelist = [0]
    dirlist = [0]
    dir_list = os.listdir("results/radar_mau/")
    for dir in dir_list:
        path = os.path.join("results/radar_mau/", dir, "data.txt")
        with open(path, mode="r") as f:
            line = f.readline()
            score = float(line.split(":")[-1])
            minscore = min(scorelist)
            if score > minscore:
                if len(scorelist) < count:
                    scorelist.append(score)
                    dirlist.append(dir)
                else:
                    index = scorelist.index(minscore)
                    scorelist[index] = score
                    dirlist[index] = dir
    with open("results/best_score.txt", mode="w") as f:
        f.write(str(scorelist) + "\n")
        f.write(str(dirlist))


# THRESHOLDS = np.array([0.5, 2, 5, 10, 30])
# BALANCING_WEIGHTS = (1, 1, 2, 5, 10, 30)

THRESHOLDS = {10: np.array([0.05, 0.1, 0.2]),
              35: np.array([0.14285714, 0.308571428, 0.5]),
              70: np.array([0.4285714, 0.57142858, 0.71428571])}

WEIGHTS = {10: (1, 2, 3, 30),
           35: (1, 2, 3, 30),
           70: (1, 2, 3, 30), }


class Weighted_mse_mae(nn.Module):
    def __init__(self, factor, mse_weight=1.0, mae_weight=1.0, NORMAL_LOSS_GLOBAL_SCALE=0.00005, LAMBDA=0.1):
        super().__init__()
        self.NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.factor = factor
        self._lambda = LAMBDA

    def forward(self, input, target):
        target =torch.clip(target,0,1)
        balancing_weights = WEIGHTS[self.factor]
        weights = torch.ones_like(input) * balancing_weights[0]
        # thresholds = [rainfall_to_pixel(ele) for ele in THRESHOLDS]
        thresholds = THRESHOLDS[self.factor]
        for i, threshold in enumerate(thresholds):
            weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * (target >= threshold).float()
        # weights = weights * mask.float()result = {Tensor} tensor([[[-0.8939, -1.0224, -1.1566,  ...,  1.3844,  1.0185, -0.9009],\n         [-0.4063, -0.3823, -0.3098,  ..., -0.5043,  0.0477, -0.8242],\n         [ 0.6024, -0.3302,  1.9940,  ..., -0.0709, -0.5939, -0.4209],\n         ...,\n         [-0.5955,  0.9327,  … View
        weights = weights.float()
        # input: S*B*1*H*W
        # error: S*B
        mse = torch.sum(weights * ((input - target) ** 2), (2, 3, 4))
        mae = torch.sum(weights * (torch.abs((input - target))), (2, 3, 4))
        if self._lambda is not None:
            B, S = mse.size()
            w = torch.arange(1.0, (1.0 + S * self._lambda), self._lambda)
            w[:-10] = 1.0
            w = w[:14]
            if torch.cuda.is_available():
                w = w.to(mse.get_device())
            mse = w * mse
            mae = w * mae
            # mae = (w * mae.permute(1, 0)).permute(1, 0)
        return self.NORMAL_LOSS_GLOBAL_SCALE * (
                    self.mse_weight * torch.mean(mse) + self.mae_weight * torch.mean(mae))  # 将整个bath的数据损失平均了


if __name__ == '__main__':
    # image_num_TestB(root="../MAU-master/data/", factor=10, directory="TestB1/", filename="TestB1.csv")
    # image_num_Testa(root="../MAU-master/data/", csvname="TestA.csv", factor=10)
    # best_score(10)

    loss = Weighted_mse_mae(factor=10, LAMBDA=0.1)
    a = torch.randn((2, 10, 1, 480, 560)).cuda()
    b = torch.randn((2, 10, 1, 480, 560)).cuda()
    print(loss(a , b ))
