import numpy as np
import torch


def crLoss(data, label, class_num):
    data = data.cuda()
    label = label.cuda()
    groups = []
    group_center = []
    for i in range(class_num):
        temp1 = data[[label == i]]
        length = len(temp1)
        if length > 0:
            temp2 = sum(temp1)
            group_center.append(temp2 / length)
        groups.append(temp1)
    d1 = -mutiVertDistance(group_center) 
    d2 = 0.0
    for group in groups:
        if len(group) > 1:
            d2 += mutiVertDistance(group)
    d2 = d2 / len(groups)
    return d1, d2


def mutiVertDistance(arr):
    n = len(arr)
    temp = torch.zeros(n, arr[0].shape[0]).cuda()
    for i in range(n):
        temp[i] = arr[i]
    temp = temp / (temp ** 2).sum(axis=1, keepdims=True) ** 0.5
    distance = (torch.sum(1.0 - torch.mm(temp, temp.T)))
    return distance / n / (n - 1)

