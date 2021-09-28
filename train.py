from __future__ import print_function
import argparse
import os
import random
import time

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from sklearn.cluster import KMeans
from torch.autograd import Variable
import math

import classifier
import util
import sys
import model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import torch.nn.functional as F
from crUtil import crLoss

# from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CUB', help='CUB')
parser.add_argument('--k', type=int, default=1, help='k for knn')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--syn_num', type=int, default=400, help='number features to generate per class')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--ccf', type=float, default=1.0, help='the weight for the ccf loss')
parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
parser.add_argument('--center_weight', type=float, default=10, help='the weight for the center loss')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--nepoch', type=int, default=501, help='number of epochs to train for')
parser.add_argument('--dataroot', default='data', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--gzsl', action='store_true', default=True, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=True,
                    help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=312, help='size of semantic features')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=0.2, help='weight of the classification loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
parser.add_argument('--manualSeed', type=int, default=3483, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--nclass_seen', type=int, default=150, help='number of seen classes')
parser.add_argument('--lr_dec_ep', type=int, default=12, help='lr decay for every n epoch')
parser.add_argument('--lr_dec_rate', type=float, default=0.95, help='lr decay rate')
parser.add_argument('--mapSize', type=int, default=2048, help='the size of mapping output')
parser.add_argument('--test_epoch', type=int, default=25, help='view result per test_epoch')
parser.add_argument('--use_classify', type=bool, default=False, help='use classify or not')

opt = parser.parse_args()
print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

# cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

# initialize generator and discriminator
netG = model.Generator(opt)
discriminator = model.D2(opt)
print(netG)
print(discriminator)

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
input_label = torch.LongTensor(opt.batch_size)

if opt.cuda:
    discriminator.cuda()
    netG.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    input_label = input_label.cuda()


def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))


def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    with torch.no_grad():
        for i in range(nclass):
            iclass = classes[i]
            iclass_att = attribute[iclass]
            syn_att.copy_(iclass_att.repeat(num, 1))
            syn_noise = Variable(torch.randn(num, opt.nz)).cuda()
            output = netG(syn_noise, syn_att)
            syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
            syn_label.narrow(0, i * num, num).fill_(iclass)
    return syn_feature, syn_label


# setup optimizer
optimizerD = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes):
    acc_per_class = 0
    for i in target_classes:
        idx = (test_label == i)
        if torch.sum(idx) == 0:
            acc_per_class += 0
        else:
            acc_per_class += float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
    acc_per_class /= float(target_classes.size(0))
    return acc_per_class


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates, _, _ = netD(interpolates)
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty


def KNNPredict(X_train, y_train, X_test, k=5):
    sim = -1 * euclidean_distances(X_test.cpu().data.numpy(), X_train.cpu().data.numpy())
    idx_mat = np.argsort(-1 * sim, axis=1)[:, 0: k]
    preds = np.array([np.argmax(np.bincount(item)) for item in y_train[idx_mat]])
    return preds


final_result = {
    "acc_unseen": 0,
    "acc_seen": 0,
    "H": 0
}


start = time.time()
for start_step in range(0, opt.nepoch):

    for p in discriminator.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update

    # train D
    for iter_d in range(5):
        sample()
        discriminator.zero_grad()

        input_resv = Variable(input_res)
        input_attv = Variable(input_att)

        D_real, C_real, mR = discriminator(input_resv)  
        C_loss_real = F.cross_entropy(C_real, input_label)
        D_loss_real = D_real.mean()
        crR1, crR2 = crLoss(mR, input_label, opt.nclass_all)

        # fake
        noise.normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev, input_attv)
        D_fake, C_fake, mF = discriminator(fake.detach()) 
        C_loss_fake = F.cross_entropy(C_fake, input_label)
        D_loss_fake = D_fake.mean()
        crF1, crF2 = crLoss(mF, input_label, opt.nclass_all)
        gradient_penalty = calc_gradient_penalty(discriminator, input_resv, fake.data)

        center_loss1 = Variable(torch.Tensor([0.0])).cuda()
        for i in range(data.ntrain_class):
            sample_idx = (input_label == i).data.nonzero().squeeze()
            if sample_idx.numel() == 0:
                center_loss1 += 0.0
            else:
                G_sample_cls = mF[sample_idx, :]
                center_loss1 += (G_sample_cls.mean(dim=0) - torch.from_numpy(data.tr_cls_centroid[i]).cuda()).pow(
                    2).sum().sqrt()

        center_loss2 = Variable(torch.Tensor([0.0])).cuda()
        for i in range(data.ntrain_class):
            sample_idx = (input_label == i).data.nonzero().squeeze()
            if sample_idx.numel() == 0:
                center_loss2 += 0.0
            else:
                G_sample_cls = mR[sample_idx, :]
                center_loss2 += (G_sample_cls.mean(dim=0) - torch.from_numpy(data.tr_cls_centroid[i]).cuda()).pow(
                    2).sum().sqrt()

        Wasserstein_D = D_loss_real - D_loss_fake
        D_cost = D_loss_fake - D_loss_real + gradient_penalty + 0.001 * D_loss_real ** 2 + (
                    center_loss1 + center_loss2) * opt.center_weight
        D_cost += (crR1 + crR2 + crF1 + crF2) * opt.ccf  
        D_cost += C_loss_fake * opt.cls_weight + C_loss_real * opt.cls_weight
        D_cost.backward()

        optimizerD.step()

    # train G
    for p in discriminator.parameters(): 
        p.requires_grad = False  

    netG.zero_grad()
    input_attv = Variable(input_att)
    noise.normal_(0, 1)
    noisev = Variable(noise)
    fake = netG(noisev, input_attv)
    D_fake, C_fake, mF = discriminator(fake) 
    G_loss = D_fake.mean()
    G_cost = -G_loss
    C_loss = F.cross_entropy(C_fake, input_label)
    center_loss = Variable(torch.Tensor([0.0])).cuda()
    for i in range(data.ntrain_class):
        sample_idx = (input_label == i).data.nonzero().squeeze()
        if sample_idx.numel() == 0:
            center_loss += 0.0
        else:
            G_sample_cls = fake[sample_idx, :]
            center_loss += (G_sample_cls.mean(dim=0) - torch.from_numpy(data.tr_cls_centroid[i]).cuda()).pow(
                2).sum().sqrt()

    errG = G_cost + opt.cls_weight * C_loss + center_loss * opt.center_weight
    errG.backward()
    optimizerG.step()

    if (start_step + 1) % opt.lr_dec_ep == 0:
        for param_group in optimizerD.param_groups:
            param_group['lr'] = param_group['lr'] * opt.lr_dec_rate
        for param_group in optimizerG.param_groups:
            param_group['lr'] = param_group['lr'] * opt.lr_dec_rate

    log_text = 'Iter-{}; Was_D: {:.3f}; G_loss: {:.3f}; D_loss_real: {:.3f};' \
               ' D_loss_fake: {:.3f}; center_loss: {:.3f}, c1:{:.3f},c2:{:.3f} crR1: {:.3f}, crR2: {:.3f}, crF1: {:.3f}, crF2: {:.3f} cls_loss:{:.3f}' \
        .format(start_step, Wasserstein_D.item(), G_loss.item(),
                D_loss_real.item(), D_loss_fake.item(), center_loss.item(), center_loss1.item(), center_loss2.item(),
                crR1, crR2, crF1, crF2, C_loss_real.item())
    print(log_text)
    # test
    if start_step != 0 and start_step % opt.test_epoch == 0:
        netG.eval()
        discriminator.eval()
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)
        train_z = train_X.cuda()
        test_z_seen = data.test_seen_feature.cuda()
        if opt.use_classify == True:
            cls = classifier.CLASSIFIER(train_X, train_Y, data, opt.nclass_all, True, discriminator, opt.lr, 0.5,
                                             50, opt.syn_num, True)
            acc_seen = cls.acc_seen
            acc_unseen = cls.acc_unseen

        else:
            _, _, train_z = discriminator(train_z)
            _, _, test_z_seen = discriminator(test_z_seen)
            test_z_unseen = data.test_unseen_feature.cuda()
            _, _, test_z_unseen = discriminator(test_z_unseen)
            pred_Y_s = torch.from_numpy(KNNPredict(train_z, train_Y, test_z_seen, k=opt.k))
            pred_Y_u = torch.from_numpy(KNNPredict(train_z, train_Y, test_z_unseen, k=opt.k))
            acc_seen = compute_per_class_acc_gzsl(data.test_seen_label, pred_Y_s, data.seenclasses)
            acc_unseen = compute_per_class_acc_gzsl(data.test_unseen_label, pred_Y_u, data.unseenclasses)

        H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
        print('U: %.1f, S: %.1f, H: %.1f' % (acc_unseen * 100, acc_seen * 100, H * 100))

        if final_result["H"] < H:
            final_result["H"] = H
            final_result["acc_seen"] = acc_seen
            final_result["acc_unseen"] = acc_unseen
        netG.train()
        discriminator.train()

print("result:")
print('%.1f, %.1f, %.1f' % (final_result["acc_unseen"] * 100, final_result["acc_seen"] * 100, final_result["H"] * 100))
print("time used:", time.time() - start)
