import torch.nn as nn
import torch

# G
class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(1024, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.fc3 = nn.Linear(opt.attSize + opt.nz, 1024)
        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc3(h))
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h)) 
        return h

# D
class D2(nn.Module):
    def __init__(self, opt):
        super(D2, self).__init__()
        self.discriminator = nn.Linear(opt.mapSize, 1)
        self.hidden = nn.Linear(opt.mapSize, 1024)
        self.classifier = nn.Linear(1024, opt.nclass_seen)
        self.logic = nn.LogSoftmax(dim=1)
        self.mapping = nn.Linear(opt.mapSize, 4096)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.mapping2 = nn.Linear(4096,opt.mapSize)
        self.apply(weights_init)

    def forward(self, x):
        m = self.lrelu(self.mapping(x))
        m = self.lrelu(self.mapping2(m))
        dis_out = self.discriminator(m) 
        h = self.lrelu(self.hidden(m))
        pred = self.logic(self.classifier(h))
        return dis_out, pred, m 


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

