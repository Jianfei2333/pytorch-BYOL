import torch
import torch.nn as nn
from torchvision import models

def _init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1 and len(m.weight.shape) > 1:
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.constant_(m.weight.bias)

class Normalization(nn.Module):
    def __init__(self, mean, std, n_channels=3):
        super(Normalization, self).__init__()
        self.n_channels=n_channels
        if mean is None:
            mean = [.5] * n_channels
        if std is None:
            std = [.5] * n_channels
        self.mean = torch.tensor(list(mean)).reshape((1, self.n_channels, 1, 1))
        self.std = torch.tensor(list(std)).reshape((1, self.n_channels, 1, 1))
        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
    
    def forward(self, x):
        y = (x - self.mean / self.std)
        return y

class Encoder(nn.Module):
    def __init__(self, mean, std, n_channels=3, backbone='resnet50'):
        super(Encoder, self).__init__()
        self.norm = Normalization(mean, std, n_channels)
        if backbone == 'resnet18':
            self.encoder = nn.Sequential(*list(models.resnet18(pretrained=False).children())[:-1]+[nn.Flatten()])
        else:
            self.encoder = nn.Sequential(*list(models.resnet50(pretrained=False).children())[:-1]+[nn.Flatten()])

        self.encoder.apply(_init_weight)

    def forward(self, x):
        x_norm = self.norm(x)
        return self.encoder(x_norm)

class Encoder_small(nn.Module):
    """ Change the first Convolution layer to 3x3, and remove the first pooling.
    """
    def __init__(self, mean, std, n_channels=3, backbone='resnet50'):
        super(Encoder_small, self).__init__()
        self.norm = Normalization(mean, std, n_channels)
        if backbone == 'resnet18':
            self.encoder = nn.Sequential(*list(models.resnet18(pretrained=False).children())[:-1]+[nn.Flatten()])
        else:    
            self.encoder = nn.Sequential(*list(models.resnet50(pretrained=False).children())[:-1]+[nn.Flatten()])
        self.encoder[0] = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3), bias=False)
        self.encoder[3] = nn.Identity()

        self.encoder.apply(_init_weight)

    def forward(self, x):
        x_norm = self.norm(x)
        return self.encoder(x_norm)

class MLP(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(MLP, self).__init__()
        self.n_hidden = n_hidden
        self.projector = nn.Sequential(
            nn.Linear(in_features=n_in, out_features=n_hidden, bias=False),
            nn.BatchNorm1d(num_features=n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=n_hidden, out_features=n_out, bias=False)
        )

    def forward(self, x):
        return self.projector(x)

class Branch(nn.Module):
    def __init__(self, encoder, projector):
        super(Branch, self).__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, x):
        fm = self.encoder(x)
        y = self.projector(fm)
        return y

class BYOL(nn.Module):
    def __init__(self, online_branch, target_branch, predictor):
        super(BYOL, self).__init__()
        self.online = online_branch
        self.target = target_branch
        self.predictor = predictor
        for param_q, param_k in zip(self.online.parameters(), self.target.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def forward(self, x_view1, x_view2):
        q1 = self.predictor(self.online(x_view1))
        q2 = self.predictor(self.online(x_view2))
        with torch.no_grad():
            t1 = self.target(x_view2)
            t2 = self.target(x_view1)

        return (q1, q2, t1, t2)

    def momentum_update(self, tau):
        for param_q, param_k in zip(self.online.parameters(), self.target.parameters()):
            param_k.data = param_k.data * tau + param_q.data * (1. - tau)

class Downstream(nn.Module):
    def __init__(self, encoder, n_class):
        super(Downstream, self).__init__()
        self.encoder = encoder
        self.n_class = n_class

        x = torch.rand((1, 3, 224, 224)).to(next(self.encoder.parameters()).device)
        _, c = self.encoder(x).shape
        n_channel = c

        self.classifier = nn.Linear(in_features=n_channel, out_features=n_class, bias=True)

    def forward(self, x):
        pass

class SemisupervisedFinetune(Downstream):
    def __init__(self, encoder, n_class):
        super(SemisupervisedFinetune, self).__init__(encoder, n_class)
    
    def forward(self, x):
        rep = self.encoder(x)
        output = self.classifier(rep)
        return output

class LinearEvaluator(Downstream):
    def __init__(self, encoder, n_class):
        super(LinearEvaluator, self).__init__(encoder, n_class)
    
    def forward(self, x):
        with torch.no_grad():
            rep = self.encoder(x)
        output = self.classifier(rep)
        return output