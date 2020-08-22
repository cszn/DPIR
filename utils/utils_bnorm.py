import torch
import torch.nn as nn


def deleteLayer(model, layer_type=nn.BatchNorm2d):
    for k, m in list(model.named_children()):
        if isinstance(m, layer_type):
            del model._modules[k]
        deleteLayer(m, layer_type)


def merge_bn(model):
    ''' by Kai Zhang, 11/01/2019.
    '''
    prev_m = None
    for k, m in list(model.named_children()):
        if (isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)) and (isinstance(prev_m, nn.Conv2d) or isinstance(prev_m, nn.Linear) or isinstance(prev_m, nn.ConvTranspose2d)):

            w = prev_m.weight.data

            if prev_m.bias is None:
                zeros = torch.Tensor(prev_m.out_channels).zero_().type(w.type())
                prev_m.bias = nn.Parameter(zeros)
            b = prev_m.bias.data

            invstd = m.running_var.clone().add_(m.eps).pow_(-0.5)
            if isinstance(prev_m, nn.ConvTranspose2d):
                w.mul_(invstd.view(1, w.size(1), 1, 1).expand_as(w))
            else:
                w.mul_(invstd.view(w.size(0), 1, 1, 1).expand_as(w))
            b.add_(-m.running_mean).mul_(invstd)
            if m.affine:
                if isinstance(prev_m, nn.ConvTranspose2d):
                    w.mul_(m.weight.data.view(1, w.size(1), 1, 1).expand_as(w))
                else:
                    w.mul_(m.weight.data.view(w.size(0), 1, 1, 1).expand_as(w))
                b.mul_(m.weight.data).add_(m.bias.data)

            del model._modules[k]
        prev_m = m
        merge_bn(m)


def add_bn(model, for_init=True):
    ''' by Kai Zhang, 11/01/2019.
    '''
    for k, m in list(model.named_children()):
        if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d)):
            if for_init:
                b = nn.BatchNorm2d(m.out_channels, momentum=None, affine=False, eps=1e-04)
            else:
                b = nn.BatchNorm2d(m.out_channels, momentum=0.1, affine=True, eps=1e-04)
                b.weight.data.fill_(1)

            new_m = nn.Sequential(model._modules[k], b)
            model._modules[k] = new_m
        add_bn(m, for_init)


def deploy_sequential(model):
    ''' by Kai Zhang, 11/01/2019.
    singleton children
    '''
    for k, m in list(model.named_children()):
        if isinstance(m, nn.Sequential):
            if m.__len__() == 1:
                model._modules[k] = m.__getitem__(0)
        deploy_sequential(m)

