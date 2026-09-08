import argparse
import os
import os.path as osp
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torchvision import transforms
import torchvision 
from src.data.data import load_dataset
from src.models import load_model
from src.utils import loss
from src.utils.loss import CrossEntropyLabelSmooth

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from mrl import MRL_Linear_Layer, Matryoshka_CE_Loss
from aug import * 


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf'):
    
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    
    optimizer.zero_grad()
    
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomResizedCrop(32, scale=(0.8, 1.2), ratio=(0.75, 1.33), interpolation=2),
    transforms.RandomHorizontalFlip(),
    ImageJitter(jitter_param),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

NESTING_LIST = [8, 16, 32, 64, 128, 256, 512]
class MatryoshkaModel(nn.Module):
    def __init__(self, base_model=None, num_nested_models=None):
        super(MatryoshkaModel, self).__init__()
        resnet50 = load_model('ResNet18_10')
        self.base_model = nn.Sequential(*list(resnet50.children())[:-1])
        del resnet50
        self.base_model.add_module('Identity', nn.Identity())

        nest_list = [8, 16, 32, 64, 128, 256, 512]
        self.linear_layer = MRL_Linear_Layer(nesting_list=NESTING_LIST, num_classes=10, efficient=False)
        
        self.init_linear()

    def forward(self, x):
        features = self.base_model(x)
        features = torch.flatten(features, 1)
        nested_outputs = self.linear_layer(features)
        return nested_outputs

    def init_linear(self):
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    torch.backends.cudnn.enable = True
    torch.backends.cudnn.enable = True
    parser = argparse.ArgumentParser(description='train_source_os')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=2, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=20000, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=512, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='cifar10',
                        choices=['VISDA-C', 'office', 'officehome', 'office-caltech', 'domainnet126', 'cifar10',
                                 'cifar100'])
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--net', type=str, default='ResNet18_10',
                        help="vgg16, ResNet50_10, resnet101, vit, WideResNet_8,ResNet18_8")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="linear", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='./ckpt/models')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--ckpt', type=str, default='./ckpt_mod')
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['cosine', 'multistep'])
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--steps', type=list, default=[60, 120, 160, 2048])
    parser.add_argument('--num_nested_models', type=int, default=3, help="number of nested models")
    parser.add_argument('--relative_importance', type=float, default=0.1, help="relative importance of nested models")
    
    parser.add_argument('--model_name', type=str, default='ResNet18')
    args = parser.parse_args()

    run_name = './models_cifar_train'
    
    if args.dset == 'cifar10':
        
        torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True)

    train_dataset, train_loader = load_dataset(args.dset, args.data_dir, args.batch_size, args.worker,
                                               split='train', transforms=transform_train)
    test_dataset, test_loader = load_dataset(args.dset, args.data_dir, args.batch_size, args.worker, split='test',
                                             transforms=transform_test)

    model_name = args.net

    model_name = run_name
    

    root_dir = args.ckpt
    ckpt_dir = os.path.join(root_dir, 'ckpt')
    ckpt_dir = os.path.join(ckpt_dir, model_name)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    model_name = 'cifar10_pretrained_model_mrl_' + model_name + '.pt'
    
    model_name = os.path.join(ckpt_dir, model_name)
    
    model = MatryoshkaModel().cuda()
    
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    l1 = args.relative_importance
    l1 = [l1]
    l1 = l1* len(NESTING_LIST)
    
    mrl_loss = Matryoshka_CE_Loss(relative_importance=l1)

    CELOSS = nn.CrossEntropyLoss()
    model.train()
    max_acc = 0
    print('Traiing for %d epochs' % args.max_epoch)
    for epoch in range(args.max_epoch):
        model.train()
        train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            outputs = model(inputs)
            loss = mrl_loss(outputs, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if i % 400 == 0:
                print('epoch:%d,iter:%d,loss:%.4f' % (epoch, i, loss.item()))
        
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        acc_nested = [0] * len(NESTING_LIST)

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = model(inputs)
                for j, output in enumerate(outputs):
                    acc_nested[j] += (torch.max(output, 1)[1] == labels).float().sum().item()

        acc_nested = [a / len(test_loader.dataset) for a in acc_nested]

        print('epoch:%d' % epoch)
        for i, acc in enumerate(acc_nested):
            print(f'\t Accuracy of nested model {i}: nesting size: {NESTING_LIST[i]}: Accuracy {acc}')
        print('\n')            

        if max(acc_nested) > max_acc:
            max_acc = max(acc_nested)
            
            model_name_epoch_wise = 'cifar10_mrl__epoch_' + str(epoch) + '.pt'
            ckpt_saver = os.path.join(ckpt_dir, model_name_epoch_wise)
            torch.save(model.state_dict(), ckpt_saver)
            print('Model saved!, path :', ckpt_saver)
            
            
            epoch_number = epoch
        scheduler.step()


