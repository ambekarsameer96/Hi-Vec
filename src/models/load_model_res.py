import os

import timm
from torchvision.models import resnet50 as resnet50_img, ResNet50_Weights, convnext_base, ConvNeXt_Base_Weights, \
    efficientnet_b0, \
    EfficientNet_B0_Weights, resnet18 as resnet18_img, ResNet18_Weights
from ..models import *
from .officehome_vit import OfficeHome_ViT
from .domainnet126_vit import DomainNet126_ViT
from .Res import resnet18 as resnet18_cifar, resnet50 as resnet50_cifar
from .BigResNet import SupConResNet, LinearClassifier
from .SSHead import ExtractorHead

# from .model_loader import MatryoshkaModel, NESTING_LIST, MatryoshkaModel_r18, NESTING_LIST_r18, MatryoshkaModel_r18_100


#keep only resnet models here in load model function model_name, checkpoint_dir=None, domain=None
def load_model_res(model_name, checkpoint_dir=None, domain=None):
    if model_name == 'resnet18':
        model = resnet18_img(pretrained=pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        return model
    elif model_name == 'ResNet18_10':
        model = resnet18_cifar(num_classes=10)
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, 'ResNet18_10.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))
            model.load_state_dict(torch.load(checkpoint_path))

    elif model_name == 'ResNet18_100':
        model = resnet18_cifar(num_classes=100)
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, 'ResNet18_100.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))
            model.load_state_dict(torch.load(checkpoint_path))

    elif model_name == 'ResNet18_8':
        model = resnet18_cifar(num_classes=8)

    return model