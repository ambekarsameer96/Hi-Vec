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
from torchvision.models import resnet50
from .model_loader import MatryoshkaModel, NESTING_LIST, MatryoshkaModel_r18, NESTING_LIST_r18, MatryoshkaModel_r18_100, MatryoshkaModel_r18_8_2, MatryoshkaModel_imagenet_mrl, MatryoshkaModel_imagenet_mrl_mod, apply_blurpool, get_ckpt


def load_model(model_name, checkpoint_dir=None, domain=None):
    
    if model_name == 'Hendrycks2020AugMix_ResNeXt':
        model = Hendrycks2020AugMixResNeXtNet()
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, 'Hendrycks2020AugMix_ResNeXt.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))
            model.load_state_dict(torch.load(checkpoint_path))
        else:
            raise ValueError('No checkpoint path provided')

    elif model_name == 'resnet50_mrl_cifar10':
        
        
        model = MatryoshkaModel(num_classes = 10)
        
        print('Loading model from checkpoint', checkpoint_dir)
        checkpoint_dir = './ckpt_mod/ckpt/royal-butterfly-87/'
        if checkpoint_dir is not None:
            
            checkpoint_path = os.path.join(checkpoint_dir, 'cifar10_mrl__epoch_2034.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))
            model.load_state_dict(torch.load(checkpoint_path), strict=False)
            print('===================Model has been loaded from checkpoint===================', checkpoint_path)

    elif model_name == 'resnet18_mrl_cifar10':
        
        model = MatryoshkaModel_r18(num_classes = 10)
        
        print('Loading model from checkpoint', checkpoint_dir)
        checkpoint_dir = './ckpt_mod/ckpt/solar-disco-111/'
        if checkpoint_dir is not None:
            
            checkpoint_path = os.path.join(checkpoint_dir, 'cifar10_mrl__epoch_1796.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))
            model.load_state_dict(torch.load(checkpoint_path), strict=False)
            print('===================Model has been loaded from checkpoint===================', checkpoint_path)

    elif model_name == 'resnet18_mrl_cifar10_8_2':
        
        model = MatryoshkaModel_r18_8_2(num_classes = 8)
        
        print('Loading model from checkpoint', checkpoint_dir)
        
        checkpoint_dir = './ckpt_mod/cifar10_8_2/ckpt/pious-salad-141'
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, 'cifar10_mrl__epoch_9020.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))
            model.load_state_dict(torch.load(checkpoint_path), strict=False)
            print('===================Model has been loaded from checkpoint===================', checkpoint_path)

    elif model_name == 'resnet18_mrl_cifar100':
        
        model = MatryoshkaModel_r18_100(num_classes = 100)
        print('Loading model from checkpoint', checkpoint_dir)
        checkpoint_dir = './models/cifar100_saved'
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, 'cifar100_mrl__epoch_9250.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))    
            print('===================Model has been loaded from checkpoint===================', checkpoint_path)

   

    elif model_name == 'Hendrycks2020AugMix_ResNeXt_80':
        model = Hendrycks2020AugMixResNeXtNet(num_classes=80)
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, 'Hendrycks2020AugMix_ResNeXt_80.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))
            model.load_state_dict(torch.load(checkpoint_path))
    elif model_name == 'resnet50':
        model = resnet50_img(weights=ResNet50_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet50_mrl':
        
        path = './imagenet_mrl/final_weights_resnet50.pt'
        
        model = Resnet.resnet50(pretrained=False)
        nesting_list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        model.fc = ml.MRL_Linear_Layer(nesting_list, efficient=False)
        apply_blurpool(model)
        model.load_state_dict(get_ckpt(path))
        print('=========Loaded model from checkpoint=========', path)

        model = model.cuda()
        


        

    elif model_name == 'ResNet50_10':
        model = resnet50_cifar(num_classes=10)
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, 'ResNet50_10.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))
            model.load_state_dict(torch.load(checkpoint_path))
    elif model_name == 'ResNet50_100':
        model = resnet50_cifar(num_classes=100)
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, 'ResNet50_100.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))
            model.load_state_dict(torch.load(checkpoint_path))
    elif model_name == 'WideResNet':
        model = WideResNet()
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, 'WideResNet.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))
            model.load_state_dict(torch.load(checkpoint_path))
        else:
            raise ValueError('No checkpoint path provided')
    elif model_name == 'WideResNet_8':
        model = WideResNet(num_classes=8)
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, 'WideResNet_8.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))
            model.load_state_dict(torch.load(checkpoint_path))
    elif model_name == 'ResNet18_8':
        model = resnet18_cifar(num_classes=8)
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, 'ResNet18_8.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))
            model.load_state_dict(torch.load(checkpoint_path))
    elif model_name == 'ResNet18_10':
        model = resnet18_cifar(num_classes=10)
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, 'ResNet18_10.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))
            model.load_state_dict(torch.load(checkpoint_path))
    elif model_name == 'ResNet18_80':
        model = resnet18_cifar(num_classes=80)
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, 'ResNet18_80.pt')
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
    elif model_name == 'ResNet18_1000':
        model = resnet18_img(weights=ResNet18_Weights.IMAGENET1K)
    elif model_name == 'officehome_shot':
        model = OfficeHome_Shot()
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, 'officehome', domain, 'model.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))
            model.load_state_dict(torch.load(checkpoint_path))
    elif model_name == 'domainnet126_shot':
        model = DomainNet126_Shot()
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, 'domainnet126', domain, 'model.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))
            model.load_state_dict(torch.load(checkpoint_path))
    elif model_name == 'vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
    elif model_name == 'convnext_base':
        model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
    elif model_name == 'efficientnet_b0':
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    elif model_name == 'officehome_vit':
        model = OfficeHome_ViT()
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, 'officehome_vit', domain, 'model.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))
            model.load_state_dict(torch.load(checkpoint_path))
    elif model_name == 'domainnet126_vit':
        model = DomainNet126_ViT()
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, 'domainnet126_vit', domain, 'model.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))
            model.load_state_dict(torch.load(checkpoint_path))
    elif model_name == 'resnet50_cifar10':
        classifier = LinearClassifier(num_classes=10)
        ssh = SupConResNet().cuda()
        ext = ssh.encoder
        net = ExtractorHead(ext, classifier)
        if checkpoint_dir is not None:
            ckpt_path = os.path.join(checkpoint_dir, 'cifar10_joint_resnet50', 'ckpt.pth')
            if not os.path.exists(ckpt_path):
                raise ValueError('No checkpoint found at {}'.format(ckpt_path))
            ckpt = torch.load(ckpt_path)
            state_dict = ckpt['model']
            net_dict = {}
            head_dict = {}
            
    
        

        




        for k, v in state_dict.items():
            if k[:4] == "head":
                k = k.replace("head.", "")
                head_dict[k] = v
            else:
                k = k.replace("encoder.", "ext.")
                k = k.replace("fc.", "head.fc.")
                net_dict[k] = v
        net.load_state_dict(net_dict)
        model = net
    else:
        raise ValueError('Unknown model name')

    return model
