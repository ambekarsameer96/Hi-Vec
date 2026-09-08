from mrl import MRL_Linear_Layer
from torchvision.models import resnet as Resnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy 
import timm
# NESTING_LIST = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]

# class MatryoshkaModel(nn.Module):

#     def __init__(self, base_model=None,num_classes = 10, num_nested_models=None):

#         super(MatryoshkaModel, self).__init__()
#         self.base_model = Resnet.__dict__['resnet50'](pretrained=True)
#         #remove fc layer
#         self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])

#         self.base_model.add_module('Identity', nn.Identity())
#         # nest_list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
#         nest_list = NESTING_LIST
        

#         self.linear_layer = MRL_Linear_Layer(nesting_list=NESTING_LIST, num_classes=num_classes, efficient=False)
#         self.model = nn.Sequential(self.base_model, self.linear_layer)

#     def forward(self, x):
#         features = self.base_model(x)
#         # import pdb ; pdb.set_trace()
#         features = torch.flatten(features, 1)
#         nested_outputs = self.linear_layer(features)

#         return nested_outputs

# from .load_model import load_model
from .load_model_res import load_model_res as load_model


NESTING_LIST = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
class MatryoshkaModel(nn.Module):
    def __init__(self, base_model=None, num_classes = 10, num_nested_models=None):
        super(MatryoshkaModel, self).__init__()
        # self.base_model = base_model
        #use resnet50 as the base model without the last laye, create a new resnet model object 
        #pytorch create a new model object with pretrained resnet 50 
        # resnet50_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
        # self.base_model = nn.Sequential(*list(resnet50_model.children())[:-1])
        #renset 50 with 32*32 input
        resnet50 = load_model('ResNet50_10')
        #remove the last layer
        self.base_model = nn.Sequential(*list(resnet50.children())[:-1])
        #add identity layer
        self.base_model.add_module('Identity', nn.Identity())

        # self.nesting_list = nn.ModuleList([nn.Linear(2048, 10) for _ in range(num_nested_models)])
        # nest_list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        NESING_LIST = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        self.linear_layer = MRL_Linear_Layer(nesting_list=NESTING_LIST, num_classes=10, efficient=False)
        #make linear layer 
        # self.model = nn.Sequential(self.base_model, self.linear_layer)
        #merge the base model and the linear layer
        # self.model = nn.Sequential(self.base_model, self.linear_layer)
        # import pdb; pdb.set_trace()


    def forward(self, x):
        features = self.base_model(x)
        features = torch.flatten(features, 1)
        nested_outputs = self.linear_layer(features)
        #flatten 
        # import pdb; pdb.set_trace()
        # return self.linear_layer(features)
        return nested_outputs


#For resnet 18 now 

NESTING_LIST_r18 = [8, 16, 32, 64, 128, 256, 512]
class MatryoshkaModel_r18(nn.Module):
    def __init__(self, base_model=None, num_classes = 10, num_nested_models=None):
        super(MatryoshkaModel_r18, self).__init__()
        # self.base_model = base_model
        #use resnet50 as the base model without the last laye, create a new resnet model object 
        #pytorch create a new model object with pretrained resnet 50 
        # resnet50_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
        # self.base_model = nn.Sequential(*list(resnet50_model.children())[:-1])
        #renset 50 with 32*32 input
        # resnet50 = load_model('ResNet50_10')
        resnet50 = load_model('ResNet18_10')
        #remove the last layer
        self.base_model = nn.Sequential(*list(resnet50.children())[:-1])
        #add identity layer
        self.base_model.add_module('Identity', nn.Identity())

        # self.nesting_list = nn.ModuleList([nn.Linear(2048, 10) for _ in range(num_nested_models)])
        # nest_list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        NESING_LIST_r18 = [8, 16, 32, 64, 128, 256, 512, ]
        self.linear_layer = MRL_Linear_Layer(nesting_list=NESING_LIST_r18, num_classes=10, efficient=False)
        #make linear layer 
        # self.model = nn.Sequential(self.base_model, self.linear_layer)
        #merge the base model and the linear layer
        # self.model = nn.Sequential(self.base_model, self.linear_layer)
        # import pdb; pdb.set_trace()


    def forward(self, x):
        features = self.base_model(x)
        features = torch.flatten(features, 1)
        nested_outputs = self.linear_layer(features)
        #flatten 
        # import pdb; pdb.set_trace()
        # return self.linear_layer(features)
        return nested_outputs

class MatryoshkaModel_r18_100(nn.Module):
    def __init__(self, base_model=None, num_classes = 100, num_nested_models=None):
        super(MatryoshkaModel_r18_100, self).__init__()
        # self.base_model = base_model
        #use resnet50 as the base model without the last laye, create a new resnet model object 
        #pytorch create a new model object with pretrained resnet 50 
        # resnet50_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
        # self.base_model = nn.Sequential(*list(resnet50_model.children())[:-1])
        #renset 50 with 32*32 input
        # resnet50 = load_model('ResNet50_10')
        resnet50 = load_model('ResNet18_10')
        #remove the last layer
        self.base_model = nn.Sequential(*list(resnet50.children())[:-1])
        #add identity layer
        self.base_model.add_module('Identity', nn.Identity())

        # self.nesting_list = nn.ModuleList([nn.Linear(2048, 10) for _ in range(num_nested_models)])
        # nest_list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        NESING_LIST_r18 = [8, 16, 32, 64, 128, 256, 512, ]
        self.linear_layer = MRL_Linear_Layer(nesting_list=NESING_LIST_r18, num_classes=100, efficient=False)
        #make linear layer 
        # self.model = nn.Sequential(self.base_model, self.linear_layer)
        #merge the base model and the linear layer
        # self.model = nn.Sequential(self.base_model, self.linear_layer)
        # import pdb; pdb.set_trace()


    def forward(self, x):
        features = self.base_model(x)
        features = torch.flatten(features, 1)
        nested_outputs = self.linear_layer(features)
        #flatten 
        # import pdb; pdb.set_trace()
        # return self.linear_layer(features)
        return nested_outputs







NESTING_LIST_r18 = [8, 16, 32, 64, 128, 256, 512]
class MatryoshkaModel_r18_8_2(nn.Module):
    def __init__(self, base_model=None, num_classes = 8, num_nested_models=None):
        super(MatryoshkaModel_r18_8_2, self).__init__()
        # self.base_model = base_model
        #use resnet50 as the base model without the last laye, create a new resnet model object 
        #pytorch create a new model object with pretrained resnet 50 
        # resnet50_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
        # self.base_model = nn.Sequential(*list(resnet50_model.children())[:-1])
        #renset 50 with 32*32 input
        # resnet50 = load_model('ResNet50_10')
        # resnet50 = load_model('ResNet18_10')
        resnet50 = load_model('ResNet18_8')
        #remove the last layer
        self.base_model = nn.Sequential(*list(resnet50.children())[:-1])
        #add identity layer
        self.base_model.add_module('Identity', nn.Identity())

        # self.nesting_list = nn.ModuleList([nn.Linear(2048, 10) for _ in range(num_nested_models)])
        # nest_list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        NESING_LIST_r18 = [8, 16, 32, 64, 128, 256, 512, ]
        self.linear_layer = MRL_Linear_Layer(nesting_list=NESING_LIST_r18, num_classes=8, efficient=False)
        #make linear layer 
        # self.model = nn.Sequential(self.base_model, self.linear_layer)
        #merge the base model and the linear layer
        # self.model = nn.Sequential(self.base_model, self.linear_layer)
        # import pdb; pdb.set_trace()


    def forward(self, x):
        features = self.base_model(x)
        features = torch.flatten(features, 1)
        nested_outputs = self.linear_layer(features)
        #flatten 
        # import pdb; pdb.set_trace()
        # return self.linear_layer(features)
        return nested_outputs




class MatryoshkaModel_imagenet_mrl_mod(nn.Module):

    def __init__(self, base_model=None, num_nested_models=None, num_classes=1000, model_path = None, model_ckpt = None):

        super(MatryoshkaModel_imagenet_mrl_mod, self).__init__()
        # self.base_model = Resnet.__dict__['resnet50'](pretrained=True)
        #import timm r50 
        # self.base_model = timm.create_model('resnet50', pretrained=False)
        # r50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        #import from torchvision resnet 50 
        # from torchvision.models import resnet as Resnet
        r50 = timm.create_model('resnet50_gn', pretrained=False)
        # r50 = Resnet.resnet50(pretrained=False)
        
        
    
        # self.base_model = copy.deepcopy(r50)
        

        #remove fc layer
        # self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])

        # self.base_model.add_module('Identity', nn.Identity())
        nest_list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        
        r50.fc = MRL_Linear_Layer(nesting_list=NESTING_LIST, num_classes=num_classes, efficient=False)
        # self.fc = MRL_Linear_Layer(nesting_list=nest_list, num_classes=num_classes, efficient=False)
        # self.model = nn.Sequential(self.base_model, self.linear_layer)
        # apply_blurpool(r50)
        # import pdb ; pdb.set_trace()
        #model loader
        # path = './imagenet_mrl/final_weights_r50.pt'
        path = model_path
        r50.load_state_dict(get_ckpt_mod(path), strict=False)
        # self.fc = copy.deepcopy(r50.fc)
        self.fc = MRL_Linear_Layer(nesting_list=NESTING_LIST, num_classes=num_classes, efficient=False)
        self.fc.load_state_dict(r50.fc.state_dict())
        del r50.fc
        #now seperate the models 
        self.base_model = copy.deepcopy(r50)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
        self.base_model.add_module('Identity', nn.Identity())
        #copy without the fc layers to base_model 
        # self.base_model = nn.Sequential(*list(r50.children())[:-1])
        # import pdb ; pdb.set_trace()
        
        # import pdb ; pdb.set_trace()
        del r50




    def forward(self, x):
        features = self.base_model(x)
        # import pdb ; pdb.set_trace()
        features = torch.flatten(features, 1)
        # import pdb ; pdb.set_trace()
        nested_outputs = self.fc(features)

        return nested_outputs


class MatryoshkaModel_imagenet_mrl(nn.Module):

    def __init__(self, base_model=None, num_nested_models=None, num_classes=1000, model_ckpt = None):

        super(MatryoshkaModel_imagenet_mrl, self).__init__()
        # self.base_model = Resnet.__dict__['resnet50'](pretrained=True)
        #import timm r50 
        # self.base_model = timm.create_model('resnet50', pretrained=False)
        r50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        
    
        # self.base_model = copy.deepcopy(r50)
        

        #remove fc layer
        # self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])

        # self.base_model.add_module('Identity', nn.Identity())
        nest_list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        
        r50.fc = MRL_Linear_Layer(nesting_list=NESTING_LIST, num_classes=num_classes, efficient=False)
        # self.fc = MRL_Linear_Layer(nesting_list=nest_list, num_classes=num_classes, efficient=False)
        # self.model = nn.Sequential(self.base_model, self.linear_layer)
        apply_blurpool(r50)

        #model loader
        path = './imagenet_mrl/final_weights_r50.pt'
        r50.load_state_dict(get_ckpt(path))
        # self.fc = copy.deepcopy(r50.fc)
        self.fc = MRL_Linear_Layer(nesting_list=NESTING_LIST, num_classes=num_classes, efficient=False)
        self.fc.load_state_dict(r50.fc.state_dict())
        del r50.fc
        #now seperate the models 
        self.base_model = copy.deepcopy(r50)
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
        self.base_model.add_module('Identity', nn.Identity())
        #copy without the fc layers to base_model 
        # self.base_model = nn.Sequential(*list(r50.children())[:-1])
        # import pdb ; pdb.set_trace()
        
        # import pdb ; pdb.set_trace()
        del r50




    def forward(self, x):
        features = self.base_model(x)
        # import pdb ; pdb.set_trace()
        features = torch.flatten(features, 1)
        # import pdb ; pdb.set_trace()
        nested_outputs = self.fc(features)

        return nested_outputs

class BlurPoolConv2d(torch.nn.Module):
	def __init__(self, conv):
		super().__init__()
		default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
		filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
		self.conv = conv
		self.register_buffer('blur_filter', filt)

	def forward(self, x):
		blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
						   groups=self.conv.in_channels, bias=None)
		return self.conv.forward(blurred)

def apply_blurpool(mod: torch.nn.Module):
	for (name, child) in mod.named_children():
		if isinstance(child, torch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16):
			setattr(mod, name, BlurPoolConv2d(child))
		else: apply_blurpool(child)
  

def get_ckpt(path):
	ckpt=path
	ckpt = torch.load(ckpt, map_location='cpu')
	plain_ckpt={}
	for k in ckpt.keys():
		plain_ckpt[k[7:]] = ckpt[k] # remove the 'module' portion of key if model is Pytorch DDP
	return plain_ckpt

def get_ckpt_mod(path):
    ckpt=path
    ckpt = torch.load(ckpt, map_location='cpu')
    plain_ckpt={}
    for k in ckpt.keys():
        #use .replace to remove the module. part of the key
        plain_ckpt[k.replace('module.', '')] = ckpt[k] # remove the 'module' portion of key if model is Pytorch DDP
        #replace bn with gn 
        if 'bn' in k:
            plain_ckpt[k.replace('bn', 'gn')] = ckpt[k]
    return plain_ckpt