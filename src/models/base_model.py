import torch
from torch import nn
from .WideResNet import WideResNet
from src.utils.utils import split_up_model, split_up_model_MRL


class BaseModel(torch.nn.Module):
    """
    Change the model structure to perform the adaptation "AdaContrast" for other datasets
    """

    def __init__(self, model, arch_name):
        super().__init__()
        self.model = model
        if isinstance(model, WideResNet):
            self.nChannels = model.nChannels
        self.arch_name = arch_name
        if arch_name == 'vit':
            self._num_classes = model.head.out_features
            self._output_dim = model.head.in_features
        elif arch_name in ['officehome_vit', 'domainnet126_vit']:
            self._num_classes = model._num_classes
            self._output_dim = model._output_dim
        elif arch_name in ['resnet50_mrl_cifar10']:
            self.encoder, self.fc = split_up_model_MRL(model)
            # import pdb; pdb.set_trace()
            self._output_dim = 2048 
            #get in fea
        else:
            self.encoder, self.fc = split_up_model(model)
            if isinstance(self.fc, nn.Sequential):
                for module in self.fc.modules():
                    if isinstance(module, nn.Linear):
                        self._num_classes = module.out_features
                        self._output_dim = module.in_features
            elif isinstance(self.fc, nn.Linear):
                self._num_classes = self.fc.out_features
                self._output_dim = self.fc.in_features
            else:
                raise ValueError("Unable to detect output dimensions")

    def forward(self, x, return_feats=False):
        # 1) encoder feature
        if self.arch_name == 'vit':
            feat = self.model.forward_features(x)
            feat = self.model.fc_norm(feat[:, 0])
            logits = self.model.head(feat)
        elif self.arch_name in ['officehome_vit', 'domainnet126_vit']:
            return self.model(x, return_feats)
        elif self.arch_name in ['ResNet50_10', 'ResNet50_100']:
            feat = self.encoder(x)
            feat = feat.view(feat.size(0), -1)
            logits = self.fc(feat)
        else:
            feat = self.encoder(x)
            if self.arch_name == 'WideResNet':
                feat = torch.nn.functional.avg_pool2d(feat, 8)
                feat = feat.view(-1, self.nChannels)
            elif self.arch_name == 'vit':
                feat = feat[:, 0]

            elif self.arch_name == 'resnet50_mrl_cifar10':
                feat = self.encoder(x)
                feat = feat.view(feat.size(0), -1)
                logits = self.fc(feat)
                logits = logits[0]
                #get encoder features base_model
            feat = torch.flatten(feat, 1)
            logits = self.fc(feat)
            
            # elif self.arch_name == 'resnet50_mrl_cifar10':
            #     feat = self.encoder(x)
            #     feat = feat.view(feat.size(0), -1)
            #     logits = self.fc(feat)
            #     #get encoder features base_model
                

                

        if return_feats:
            return feat, logits
        return logits

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def output_dim(self):
        return self._output_dim

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()
