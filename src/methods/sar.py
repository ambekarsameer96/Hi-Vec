"""
Copyright to SAR Authors, ICLR 2023 Oral (notable-top-5%)
built upon on Tent code.
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import math
import numpy as np
from ..utils.sam import SAM




from src.methods.mrl_utils import MRL_output_select, MRL_gradient_select

from src.utils.utils import get_accuracy, merge_cfg_from_args, get_args

from src.methods.mrl_utils_new import MRL_auto_gradients_select_data_loop, MRL_merge, MRL_merge_cosine, MRL_merge_mean, MRL_auto_gradients_select_data_loop_rgn, MRL_merge_mean_position, mutual_info_score


args = get_args()


executed = False


def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data


class SAR(nn.Module):
    """SAR online adapts a model by Sharpness-Aware and Reliable entropy minimization during testing.
    Once SARed, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, lr, batch_size, num_classes, episodic=False, reset_constant=0.2, steps=1, e_margin=0.4*math.log(10000)):
        super().__init__()
        self.model, self.optimizer = self.prepare_SAR_model_and_optimizer(model,lr ,batch_size)
        self.steps = steps
        
        self.episodic = episodic

        self.margin_e0 = e_margin  
        self.reset_constant_em = reset_constant  
        self.ema = None  

        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)


    def forward(self, x):
        global executed
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            if executed == False:
                

                self.model, nmd = self.gradient_select(x, self.model, self.optimizer, args)

                args.nmd = nmd
                
                executed = False
            
            outputs, ema, reset_flag, mi_score  = forward_and_adapt_sar(x, self.model, self.optimizer, self.margin_e0, self.reset_constant_em, self.ema)
            if reset_flag:
                self.reset()
            self.ema = ema  

            if mi_score is True:
                
                model_merge = args.model_merge
                if model_merge == 'normal':
                    self.model = MRL_merge(self.model, args)
                elif model_merge == 'cosine':
                    self.model = MRL_merge_cosine(self.model, args)

                elif model_merge == 'mean':
                    self.model = MRL_merge_mean(self.model, args)

                elif model_merge == 'mean_position':
                    self.model = MRL_merge_mean_position(self.model, args)

                elif model_merge == 'none':
                    pass
                else:
                    raise Exception("Model merge not implemented")
                    exit()
            
            


        return outputs
    @torch.enable_grad()
    def gradient_select(self, data, model, optimizer, args):
        
        model, nmd = MRL_auto_gradients_select_data_loop(data, model, optimizer, args)

        return model, nmd

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None

    @staticmethod
    def configure_model(model):
        """Configure model for use with SAR."""
        
        model.train()
        
        model.requires_grad_(False)
        
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
            
            if isinstance(m, nn.Linear):
                m.requires_grad_(True)
        return model

    @staticmethod
    def prepare_SAR_model_and_optimizer(model, lr, batch_size):
        model = SAR.configure_model(model)
        params, param_names = collect_params(model)
        
        
        base_optimizer = torch.optim.SGD
        if batch_size == 1:
            lr = 2 * lr
        optimizer = SAM(params, base_optimizer, lr=lr, momentum=0.9)
        return model, optimizer


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  
def forward_and_adapt_sar(x, model, optimizer, margin, reset_constant, ema):
    """Forward and adapt model input data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    optimizer.zero_grad()
    
    
    outputs = model(x)
    output_1_copy1 = tuple(tensor.clone() for tensor in outputs)
    
    outputs = MRL_output_select(outputs, args)
    
    entropys = softmax_entropy(outputs)
    filter_ids_1 = torch.where(entropys < margin)
    entropys = entropys[filter_ids_1]
    loss = entropys.mean(0)
    loss.backward()

    mi_score = mutual_info_score(output_1_copy1,args)
    del output_1_copy1
    if mi_score is True:
        optimizer.first_step(zero_grad=True) 
    
    
    ent_output = model(x)
    
    
    ent_outputs = MRL_output_select(ent_output, args)
    
    entropys2 = softmax_entropy(ent_outputs)
    entropys2 = entropys2[filter_ids_1]  
    loss_second_value = entropys2.clone().detach().mean(0)
    filter_ids_2 = torch.where(entropys2 < margin)  
    loss_second = entropys2[filter_ids_2].mean(0)
    if not np.isnan(loss_second.item()):
        ema = update_ema(ema, loss_second.item())  

    
    loss_second.backward()


    if mi_score is True:
        optimizer.second_step(zero_grad=True)

    reset_flag = False
    if ema is not None:
        if ema < reset_constant:
            print(f"ema < reset_constant: {reset_constant}, now reset the model")
            reset_flag = True

    return outputs, ema, reset_flag, mi_score


def collect_params(model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        
        if 'layer4' in nm:
            continue
        if 'conv5_x' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def check_model(model):
    """Check model for compatability with SAR."""
    is_training = model.training
    assert is_training, "SAR needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "SAR needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "SAR should not update all params: " \
                               "check which require grad"
    has_norm = any([isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)) for m in model.modules()])
    assert has_norm, "SAR needs normalization layer parameters for its optimization"
