"""
Copyright to Tent Authors ICLR 2021 Spotlight
"""

from argparse import ArgumentDefaultsHelpFormatter
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

from torch.autograd import Variable


from src.methods.mrl_utils import MRL_output_select, MRL_gradient_select

from src.utils.utils import get_accuracy, merge_cfg_from_args, get_args

from src.methods.mrl_utils_new import MRL_auto_gradients_select_data_loop, MRL_merge, MRL_merge_cosine


args = get_args()
executed = False
class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)


    def forward(self, x):
        global executed
        if self.episodic:
            self.reset()
        if self.steps > 0:
            for _ in range(self.steps):
                if executed == False:
                    self.model, nmd = self.gradient_select(x, self.model, self.optimizer, args)

                    args.nmd = nmd
                    
                    executed = False
                
                outputs = forward_and_adapt(x, self.model, self.optimizer)
                
                self.model = MRL_merge_cosine(self.model, args)
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)
                
                outputs = MRL_output_select(outputs, args)
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

    def reset_steps(self, new_steps):
        self.steps = new_steps

    @staticmethod
    def configure_model(model):
        """Configure model for use with tent."""
        
        model.train()
        
        model.requires_grad_(False)
        
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
        return model



    @staticmethod
    def collect_params(model):
        """Collect the affine scale + shift parameters from batch norms.
        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    @staticmethod
    def check_model(model):
        """Check model for compatability with tent."""
        is_training = model.training
        assert is_training, "tent needs train mode: call model.train()"
        param_grads = [p.requires_grad for p in model.parameters()]
        has_any_params = any(param_grads)
        has_all_params = all(param_grads)
        assert has_any_params, "tent needs params to update: " \
                               "check which require grad"
        
        
        has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
        assert has_bn, "tent needs normalization for its optimization"


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

def mean_softmax_entropy(x:torch.Tensor)->torch.Tensor:
    temprature = 1
    x = x / temprature
    mean_probe_d=torch.mean(x.softmax(1),dim=0)
    entropy=-torch.sum(mean_probe_d*torch.log(mean_probe_d))
    return entropy


@torch.jit.script
def energy(x: torch.Tensor) -> torch.Tensor:
    """Energy calculation from logits."""
    temprature = 1
    x = -(temprature*torch.logsumexp(x / temprature, dim=1))
    if torch.rand(1) > 0.95:
        print(x.mean(0).item())
    return x


@torch.enable_grad()  
def forward_and_adapt(x, model, optimizer):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    
    outputs = model(x)
    outputs_mrl = MRL_output_select(outputs, args)
    loss = softmax_entropy(outputs_mrl).mean(0)
    
    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
    return outputs_mrl

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)