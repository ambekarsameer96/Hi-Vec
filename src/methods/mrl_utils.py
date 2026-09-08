import numpy as np
import torch
import torch.nn.functional as F



def MRL_output_select(outputs, args):
    num_models = len(outputs)
    nmd = args.nmd
    output = outputs[nmd].clone()
    return output

def MRL_gradient_select(model, args):
    nmd = args.nmd
    for i, (name, p) in enumerate(model.named_parameters()):
        fc_name = 'classifier_' + str(nmd)
        if 'classifier' in name and fc_name not in name:
            p.requires_grad = False
        
        if 'classifier' in name and fc_name in name:
            p.requires_grad = True
    return model