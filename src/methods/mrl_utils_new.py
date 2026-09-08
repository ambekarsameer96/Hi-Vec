import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import re
#import wandb from main 
# from tta_28_10 import wandb
# def MRL_output_select(outputs):

#     #detetct the number of nested models through shape 
#     num_models = len(outputs)
#     #select first one as the reference but retain grad 
#     output = outputs[0].clone()
#     return output
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchmetrics.clustering import MutualInfoScore



def mutual_info_score(outputs, args):

    best_classifier = args.nmd
    mi_metric  = MutualInfoScore()

    # outputs_clone = outputs.copy()
    # outputs_clone = copy.deepcopy(outputs)
    outputs_clone = tuple(tensor.clone() for tensor in outputs)
    best_classifier_output = outputs_clone[best_classifier]
    # import pdb; pdb.set_trace()


    #best classifier output
    # best_classifier_output = outputs_clone[best_classifier]
    #remove this from the outputs
    # outputs_clone.pop(best_classifier)
    #calculate the mutual info score
    mi_scores = []
    for i, output in enumerate(outputs_clone):
        if i==int(best_classifier):
            continue

        inp1 = best_classifier_output.unsqueeze(0)
        inp2 = output.unsqueeze(0)
        # inp1 = inp1.view(-1)
        # inp2 = inp2.view(-1)
        inp1 = inp1.view(-1).to(torch.int64)
        inp2 = inp2.view(-1).to(torch.int64)

        # mi_score = mi_metric(best_classifier_output, output)
        mi_score = mi_metric(inp1, inp2)
        mi_scores.append(mi_score)

    # import pdb; pdb.set_trace()
    
    mean_mi_score = torch.mean(torch.stack(mi_scores))
    # import pdb; pdb.set_trace()
    print('Mean MI score:', mean_mi_score)

    mi_threshold = args.mi_threshold
    if mean_mi_score < mi_threshold:
        return True
    else:
        return False






def MRL_output_select(outputs, args):
    #detetct the number of nested models through shape 
    num_models = len(outputs)
    nmd = args.nmd
    #select first one as the reference but retain grad 
    # output = outputs[0].clone()
    output = outputs[nmd].clone()
    return output

def MRL_gradient_select(model, args):
    #detetct the number of nested models through shape 
    # num_models = len(grads)
    nmd = args.nmd
    #disable gradient for fc layers except the nmd model
    # print('Model name:', model)
    # for name, param in model.named_parameters():
    #     print('Name:', name,'Param:', param)

    # import pdb; pdb.set_trace()
    for i, (name, p) in enumerate(model.named_parameters()):
        fc_name = 'classifier_' + str(nmd)
        # if 'fc' in name and i != nmd:
        #     import pdb; pdb.set_trace()
        # if 'classifier' in name and fc_name not in name:
        #     p.requires_grad = False
        # #enable gradient update for all 
        # if 'classifier' in name and fc_name in name:
        #     p.requires_grad = True
        #     print('Gradient enabled for:', name)
        #enable gradient for all 
        if 'classifier' in name:
            p.requires_grad = True
            # print('Gradient enabled for:', name)

    
    # #print the layers which have grad false
    # print('Layers with gradient false')
    # for name, param in model.named_parameters():
    #     if param.requires_grad == False:
    #         print('Name:', name,'Param:', param)
    # import pdb; pdb.set_trace()
    
    # print('========Gradient update disabled for all fc layers except classifier_' + str(nmd) + '========')

    return model




def MRL_auto_gradients_select(d1):
    weights = []
    # detatch first
    d1 = {k: v.detach() for k, v in d1.items()}
    all_weights = []
    all_bias = []
    layer_gradient = {}
    for k, v in d1.items():
        # check if bn and weight is present
        #check if downsample is not present
        if 'classifier' in k:
            if 'weight' in k:
                all_weights.append(v)
                # print('Key and value are: ', k, v)
                # print('Key is : ', k)
                #add the layer name and gradient to the dictionary
                layer_gradient[k] = v
    
    # import pdb; pdb.set_trace()

    #calculate whoch MRL layer has the highest gradoent value
    # max_value = max(layer_gradient.values())
    #calaulcate torch.norm for each layer
    max_value = min([torch.norm(v) for v in layer_gradient.values()])

    # max_value = min([torch.var(v) for v in layer_gradient.values()])
    
    max_key = [k for k, v in layer_gradient.items() if torch.norm(v) == max_value]
    # print('Max value:', max_value)
    # print('Max key:', max_key)
    #extract the number from key 
    # nmd = int(max_key[0].split('_')[-1])
    # import pdb; pdb.set_trace()
    max_key = str(max_key)
    number = int(re.search(r'_(\d+)', max_key).group(1))
    nmd = number
    # import pdb; pdb.set_trace()
            
    # all_weight_gradients = all_weights
    # target_size = 512
    # for i, tensor in enumerate(all_weight_gradients):
    #     current_size = tensor.shape[-1]
    #     pad_size = target_size - current_size
    #     if pad_size > 0:
    #         all_weight_gradients[i] = torch.nn.functional.pad(tensor, (0, pad_size), 'constant', 0).to(device)
    # all_bias_gradients = all_bias
    # for i, tensor in enumerate(all_bias_gradients):
    #     current_size = tensor.shape[-1]
    #     pad_size = target_size - current_size
    #     if pad_size > 0:
    #         all_bias_gradients[i] = torch.nn.functional.pad(tensor, (0, pad_size), 'constant', 0).to(device)
    # weights = torch.stack(all_weight_gradients, 0).to(device)
    # bias = torch.stack(all_bias_gradients, 0).to(device)
    # all_weight_gradients = torch.stack(all_weights).to(device)
    # all_weight_gradients = all_weight_gradients.squeeze()
    # import pdb; pdb.set_trace()
    
    # all_bias_gradients = torch.stack(all_bias).to(device)
    
    # import pdb; pdb.set_trace()
    # return weights, bias
    return nmd


def MRL_auto_gradients_select_rgn(d1):
    weights = []
    # detatch first
    # d1 = {k: v() for k, v in d1.items()}
    d1 = {k: v for k, v in d1.items()}
    all_weights = []
    all_bias = []
    layer_rgn = {}

    for k, v in d1.items():
        # check if bn and weight is present
        #check if downsample is not present
        if 'classifier' in k:
            if 'weight' in k:
                all_weights.append(v)
                # print('Key and value are: ', k, v)
                # print('Key is : ', k)
                #add the layer name and gradient to the dictionary
                layer_rgn[k] = v

    #calculate whoch MRL layer has the highest gradoent value
    # max_value = max(layer_gradient.values())
    #calculate min thats it 
    max_value = min([v for v in layer_rgn.values()])
    min_key = [k for k, v in layer_rgn.items() if v == max_value]
    #extract the number from key
    # ['linear_layer.nesting_classifier_0.weight'] from this extract number 0 
    # nmd = int(min_key[0].split('_')[-1])
    nmd = int(min_key[0].split('_')[-1].split('.')[0])
    

    

    
    return nmd

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    #temprature = 1.1 #0.9 #1.2
    #x = x ** temprature #torch.unsqueeze(temprature, dim=-1)
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def MRL_auto_gradients_select_data_loop(data, model, optimizer, args):
    nmd  = args.nmd
    #copy_optimizer and model
    # optimizer_copy = copy.deepcopy(optimizer)
    
    model_copy = copy.deepcopy(model)
    #set model to train 
    optimizer_copy = torch.optim.Adam(model_copy.parameters(), lr=0.001)
    model_copy.train()
    # optimizer_copy = optimizer_copy.to(device)

    data = data.to(device)
    model_copy = model_copy.to(device)
    # if final_backward != 0:
    #     loss.backward()
        #gradient track for MRL
    #enable gardient for all linear layers 
    # import pdb; pdb.set_trace()
    for name, param in model_copy.named_parameters():
        if 'classifier' in name:
            param.requires_grad = True
            # print('Gradient enabled for:', name)
            # import pdb; pdb.set_trace()
        #if its linear layer 
        if isinstance(param, torch.nn.Linear):
            param.requires_grad = True
            # print('Gradient enabled for:', name)
            # import pdb; pdb.set_trace()

    # if 1==1:
    # Matryoshka_Adapt_Loss(softmax_entropy)
    entropy_loss_2 =  Matryoshka_Adapt_Loss(softmax_entropy)
    if 1==1:
        optimizer_copy.zero_grad()
        outputs = model_copy(data)
        loss = entropy_loss_2.forward_entropy_mean(outputs)
        loss = loss.mean(0)
        # import pdb; pdb.set_trace()
        
        loss.backward()
        auto_gradient = True
        args.auto_gradient = auto_gradient
        if args.auto_gradient:
            
            # count = 0
            scores = {}
            fc_layer_list = []
            for name, m in model_copy.named_modules():
            #for fc layer 
                if isinstance(m, torch.nn.Linear):
                    #take both gradients with weight and bias as keys 
                    # # scores[name] = torch.clone(m.weight.grad.clone()).detach()
                    # print('Name is : ', name)
                    # print('Shape of weight is : ', m)
                    # import pdb; pdb.set_trace()
                    #check if gradient exists 
                    if m.weight.grad is None:
                        print('Gradient does not exist for:', name)
                        continue
                    scores[name + '.weight'] = torch.clone(m.weight.grad.clone()).detach()
                    # scores[name + '.bias'] = torch.clone(m.bias.grad.clone()).detach()
                    fc_layer_list.append(name)
                    # print('Gradient extracted for:', name)


            all_scores = torch.cat([torch.flatten(v) for v in scores.values()])
            grad_flow = torch.sum(all_scores)
            nmd = MRL_auto_gradients_select(scores)
            # Start comment 
            # print('Updated NMD from the previous:', args.nmd , 'to:', nmd)
            # End comment
            # wandb.log('nmd updated to:', nmd)
            # args.nmd = nmd
            # dimension = nmd
            # nmd_global = nmd
            #call configure model to update the gradients
            #turn on gradient only for the selected fc layer based on nmd 
            for i, (name, p) in enumerate(model.named_parameters()):
                fc_name = 'classifier_' + str(nmd)
                # if 'fc' in name and i != nmd:
                #     import pdb; pdb.set_trace()
                if 'classifier' in name and fc_name not in name:
                    p.requires_grad = False
                    # print('Gradient disabled for:', name)
                # #enable grad for nmd 
                if 'classifier' in name and fc_name in name:
                    p.requires_grad = True
                    # print('Gradient enabled for:', name)
                #enable grad for all classifiers
                # if 'classifier' in name:
                #     p.requires_grad = True
                    
            # count = 2
            # executed = True
            


            
            # optimizer_copy.step()
            # optimizer_copy.zero_grad()

            return model, nmd


#
def MRL_auto_gradients_select_data_loop_rgn(data, model, optimizer, args):
    nmd  = args.nmd
    #copy_optimizer and model
    # optimizer_copy = copy.deepcopy(optimizer)
    
    model_copy = copy.deepcopy(model)
    #set model to train 
    optimizer_copy = torch.optim.Adam(model_copy.parameters(), lr=0.001)
    model_copy.train()
    # optimizer_copy = optimizer_copy.to(device)

    data = data.to(device)
    model_copy = model_copy.to(device)
    # if final_backward != 0:
    #     loss.backward()
        #gradient track for MRL
    #enable gardient for all linear layers 
    # import pdb; pdb.set_trace()
    for name, param in model_copy.named_parameters():
        if 'classifier' in name:
            param.requires_grad = True
            # print('Gradient enabled for:', name)
            # import pdb; pdb.set_trace()
        #if its linear layer 
        if isinstance(param, torch.nn.Linear):
            param.requires_grad = True
            # print('Gradient enabled for:', name)
            # import pdb; pdb.set_trace()

    # if 1==1:
    # Matryoshka_Adapt_Loss(softmax_entropy)
    entropy_loss_2 =  Matryoshka_Adapt_Loss(softmax_entropy)
    if 1==1:
        optimizer_copy.zero_grad()
        outputs = model_copy(data)
        loss = entropy_loss_2.forward_entropy_mean(outputs)
        loss = loss.mean(0)
        # import pdb; pdb.set_trace()
        
        loss.backward()
        auto_gradient = True
        args.auto_gradient = auto_gradient
        if args.auto_gradient:
            
            # count = 0
            scores_grad = {}
            scores_param = {}
            scores_rgn = {}

            fc_layer_list = []
            for name, m in model_copy.named_modules():
            #for fc layer 
                if isinstance(m, torch.nn.Linear):
                    #take both gradients with weight and bias as keys 
                    # # scores[name] = torch.clone(m.weight.grad.clone()).detach()
                    # print('Name is : ', name)
                    # print('Shape of weight is : ', m)
                    # import pdb; pdb.set_trace()
                    #check if gradient exists 
                    if m.weight.grad is None:
                        print('Gradient does not exist for:', name)
                        continue
                    # scores[name + '.weight'] = torch.clone(m.weight.grad.clone()).detach()
                    #get gard_norm and param_norm 
                    scores_grad[name + '.weight'] = torch.clone(m.weight.grad.clone()).detach()
                    grad = torch.norm(m.weight.grad.clone().detach()).item()
                    #param norm 
                    scores_param[name + '.weight'] = torch.clone(m.weight.clone()).detach()
                    param = torch.norm(m.weight.clone().detach()).item()
                    # import pdb; pdb.set_trace()
                    # scores_rgntorch.norm(m.weight.grad.clone()) / [ torch.norm(m.weight.clone()) + 1e-8]
                    # scoes_rgn[name + '.weight'] = torch.norm(m.weight.grad.clone()) / [ torch.norm(m.weight.clone()) + 1e-8]
                    scores_rgn[name + '.weight'] = grad / (param + 1e-8)
                    

                    # scores[name + '.bias'] = torch.clone(m.bias.grad.clone()).detach()
                    fc_layer_list.append(name)
                    # print('Gradient extracted for:', name)


            # all_scores = torch.cat([torch.flatten(v) for v in scores.values()])
            # grad_flow = torch.sum(all_scores)
            # nmd = MRL_auto_gradients_select(scores)

            # all_rgns = torch.cat([torch.flatten(v) for v in scores_rgn.values()])
            all_rgns = torch.tensor([v for v in scores_rgn.values()])
            # rgn_flow = torch.sum(all_rgns)
            nmd = MRL_auto_gradients_select_rgn(scores_rgn)
            # import pdb; pdb.set_trace()
            # Start comment 
            print('Updated NMD from the previous:', args.nmd , 'to:', nmd)
            # End comment
            # wandb.log('nmd updated to:', nmd)
            # args.nmd = nmd
            # dimension = nmd
            # nmd_global = nmd
            #call configure model to update the gradients
            #turn on gradient only for the selected fc layer based on nmd 
            for i, (name, p) in enumerate(model.named_parameters()):
                fc_name = 'classifier_' + str(nmd)
                # if 'fc' in name and i != nmd:
                #     import pdb; pdb.set_trace()
                if 'classifier' in name and fc_name not in name:
                    p.requires_grad = False
                    # print('Gradient disabled for:', name)
                # #enable grad for nmd 
                if 'classifier' in name and fc_name in name:
                    p.requires_grad = True
                    # print('Gradient enabled for:', name)
                #enable grad for all classifiers
                # if 'classifier' in name:
                #     p.requires_grad = True
                    
            # count = 2
            # executed = True
            


            
            # optimizer_copy.step()
            # optimizer_copy.zero_grad()

            return model, nmd


def MRL_merge(model, args):
    # print('The layer which has been updated with gradient:', args.nmd)

    #store weights of the updated layer
    # model_copy = copy.deepcopy(model)

    grad_scaling = args.grad_scaling
    #based on nmd iterate through the layers and store the weights 
    d1 = {}
    d2 = {}
    for i, (name, p) in enumerate(model.named_parameters()):
        fc_name = 'classifier_' + str(args.nmd)
        # if 'fc' in name and i != nmd:
        #     import pdb; pdb.set_trace()
        if 'classifier' in name and fc_name in name:
            #store the weights 

            # print('Layer name:', name)
            # print('Layer weights:', p)
            # import pdb; pdb.set_trace()
            d1[name] = p.clone().detach()
        # import pdb; pdb.set_trace()
        elif 'classifier' in name and fc_name not in name:
            #save the weights of the other layers
            d2[name] = p.clone().detach()

    #calculate norm of d1
    # import pdb; pdb.set_trace()
    # d1_norm_weight = torch.norm(d1['classifier_' + str(args.nmd) + '.weight'])
    # d1_norm_bias = torch.norm(d1['classifier_' + str(args.nmd) + '.bias'])

    #iterate over d1 and calculate the norm 
    for k, v in d1.items():
        if 'weight' in k:
            d1_norm_weight = torch.norm(v)
        if 'bias' in k:
            d1_norm_bias = torch.norm(v)

    #scaling factor
    # d1_norm_weight = d1_norm_weight * 0.1
    # d1_norm_bias = d1_norm_bias * 0.1
    #defaults above 
    d1_norm_weight = (d1_norm_weight * grad_scaling)
    d1_norm_bias = (d1_norm_bias * grad_scaling)
    


    #now add above to other layers of classifier only 
    for i, (name, p) in enumerate(model.named_parameters()):
        fc_name = 'classifier_' + str(args.nmd)
        # if 'fc' in name and i != nmd:
        #     import pdb; pdb.set_trace()
        if 'classifier' in name and fc_name not in name:
            #add the norm of d1 to other layers
            if 'weight' in name:
                # print('------------------------------')
                # print('Previous norm of weight:', torch.norm(p.data), 'layer name:', name)
                # p.data = p.data.clone() + d1_norm_weight
                p.data += d1_norm_weight
                # print('After adding norm of weight:', torch.norm(p.data), 'layer name:', name)
                # print('------------------------------')
                
            if 'bias' in name:
                # p.data = p.data.clone() + d1_norm_bias
                p.data += d1_norm_bias
    
            

    # #print key values of d1 
    # for k, v in d1.items():
    #     print('Key:', k)
    #     print('Shape:', v.shape)
    
    # print('The number of layers in d1:', len(d1))
    # #print d2 
    # for k, v in d2.items():
    #     print('Key:', k)
    #     print('Shape:', v.shape)

    # import pdb; pdb.set_trace()
    # print('*********************Model merging completed*********************')

    return model



#Extra functions below 
import torch
import torch.nn.functional as F

def cosine_similarity(tensor1, tensor2):
    """Compute the cosine similarity between two tensors."""
    # Flatten both tensors using .reshape(-1)
    tensor1_flat = tensor1.reshape(-1)
    tensor2_flat = tensor2.reshape(-1)
    
    # Compute cosine similarity
    cos_sim = torch.dot(tensor1_flat, tensor2_flat) / (torch.norm(tensor1_flat) * torch.norm(tensor2_flat))

    #use torch cosine similarity
    cos_sim_t = F.cosine_similarity(tensor1_flat, tensor2_flat, dim=0)
    #use unsqueeze 
    # cos_sim_t = F.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0), dim=1)
    #get one value of cosine similarity
    # cos_sim = cos_sim_t.item()
    # import pdb; pdb.set_trace()
    return cos_sim


def resize_tensor(tensor, target_shape):
    """Resize (pad or truncate) a tensor to match the target shape."""
    tensor_shape = tensor.shape
    
    # If the tensor needs to be padded (if it's smaller than the target shape)
    if tensor_shape[1] < target_shape[1]:
        padding = (0, target_shape[1] - tensor_shape[1])  # Padding only on the last dimension
        tensor_resized = F.pad(tensor, padding)
    
    # If the tensor needs to be truncated (if it's larger than the target shape)
    elif tensor_shape[1] > target_shape[1]:
        tensor_resized = tensor[:, :target_shape[1]]  # Truncate to the target shape

        # import pdb; pdb.set_trace()

    
    else:
        tensor_resized = tensor  # No resizing needed
    
    return tensor_resized



def resize_tensor_position(tensor, target_shape):
    """Resize (pad or truncate) a tensor to match the target shape."""
    tensor_shape = tensor.shape
    
    # # If the tensor needs to be padded (if it's smaller than the target shape)
    # if tensor_shape[1] < target_shape[1]:
    #     # padding = (0, target_shape[1] - tensor_shape[1])  # Padding only on the last dimension
    #     # tensor_resized = F.pad(tensor, padding)
    #     padding = (1, target_shape[1] - tensor_shape[1])  # Padding only on the last dimension
    #     tensor_resized = F.pad(tensor, padding, mode='symmetric')

    if tensor_shape[1] < target_shape[1]:
            # Padding only at the end of the last dimension
            # padding = (0, target_shape[1] - tensor_shape[1])  # (left_pad, right_pad)
            # tensor_resized = F.pad(tensor, padding)  # Use 'symmetric' or other modes if needed
            #pad zero to the tensor boundaries with zero pad torch 
            # padding = (0, target_shape[1] - tensor_shape[1])
            padding = (0, target_shape[1] - tensor_shape[1])  # Padding only on the last dimension
            tensor_resized = F.pad(tensor, padding, mode='constant', value=0)

    
    # If the tensor needs to be truncated (if it's larger than the target shape)
    elif tensor_shape[1] > target_shape[1]:
        tensor_resized = tensor[:, :target_shape[1]]  # Truncate to the target shape
        #do symmetric padding 
        # padding = (0, target_shape[1] - tensor_shape[1])  # Padding only on the last dimension

        # import pdb; pdb.set_trace()

    
    else:
        tensor_resized = tensor  # No resizing needed
    
    return tensor_resized

def MRL_merge_cosine(model, args):
    """
    Merge model layers using cosine similarity between the updated layer and other layers.
    Handles dimension mismatch by padding or truncating the tensors.
    """
    # Store weights of the updated layer
    d1 = {}  # Updated layer (classifier_nmd)
    d2 = {}  # Other classifier layers

    grad_scaling = args.grad_scaling

    # Iterate through the model parameters and store relevant layers
    for name, p in model.named_parameters():
        fc_name = 'classifier_' + str(args.nmd)
        
        if 'classifier' in name and fc_name in name:
            # Store the updated layer's weights and biases in d1
            d1[name] = p.clone().detach()
        elif 'classifier' in name and fc_name not in name:
            # Store other layers' weights and biases in d2
            d2[name] = p.clone().detach()

    # Initialize variables for cosine similarity of weight and bias
    d1_weight = None
    d1_bias = None

    # Get the weights and biases of the updated layer (d1)
    for k, v in d1.items():
        if 'weight' in k:
            d1_weight = v
        if 'bias' in k:
            d1_bias = v

    # Now, iterate over the other classifier layers and add cosine similarity to them
    for name, p in model.named_parameters():
        fc_name = 'classifier_' + str(args.nmd)

        if 'classifier' in name and fc_name not in name:
            # For weights
            if 'weight' in name:
                # Resize the updated layer's weight to match the current layer's size
                d1_weight_resized = resize_tensor(d1_weight, p.shape)

                # Compute cosine similarity between d1's resized weight and this layer's weight
                cos_sim_weight = cosine_similarity(d1_weight_resized, p)

                # Scale and add the cosine similarity to the other layers
                # p.data += cos_sim_weight * d1_weight_resized * 0.9  # Scaling factor 0.1
                p.data += cos_sim_weight * d1_weight_resized * grad_scaling  # Scaling factor 0.1
                # import pdb; pdb.set_trace()
                # p.data += (cos_sim_weight * grad_scaling) * d1_weight_resized  # Scaling factor 0.1
                # p.data += (d1_weight_resized * d1_weight_resized) *  cos_sim_weight # Scaling factor 0.1

            # For biases
            if 'bias' in name:
                # Biases are 1D, so no need to resize (their shapes should match)
                if d1_bias.shape != p.shape:
                    raise ValueError(f"Bias sizes do not match: {d1_bias.shape} vs {p.shape}")

                # Compute cosine similarity between d1's bias and this layer's bias
                cos_sim_bias = cosine_similarity(d1_bias, p)

                # Scale and add the cosine similarity to the other layers
                # p.data += cos_sim_bias * d1_bias * 0.9  # Scaling factor 0.1
                p.data += cos_sim_bias * d1_bias * grad_scaling  # Scaling factor 0.1

                # p.data += cos_sim_bias * d1_bias * 0.1  # Scaling factor 0.1
                # p.data += (cos_sim_bias * grad_scaling) * d1_bias  # Scaling factor 0.1

    # Print completion message

    # print('*********************Model merging completed using Cosine Similarity with dimension handling*********************')

    return model



# Merge by averaging the weights of the updated layer with other layers
def MRL_merge_mean(model, args):

    # Store weights of the updated layer
    d1 = {}  # Updated layer (classifier_nmd)
    d2 = {}  # Other classifier layers

    grad_scaling = args.grad_scaling

    # Iterate through the model parameters and store relevant layers
    for name, p in model.named_parameters():
        fc_name = 'classifier_' + str(args.nmd)
        
        if 'classifier' in name and fc_name in name:
            # Store the updated layer's weights and biases in d1
            d1[name] = p.clone().detach()
        elif 'classifier' in name and fc_name not in name:
            # Store other layers' weights and biases in d2
            d2[name] = p.clone().detach()

    # Initialize variables for the updated layer's weights and biases
    d1_weight = None
    d1_bias = None

    # Get the weights and biases of the updated layer (d1)
    for k, v in d1.items():
        if 'weight' in k:
            d1_weight = v
        if 'bias' in k:
            d1_bias = v

    # Now, iterate over the other classifier layers and add the updated layer's weights to them
    for name, p in model.named_parameters():
        fc_name = 'classifier_' + str(args.nmd)

        if 'classifier' in name and fc_name not in name:
            # For weights
            if 'weight' in name:
                # Resize the updated layer's weight to match the current layer's size
                d1_weight_resized = resize_tensor(d1_weight, p.shape)

                # Scale and add the updated layer's weight to the other layers
                # p.data += d1_weight_resized * 0.1  # Scaling factor 0.1
                # p.data += d1_weight_resized * grad_scaling  # Scaling factor 0.1
                #perform avergae of the weights
                # Compute cosine similarity between d1's resized weight and this layer's weight
                cos_sim_weight = cosine_similarity(d1_weight_resized, p)
                # p.data = (p.data + d1_weight_resized) / 2 * cos_sim_weight
                # p.data += cos_sim_bias * d1_bias * grad_scaling  # Scaling factor 0.1 reference
                d1_weight_resized_scaled =  cos_sim_weight * d1_weight_resized *  grad_scaling
                # p.data = (p.data + d1_weight_resized_scaled) / 2
                #use nn.parameter 
                p.data = nn.Parameter((p.data + d1_weight_resized_scaled) / 2)
                


            # For biases
            if 'bias' in name:
                # Biases are 1D, so no need to resize (their shapes should match)
                if d1_bias.shape != p.shape:
                    raise ValueError(f"Bias sizes do not match: {d1_bias.shape} vs {p.shape}")

                # Scale and add the updated layer's bias to the other layers
                # p.data += d1_bias * 0.1  # Scaling factor 0.1
                # p.data += d1_bias * grad_scaling  # Scaling factor 0.1
                # d1_bias_resized = resize_tensor(d1_bias, p.shape)
                # import pdb ; pdb.set_trace()
                # d1_bias_resized = resize_tensor(d1_bias, p.shape)
                d1_bias_resized = d1_bias
                cos_sim_bias = cosine_similarity(d1_bias_resized, p)
                d1_bias_scaled = cos_sim_bias * d1_bias_resized * grad_scaling

                p.data = nn.Parameter((p.data + d1_bias_scaled) / 2)

    # Print completion message
    # print('*********************Model merging completed using Mean with dimension handling*********************')

    return model




    # Merge by averaging the weights of the updated layer with other layers
def MRL_merge_mean_position(model, args):

    # Store weights of the updated layer
    d1 = {}  # Updated layer (classifier_nmd)
    d2 = {}  # Other classifier layers

    grad_scaling = args.grad_scaling

    # Iterate through the model parameters and store relevant layers
    for name, p in model.named_parameters():
        fc_name = 'classifier_' + str(args.nmd)
        
        if 'classifier' in name and fc_name in name:
            # Store the updated layer's weights and biases in d1
            d1[name] = p.clone().detach()
        elif 'classifier' in name and fc_name not in name:
            # Store other layers' weights and biases in d2
            d2[name] = p.clone().detach()

    # Initialize variables for the updated layer's weights and biases
    d1_weight = None
    d1_bias = None

    # Get the weights and biases of the updated layer (d1)
    for k, v in d1.items():
        if 'weight' in k:
            d1_weight = v
        if 'bias' in k:
            d1_bias = v

    # Now, iterate over the other classifier layers and add the updated layer's weights to them
    for name, p in model.named_parameters():
        fc_name = 'classifier_' + str(args.nmd)

        if 'classifier' in name and fc_name not in name:
            # For weights
            if 'weight' in name:
                # Resize the updated layer's weight to match the current layer's size
                d1_weight_resized = resize_tensor_position(d1_weight, p.shape)

                # Scale and add the updated layer's weight to the other layers
                # p.data += d1_weight_resized * 0.1  # Scaling factor 0.1
                # p.data += d1_weight_resized * grad_scaling  # Scaling factor 0.1
                #perform avergae of the weights
                # Compute cosine similarity between d1's resized weight and this layer's weight
                cos_sim_weight = cosine_similarity(d1_weight_resized, p)
                # p.data = (p.data + d1_weight_resized) / 2 * cos_sim_weight
                # p.data += cos_sim_bias * d1_bias * grad_scaling  # Scaling factor 0.1 reference
                # import pdb; pdb.set_trace()
                d1_weight_resized_scaled =  cos_sim_weight * d1_weight_resized *  grad_scaling
                # p.data = (p.data + d1_weight_resized_scaled) / 2
                #use nn.parameter 
                p.data = nn.Parameter((p.data + d1_weight_resized_scaled) / 2)
                


            # For biases
            if 'bias' in name:
                # Biases are 1D, so no need to resize (their shapes should match)
                if d1_bias.shape != p.shape:
                    raise ValueError(f"Bias sizes do not match: {d1_bias.shape} vs {p.shape}")

                # Scale and add the updated layer's bias to the other layers
                # p.data += d1_bias * 0.1  # Scaling factor 0.1
                # p.data += d1_bias * grad_scaling  # Scaling factor 0.1
                # d1_bias_resized = resize_tensor(d1_bias, p.shape)
                # import pdb ; pdb.set_trace()
                # d1_bias_resized = resize_tensor(d1_bias, p.shape)
                d1_bias_resized = d1_bias
                cos_sim_bias = cosine_similarity(d1_bias_resized, p)
                d1_bias_scaled = cos_sim_bias * d1_bias_resized * grad_scaling

                p.data = nn.Parameter((p.data + d1_bias_scaled) / 2)

    # Print completion message
    # print('*********************Model merging completed using Mean with dimension handling*********************')

    return model




# MMerge code above 


class Matryoshka_Adapt_Loss(nn.Module):
    def __init__(self, loss_fn, relative_importance: List[float]=None, **kwargs):
        super(Matryoshka_Adapt_Loss, self).__init__()
        # self.criterion = nn.CrossEntropyLoss(**kwargs)
        self.criterion = loss_fn
        # relative importance shape: [G]
        self.relative_importance = relative_importance

    def forward(self, output, target=None):
        # output shape: [G granularities, N batch size, C number of classes]
        # target shape: [N batch size]
        
        # Calculate losses for each output and stack them. This is still O(N)
        
        # losses = torch.stack([self.criterion(output_i, target) for output_i in output])
        #target doesnt exist 
        losses = torch.stack([self.criterion(output_i) for output_i in output])

        # Set relative_importance to 1 if not specified
        rel_importance = torch.ones_like(losses) if self.relative_importance is None else torch.tensor(self.relative_importance)
        
        # Apply relative importance weights
        weighted_losses = rel_importance * losses
        return weighted_losses
    
    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_entropy_mean(self, output, target=None):
        # output shape: [G granularities, N batch size, C number of classes]
        # target shape: [N batch size]
        
        # Calculate losses for each output and stack them. This is still O(N)
        
        # losses = torch.stack([self.criterion(output_i, target) for output_i in output])
        #target doesnt exist 
        losses = torch.stack([self.criterion(output_i) for output_i in output])

        # Set relative_importance to 1 if not specified
        rel_importance = torch.ones_like(losses) if self.relative_importance is None else torch.tensor(self.relative_importance)
        
        # Apply relative importance weights
        weighted_losses = rel_importance * losses
        return weighted_losses.mean(0)
    
# def MRL_auto_gradients_extract(d1):
#     weights = []

#     # detatch first
#     d1 = {k: v.detach() for k, v in d1.items()}

#     all_weights = []
#     all_bias = []

#     for k, v in d1.items():
#         # check if bn and weight is present
#         #check if downsample is not present
#         if not '7' in k:
#             if 'downsample' not in k:
#                 if 'weight' in k:
#                     all_weights.append(v)
#                     # print('Key and value are: ', k, v)
#                     # print('Key is : ', k)
#                 if 'bias' in k:
#                     all_bias.append(v)
#                     # print('Key is : ', k)
#                     # print('Key and value are: ', k, v)

    

#     # import pdb; pdb.set_trace()
            


#     all_weight_gradients = all_weights
#     target_size = 512

#     for i, tensor in enumerate(all_weight_gradients):
#         current_size = tensor.shape[-1]
#         pad_size = target_size - current_size

#         if pad_size > 0:
#             all_weight_gradients[i] = torch.nn.functional.pad(tensor, (0, pad_size), 'constant', 0).to(device)

#     all_bias_gradients = all_bias

#     for i, tensor in enumerate(all_bias_gradients):
#         current_size = tensor.shape[-1]
#         pad_size = target_size - current_size

#         if pad_size > 0:
#             all_bias_gradients[i] = torch.nn.functional.pad(tensor, (0, pad_size), 'constant', 0).to(device)

#     weights = torch.stack(all_weight_gradients, 0).to(device)
#     bias = torch.stack(all_bias_gradients, 0).to(device)
#     # all_weight_gradients = torch.stack(all_weights).to(device)
    
#     # all_bias_gradients = torch.stack(all_bias).to(device)

    
#     # import pdb; pdb.set_trace()



#     # return weights, bias
#     # return all_weight_gradients, all_bias_gradients
#     return weights, bias