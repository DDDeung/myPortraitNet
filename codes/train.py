import torch
import torch.nn as nn
from tqdm import tqdm
from loss import *
from torch.autograd import Variable
from metrice import *

# use "mps" if possible
# device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"
# -----train hyperparameters-------
batch_size = 64

epoches = 2000

weight_decay = 5e-4

learning_rate = 1e-3

lr_decay = 0.95

lr_decay_epoch = epoches // 20


# -----model hyperparameters-------
# temperature in consistency constraint loss
temperature = 1
# the weight of consistency constraint loss
alpha = 2

# def adjust_learning_rate(optimizer, epoch, multiple):
#     """Sets the learning rate to the initial LR decayed by 0.95 every 20 epochs"""
#     lr = learning_rate * (lr_decay ** (epoch // 20))
#     for i, param_group in enumerate(optimizer.param_groups):
#         param_group['lr'] = lr * multiple[i]
#     pass


def train(dataloader, model, optimizer, device="cpu"):
    model.train()  # switch to train mode
    loss_Softmax = nn.CrossEntropyLoss(ignore_index=255) # mask loss
    loss_Focalloss = FocalLoss(gamma=2) # boundary loss

    losses = []
    iou = 0
    loader = tqdm(dataloader)

    # for i, (input_deformation, input_texture, mask, mask_boundary) in enumerate(loader):
    for i, (input_deformation, input_texture, mask_boundary, mask) in enumerate(loader):

        #Send to device
        input_deformation = input_deformation.to(device)
        input_texture = input_texture.to(device)
        mask = mask.to(device)
        mask_boundary = mask_boundary.to(device)

        #load input to model
        output_mask_deformation, output_edge_deformation = model(input_deformation)
        output_mask_texture, output_edge_texture = model(input_texture)

        #mask loss and edge loss
        loss_mask_texture = loss_Softmax(output_mask_texture,mask)
        loss_edge_texture = loss_Focalloss(output_edge_texture,mask_boundary)*0.1
        loss_mask_deformation = loss_Softmax(output_mask_deformation,mask)
        loss_edge_deformation = loss_Focalloss(output_edge_deformation,mask_boundary)*0.1
       
       # consistency constraint loss: KL distance
        loss_kl_mask = loss_KL(output_mask_texture, Variable(output_mask_deformation.data, requires_grad=False),
                              temperature) * alpha
        loss_kl_edge = loss_KL(output_edge_texture, Variable(output_edge_deformation.data, requires_grad=False),
                              temperature) * alpha * 0.1

        #total loss
        # loss = loss_mask_texture + loss_edge_texture + loss_mask_deformation + loss_edge_deformation + loss_kl_mask + loss_kl_edge
        # loss = loss_mask + loss_mask_ori + loss_kl_mask + loss_edge
        loss = loss_mask_texture + loss_mask_deformation + loss_kl_mask + loss_edge_texture
        
        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loader.set_postfix(loss=loss.item())
        losses.append(loss.item())
        softmax = nn.Softmax(dim=1)
        prob = softmax(output_mask_texture)[0,1,:,:]
        pred = prob.data.cpu().numpy()
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0
        iou += calcIOU(pred, mask[0].data.cpu().numpy())
    return sum(losses)/len(losses), iou / len(dataloader)
    
