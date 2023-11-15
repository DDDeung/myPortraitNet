import imp
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from loss import *
from torch.autograd import Variable
import copy
from metrice import *

# -----model hyperparameters-------
# temperature in consistency constraint loss
temperature = 1
# the weight of consistency constraint loss
alpha = 2


def test(dataloader, model , device = "cpu"):
    # switch to eval mode
    model.eval()# switch to eval mode

    loss_Focalloss = FocalLoss(gamma=2)
    loss_Softmax = nn.CrossEntropyLoss(ignore_index=255)
    losses = []

    loader = tqdm(dataloader)
    iou = 0

    with torch.no_grad():
        # for idx, (img, mask, mask_boundary) in enumerate(loader):
        for i, (input_ori, input, mask_boundary, mask) in enumerate(loader):
            # Send to device
            input_ori = input_ori.to(device)
            input = input.to(device)
            mask = mask.to(device)
            mask_boundary = mask_boundary.to(device)

            # load input to model
            output_mask_texture, output_edge_texture = model(input)
            output_mask_deformation, output_edge_deformation = model(input_ori)

            # mask loss and edge loss
            loss_mask_texture = loss_Softmax(output_mask_texture, mask)
            loss_edge_texture = loss_Focalloss(output_edge_texture, mask_boundary) * 0.1
            loss_mask_deformation = loss_Softmax(output_mask_deformation, mask)
            loss_edge_deformation = loss_Focalloss(output_edge_deformation, mask_boundary) * 0.1

            # consistency constraint loss: KL distance
            loss_kl_mask = loss_KL(output_mask_texture, Variable(output_mask_deformation.data, requires_grad=False),
                                temperature) * alpha
            loss_kl_edge = loss_KL(output_edge_texture, Variable(output_edge_deformation.data, requires_grad=False),
                                temperature) * alpha * 0.1

            #total loss
            # loss = loss_mask_texture + loss_edge_texture + loss_mask_deformation + loss_edge_deformation + loss_kl_mask + loss_kl_edge
            # loss = loss_mask + loss_mask_ori + loss_kl_mask + loss_edge
            loss = loss_mask_texture + loss_mask_deformation + loss_kl_mask + loss_edge_texture

            losses.append(loss.item())

            softmax = nn.Softmax(dim=1)
            prob = softmax(output_mask_texture)[0,1,:,:]
            pred = prob.data.cpu().numpy()
            pred[pred>0.5] = 1
            pred[pred<=0.5] = 0
            iou += calcIOU(pred, mask[0].data.cpu().numpy())

    return sum(losses)/len(losses), iou / len(loader)