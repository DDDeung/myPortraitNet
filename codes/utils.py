import torch
import os
import numpy as np

def load_checkpoint(net,optimizer,model_folder,model_name,device):
    # using latest_model is higher priority than using bestmodel in the previous stages
    if model_name.split('.')[2] != "tar":
        raise ValueError("latest model file should be '.tar' file and it saves the latest checkpoint")
    model_path = os.path.join(model_folder,model_name)
    model_checkpoint = torch.load(model_path,map_location=device)
    net.load_state_dict(model_checkpoint["state_dict"])
    optimizer.load_state_dict(model_checkpoint["optimizer"])
    current_epoch = model_checkpoint["trained_num_epochs"]
    val_losses = model_checkpoint["val_losses"] # length = trained_num_epochs,loss list relative to each epoch
    val_iou = model_checkpoint["val_iou"]
    train_losses = model_checkpoint["train_losses"]
    return net,optimizer,current_epoch,train_losses,val_losses,val_iou

def save_checkpoint(state,model_folder,model_name):
    if model_name.split('.')[2] != "tar":
        raise ValueError("Check point which is saved should '.tar' file")
    torch.save(state,os.path.join(model_folder,model_name))