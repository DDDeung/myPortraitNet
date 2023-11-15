import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import eg1800dataset
from train import *
from test import *
from portraitNet import ProtraitNet
import torchvision.models.mobilenetv2
from datasets import *
from easydict import EasyDict as edict
from data.datasets import *

# dataset_root_path = "datasets/EG1800"
#
# #Get dataset  - use self-defined dataset
# train_dataset = eg1800dataset(dataset_root_path, train_or_test=True,transform_train=True)
# test_dataset = eg1800dataset(dataset_root_path, train_or_test=False,transform_train = False)
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------use official dataset definition--------------
exp_args = edict()
exp_args.istrain = True
exp_args.task = 'seg'
exp_args.datasetlist =['EG1800']
exp_args.data_root = '/Users/koala/Desktop/cuhksz2023sem1/cv/portraitNet/myPortraitNet/datasets/'
exp_args.file_root = '/Users/koala/Desktop/cuhksz2023sem1/cv/portraitNet/myPortraitNet/datasets/EG1800'

exp_args.input_height = 224
exp_args.input_width = 224


exp_args.prior_prob = 0.5 # the probability to set empty prior channel

exp_args.edgeRatio = 0.1 # the weight of boundary auxiliary loss
# exp_args.stability = True
exp_args.temperature = 1 # the temperature in consistency constraint loss, default=1
exp_args.alpha = 2 # the weight of consistency constraint loss, default=2
############################
exp_args.padding_color=128 # input normalization parameters
exp_args.img_scale = 1
exp_args.img_mean = [103.94, 116.78, 123.68] # BGR order, image mean
exp_args.img_val = [0.017, 0.017, 0.017] # BGR order, image val, default=[1/0.017, 1/0.017, 1/0.017]
##########################
exp_args.init = False # whether to use pretrained model to init portraitnet
exp_args.resume = False # whether to continue training

# set training dataset
exp_args.learning_rate = 1e-3
exp_args.momentum = 0.9
exp_args.weight_decay = 5e-4
exp_args.batch_size = 1
#######################下面没什么用
exp_args.addEdge = True
exp_args.stability = True
exp_args.use_kl = True
exp_args.useUpsample = False
exp_args.useDeconvGroup = False
exp_args.video = False
exp_args.istrain = True

dataset_train = Human(exp_args)
dataLoader_train = torch.utils.data.DataLoader(dataset_train, batch_size=64,
                                               shuffle=True)

# set testing dataset
exp_args.istrain = False
dataset_test = Human(exp_args)
dataLoader_test = torch.utils.data.DataLoader(dataset_test, batch_size=64,
                                              shuffle=False)
#---------------------------
#Get model
portrait_net = ProtraitNet(n_class=2)

#use some weights - pretrained MobileNetV2
pretrained_mobilenet_dict = torch.load("./mobilenet_v2-b0353104.pth")
pretrained_mobilenet_dict_keys = list(pretrained_mobilenet_dict.keys())
portrait_net_dict = portrait_net.state_dict()
portrait_net_dict_keys = list( portrait_net.state_dict().keys())
print ("pretrained mobilenetv2 keys: ", len(pretrained_mobilenet_dict))
print ("portraitnet keys: ", len(portrait_net_dict_keys))
weights_load = {}
for k in range(len(pretrained_mobilenet_dict_keys)):
    if pretrained_mobilenet_dict[pretrained_mobilenet_dict_keys[k]].shape == \
    portrait_net_dict[portrait_net_dict_keys[k]].shape:
        weights_load[portrait_net_dict_keys[k]] = pretrained_mobilenet_dict[pretrained_mobilenet_dict_keys[k]]
        print ('init model', portrait_net_dict_keys[k],
               'from pretrained', portrait_net_dict_keys[k])
    else:
        break
print ("init len is:", len(weights_load))
portrait_net_dict.update(weights_load)
portrait_net.load_state_dict(portrait_net_dict)
print ("load model init finish...")
portrait_net.to(device)

#Get optimizer
optimizer = torch.optim.Adam(portrait_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

#Get Lr scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,lr_decay_epoch,lr_decay)

current_epoch = 0
min_val_loss = 10000
train_losses = []
test_losses = []
test_miou = 0

#Get optimizer
for g in optimizer.param_groups:
    g['lr']=1e-3

best_val_iou = 0

for epoch in range(current_epoch, epoches):
    print("Epoch : {}/{} , learning rate : {}".format(epoch,epoches,optimizer.param_groups[0]['lr']))
    train_loss, train_iou = train(dataLoader_train,portrait_net,optimizer,device)
     # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{epoches:03d} ] train_loss = {train_loss:.5f}, train_iou = {train_iou:.5f}")

    test_loss,test_iou = test(dataLoader_test,portrait_net,device)
    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{epoches:03d} ] test_loss = {test_loss:.5f}, test_iou = {test_iou:.5f}")

    train_losses.append(train_loss)
    test_losses.append(test_loss)


     # update logs
    if test_iou > best_val_iou:
        with open(f"./log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{epoches:03d} ] test_loss = {test_loss:.5f}, test_iou = {test_iou:.5f} -> best")
    else:
        with open(f"./log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{epoches:03d} ] test_loss = {test_loss:.5f}, test_iou = {test_iou:.5f}")

    #Save best model
    if test_iou > best_val_iou:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(portrait_net.state_dict(), f"./myPortraitNet.ckpt") # only save best to prevent output memory exceed error
        best_val_iou = test_iou

    print("Train Loss : {}, Test Loss : {} , Test IOU : {} , best Test mIOU : {}".format(train_loss,test_loss,test_iou,best_val_iou))
    lr_scheduler.step()

with open("./train_loss.txt", 'w') as train_los:
    train_los.write(str(train_losses))

with open("./train_acc.txt", 'w') as train_ac:
     train_ac.write(str(test_losses))

# model_best = ProtraitNet(n_class=2).to(device)
# model_best.load_state_dict(torch.load("./myPortraitNet.ckpt"))
# test_loss,test_iou = test(test_dataloader,portrait_net,device)
# print(f" test_loss = {test_loss:.5f}, test_iou = {test_iou:.5f}")