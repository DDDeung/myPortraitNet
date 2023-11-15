from datasets import *
from easydict import EasyDict as edict

exp_args = edict()

exp_args.istrain = True
exp_args.task = 'seg'
exp_args.datasetlist =['EG1800']
exp_args.data_root = '/Users/koala/Desktop/cuhksz2023sem1/cv/portraitNet/myPortraitNet/datasets/'
exp_args.file_root = '/Users/koala/Desktop/cuhksz2023sem1/cv/portraitNet/myPortraitNet/datasets/EG1800'

exp_args.input_height = 352
exp_args.input_width = 352

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



for index, (img_ori,img, edge, mask) in enumerate(dataLoader_train):
    if index == 0:
        print(img.shape)
        print(mask.shape)
        print(edge.shape)
        break