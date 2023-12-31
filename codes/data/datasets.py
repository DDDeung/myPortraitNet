import torch
import torch.utils.data as data
import numpy as np

from data.datasets_portraitseg import PortraitSeg

class Human(data.Dataset): 
    def __init__(self, exp_args):
        assert exp_args.task in ['seg'], 'Error!, <task> should in [seg]'
        
        self.exp_args = exp_args
        self.task = exp_args.task
        self.datasetlist = exp_args.datasetlist
        self.data_root = exp_args.data_root # data_root = '/Users/koala/Desktop/cuhksz2023sem1/cv/portraitNet/myPortraitNet/datasets/'
        self.file_root = exp_args.file_root # file_root = '/home/dongx12/PortraitNet/data/select_data/'
        
        self.datasets = {}
        self.imagelist = []

        if 'EG1800' in self.datasetlist:
            ImageRoot = self.data_root + 'EG1800/Images/'
            AnnoRoot = self.data_root + 'EG1800/Labels/'
            ImgIds_Train = self.file_root + '/eg1800_train.txt'
            ImgIds_Test = self.file_root + '/eg1800_test.txt'
            exp_args.dataset = 'eg1800'
            self.datasets['eg1800'] = PortraitSeg(ImageRoot, AnnoRoot, ImgIds_Train, ImgIds_Test, self.exp_args)
        

        # image list
        for key in self.datasets.keys():
            length = len(self.datasets[key])
            for i in range(length):
                self.imagelist.append([key, i])
        
    def __getitem__(self, index):
        subset, subsetidx = self.imagelist[index]
        
        if self.task == 'seg':
            input_ori, input, output_edge, output_mask = self.datasets[subset][subsetidx]
            return input_ori.astype(np.float32), input.astype(np.float32), \
        output_edge.astype(np.int64), output_mask.astype(np.int64)
           
    def __len__(self):
        return len(self.imagelist)