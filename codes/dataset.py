import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import data_augmentation

class eg1800dataset(Dataset):
    def __init__(self, data_root_path, train_or_test, transform_train):
        self.data_root_path = data_root_path
        self.train_or_test = train_or_test
        self.transform_train = transform_train
        self.imgs = []
        if train_or_test:  # training data
            data_root_path = data_root_path + "/eg1800_train.txt"
        else:  # test data
            data_root_path =  data_root_path + "/eg1800_test.txt"

        with open(data_root_path, "r") as f:
            for line in f.readlines():
                line = line.strip('\n')
                line = './' + line
                self.imgs.append(line)
        #
        # self.raw_img_name = os.listdir(self.data_root_path)
        # self.img_filenames = self.raw_img_name[::2]
        # self.mask_filenames = self.raw_img_name[1::2]

    def __getitem__(self, index):
        # img_item = Image.open(self.data_root_path + "/Images/" + self.imgs[index]).convert("RGB")
        # mask_item = Image.open(self.data_root_path + "/Labels/" +self.imgs[index]).convert("1")
        img_item = cv2.imread(self.data_root_path + "/Images/" + self.imgs[index])
        mask_item = cv2.imread(self.data_root_path + "/Labels/" +self.imgs[index])
        mask_item[mask_item > 1] = 0
        img_item = Image.fromarray(cv2.cvtColor(img_item, cv2.COLOR_BGR2RGB))
        mask_item = Image.fromarray(cv2.cvtColor(mask_item, cv2.COLOR_BGR2GRAY))

        if self.transform_train:
            img_deformation, img_texture, mask, edge = data_augmentation.transform_train(img_item, mask_item)
            return img_deformation, img_texture, torch.Tensor(mask).long(), edge.long()
        else:
            img, mask, edge = data_augmentation.transform_test(img_item, mask_item)
            return img, torch.Tensor(mask).long(), edge.long()
        # show image test
        # img_item.show()
        # mask_item.show()
        # return img_item, mask_item
        # return original_img_item.permute(2, 0, 1), (img_item - DataAugmentation.image_mean).permute(2, 0, 1) * \
        #                                            DataAugmentation.image_val, torch.Tensor(mask_item).long(), edge

    def __len__(self):
        return len(self.imgs)


# test
dataset = eg1800dataset("datasets/EG1800", train_or_test=True,transform_train=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
for index, (img_ori, img, mask, edge) in enumerate(dataloader):
    if index == 0:
        print(img_ori.shape)
        print(img.shape)
        print(mask.shape)
        print(edge.shape)
        break
#
#
# dataset = eg1800dataset("datasets/EG1800", train_or_test=False,transform_train=False)
# dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
# for index, (img, mask, edge) in enumerate(dataloader):
#     if index == 0:
#         print(img.shape)
#         print(mask.shape)
#         print(edge.shape)
#         break