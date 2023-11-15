import random
import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torchvision import transforms

image_mean = torch.Tensor([123.68, 116.78, 103.94]) # RGB
image_val = 0.017

def get_probability(prob):
    return np.random.uniform(0, 1) < prob


def padding(img_def, img_tex, mask_ori, size=224, padding_color=128):
    height, width = img_def.size

    img_d_resize = np.zeros((max(height, width), max(height, width), 3)) + padding_color
    img_t_resize = np.zeros((max(height, width), max(height, width), 3)) + padding_color
    mask = np.zeros((max(height, width), max(height, width)))

    if (height > width):
        padding = int((height - width) / 2)
        img_d_resize[:, padding:padding + width, :] = img_def
        img_d_resize[:, padding:padding + width, :] = img_tex
        mask[:, padding:padding + width] = mask_ori
    else:
        padding = int((width - height) / 2)
        img_d_resize[:, padding:padding + width, :] = img_def
        img_d_resize[:, padding:padding + width, :] = img_tex
        mask[padding:padding + height, :] = mask_ori

    img_d_resize = np.uint8(img_d_resize)
    img_t_resize = np.uint8(img_t_resize)
    mask = np.uint8(mask)

    img_d_resize = cv2.resize(img_d_resize, (size, size), interpolation=cv2.INTER_CUBIC)
    img_t_resize = cv2.resize(img_t_resize, (size, size), interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_CUBIC)

    return Image.fromarray(cv2.cvtColor(img_d_resize, cv2.COLOR_BGR2RGB)),Image.fromarray(cv2.cvtColor(img_t_resize, cv2.COLOR_BGR2RGB)),Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))

"""
parameters from paper 4.2
"""
def process_image(img, mask, prob=0.5, rotation_range=[-45, 45], resize_scale_range=[0.5, 1.5],
                  translation_range=[-0.25, 0.25], gaussian_sigma=10, blur_kernel_sizes=[3, 5],
                              color_change_range=[0.4, 1.7], brightness_change_range=[0.4, 1.7],
                              contrast_change_range=[0.6, 1.5], sharpness_change_range=[0.8, 1.3]):
    W, H = img.size
    # original_img = img
    """
    deformation augmentation
    """
    # random horizontal flip
    if get_probability(prob):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    # # random translation
    if get_probability(prob):
        translation_amount = np.random.uniform(translation_range[0], translation_range[1], 2)

        # x1, y1, x2, y2 = 0, 0, W, H
        # x1 = int(max(x1 + W * translation_amount[0], 0))
        # y1 = int(max(y1 + H * translation_amount[1], 0))
        # x2 = int(min(x2 + W * translation_amount[0], W))
        # y2 = int(min(y2 + H * translation_amount[1], H))
        # temp_img = Image.new("RGB", (W, H))
        # temp_img.paste(img.crop([x1, y1, x2, y2]), [x1, y1])
        # img = temp_img
        # temp_img = Image.new("1", (W, H))
        # temp_img.paste(mask.crop([x1, y1, x2, y2]), [x1, y1])
        # mask = temp_img

    # random rotation
    if get_probability(prob):
        rotated_angle = np.random.uniform(rotation_range[0], rotation_range[1])
        img = img.rotate(rotated_angle)
        mask = mask.rotate(rotated_angle)

    # random resize
    if get_probability(prob):
        resize_scale = np.random.uniform(resize_scale_range[0], resize_scale_range[1])
        W, H = int(W * resize_scale), int(H * resize_scale)
        img = img.resize((W, H))
        mask = Image.fromarray(np.array(mask.resize((W, H))))

    img_deformation = img
    """
    texture augmentation
    """
    # random Gaussian noise
    if get_probability(prob):
        noise = np.random.normal(0, gaussian_sigma, (H, W, 3))
        img = Image.fromarray(np.uint8(np.clip(np.array(img) + noise, 0, 255)))

    # random blur (from official code)
    if get_probability(prob):
        select = np.random.uniform(0, 1)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        if select < 0.3:
            kernel_size = random.choice(blur_kernel_sizes)
            image = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        elif select < 0.6:
            kernel_size = random.choice(blur_kernel_sizes)
            image = cv2.medianBlur(img, kernel_size)
        else:
            kernel_size = random.choice(blur_kernel_sizes)
            image = cv2.blur(img, (kernel_size, kernel_size))
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # random color & brightness & contrast & sharpness change
    if get_probability(prob):
        color_factor = np.random.uniform(color_change_range[0], color_change_range[1])
        img = ImageEnhance.Color(img).enhance(color_factor)
        brightness_factor = np.random.uniform(brightness_change_range[0], brightness_change_range[1])
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)
        contrast_factor = np.random.uniform(contrast_change_range[0], contrast_change_range[1])
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)
        sharpness_factor = np.random.uniform(sharpness_change_range[0], sharpness_change_range[1])
        img = ImageEnhance.Sharpness(img).enhance(sharpness_factor)

    img_texture = img
    # resize to 224
    if W >= H:
        H = int(H * 224 / W)
        W = 224
        img_d_resize, img_t_resize, mask_resize = transforms.Resize([H, W])(img_deformation), transforms.Resize([H, W])(img_texture),transforms.Resize([H, W])(mask)
        img_d_resize = transforms.Pad(padding=[0, (224 - H) // 2, 0, (224 - H) // 2 + (224 - H) % 2],
                             padding_mode="constant")(img_d_resize)
        img_t_resize = transforms.Pad(padding=[0, (224 - H) // 2, 0, (224 - H) // 2 + (224 - H) % 2],
                                      padding_mode="constant")(img_t_resize)
        mask_resize = transforms.Pad(padding=[0, (224 - H) // 2, 0, (224 - H) // 2 + (224 - H) % 2],
                              padding_mode="constant")(mask_resize)
    else:
        W = int(W * 224 / H)
        H = 224
        img_d_resize, img_t_resize, mask_resize = transforms.Resize([H, W])(img_deformation), transforms.Resize([H, W])(img_texture),transforms.Resize([H, W])(mask)
        img_d_resize = transforms.Pad(padding=[(224 - W) // 2, 0, (224 - W) // 2 + (224 - W) % 2, 0],
                             padding_mode="constant")(img_d_resize)
        img_t_resize = transforms.Pad(padding=[(224 - W) // 2, 0, (224 - W) // 2 + (224 - W) % 2, 0],
                                      padding_mode="constant")(img_t_resize)
        mask_resize = transforms.Pad(padding=[(224 - W) // 2, 0, (224 - W) // 2 + (224 - W) % 2, 0],
                              padding_mode="constant")(mask_resize)

    # get edge
    edge = np.zeros((224, 224), np.uint8)
    ret, binary = cv2.threshold(np.uint8(mask_resize) * 255, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # RETR_EXTERNAL
    cv2.drawContours(edge, contours, -1, 1, 4)



    return torch.Tensor(np.array(img_d_resize)), torch.Tensor(np.array(img_t_resize)),torch.Tensor(np.array(mask_resize)),  torch.Tensor(edge)




def transform_train(img, mask):
    img_deformation, img_texture,mask, edge = process_image(img, mask)
    #normalization equation is (image − mean) × val
    img_deformation = (img_deformation - image_mean).permute(2, 0, 1) * image_val
    img_texture = (img_texture - image_mean).permute(2, 0, 1) * image_val
    # img_deformation = img_deformation.permute(2, 0, 1)
    # img_texture = img_texture.permute(2, 0, 1)
    return img_deformation, img_texture, mask, edge


def transform_test(img, mask):
    # resize to 224
    W, H = img.size
    if W >= H:
        H = int(H * 224 / W)
        W = 224
        img, mask = transforms.Resize([H, W])(img), transforms.Resize([H, W])(mask)
        img = transforms.Pad(padding=[0, (224 - H) // 2, 0, (224 - H) // 2 + (224 - H) % 2],
                             padding_mode="constant")(img)
        mask = transforms.Pad(padding=[0, (224 - H) // 2, 0, (224 - H) // 2 + (224 - H) % 2],
                              padding_mode="constant")(mask)
    else:
        W = int(W * 224 / H)
        H = 224
        img, mask = transforms.Resize([H, W])(img), transforms.Resize([H, W])(mask)
        img = transforms.Pad(padding=[(224 - W) // 2, 0, (224 - W) // 2 + (224 - W) % 2, 0],
                             padding_mode="constant")(img)
        mask = transforms.Pad(padding=[(224 - W) // 2, 0, (224 - W) // 2 + (224 - W) % 2, 0],
                              padding_mode="constant")(mask)

    # get edge
    edge = np.zeros((224, 224), np.uint8)
    ret, binary = cv2.threshold(np.uint8(mask) * 255, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # RETR_EXTERNAL
    cv2.drawContours(edge, contours, -1, 1, 4)

    #normalize
    img = (torch.Tensor(np.array(img)) - image_mean).permute(2, 0, 1) * image_val

    return img, torch.Tensor(np.array(mask)), torch.Tensor(edge)
#

from matplotlib import pyplot as plt

img_item = cv2.imread("./datasets/EG1800/Images/00002.png")
mask_item =cv2.imread("./datasets/EG1800/Labels/00002.png")

img_item = Image.fromarray(cv2.cvtColor(img_item, cv2.COLOR_BGR2RGB))
mask_item = Image.fromarray(cv2.cvtColor(mask_item, cv2.COLOR_BGR2GRAY))
#
img_deformation, img_texture, mask, edge = transform_train(img_item, mask_item)
# img_deformation_ori, img_texture_ori, mask_ori, edge_ori = process_image(img_item, mask_item)
img_test, mask_test, edge_test = transform_test(img_item, mask_item)

# show mask/edge
array1=mask.numpy()#将tensor数据转为numpy数据
# mat=cv2.cvtColor(array1,cv2.COLOR_BGR2RGB)
cv2.imshow("img",array1)
cv2.waitKey()

#show img
array2=img_deformation.numpy()#将tensor数据转为numpy数据
maxValue=array2.max()
array2=array2*255/maxValue#normalize，将图像数据扩展到[0,255]
mat2=np.uint8(array2)#float32-->uint8
print('mat_shape:',mat2.shape)#mat_shape: (3, 982, 814)
mat2=mat2.transpose(1,2,0)#mat_shape: (982, 814，3)
mat2=cv2.cvtColor(mat2,cv2.COLOR_BGR2RGB)
cv2.imshow("img",mat2)
cv2.waitKey()

