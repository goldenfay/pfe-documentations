from torchvision import transforms
from torchvision.transforms import functional as F
from PIL.Image import Image
import random
import numpy as np


def simple_transform(img):

    return transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
    ]).__call__(img)


def crop_center(img, mask, size):
    width, height = img.size
    t_width, t_height = (size, size)
    x1 = int(round((width - t_width) / 2.))
    y1 = int(round((height - t_height) / 2.))
    return img.crop((x1, y1, x1 + t_width, y1 + t_height)), mask.crop((x1, y1, x1 + t_width, y1 + t_height))


def flip(img, mask):
    return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)


def scale(img, rate, dens):
    img_w, img_h = img.size
    den_w, den_h = dens.size


    init_random_rate = random.uniform(rate[0], rate[1])

    dst_img_w = int(img_w*init_random_rate)//32*32
    dst_img_h = int(img_h*init_random_rate)//32*32

    real_rate_w = dst_img_w/img_w
    real_rate_h = dst_img_h/img_h

    dst_den_w = int(den_w*init_random_rate)//32*32
    dst_den_h = int(den_h*init_random_rate)//32*32

    den = np.array(dens.resize((dst_den_w, dst_den_h),
                              Image.BILINEAR))/real_rate_w/real_rate_h
    den = Image.fromarray(den)

    return img.resize((dst_img_w, dst_img_h), Image.BILINEAR), den



def gamma_correction(img,gamma_range=[0.4,2]):
    if random.random() < 0.5:
            gamma = random.uniform(gamma_range[0],gamma_range[1])
            return  F.adjust_gamma(img, gamma)
    else: 
            return img
            
def normalize(img,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
    return transforms.Normalize(mean=mean,std=std).__call__(img)

def denormalize(img,mean,std):
    for t, m, s in zip(img,mean,std):
            t.mul_(s).add_(m)
    return img
