import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torchvision

a = os.listdir("./TCGA/imgs/train")
a.pop(0)
m = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(size=(256,256), padding=(32,32)), transforms.ToTensor()])
n = transforms.ToTensor()
try:
    os.mkdir("./TCGA/imgs/train/new/")
    os.mkdir("./TCGA/masks/train/new/")
except:
    pass
for i in a:
    img = plt.imread("./TCGA/imgs/train/" + i)
    mask = plt.imread("./TCGA/masks/train/" + i)
    img1 = np.rot90(img, k=1)
    mask1 = np.rot90(mask, k=1)
    img2 = np.rot90(img, k=2)
    mask2 = np.rot90(mask, k=1)
    img3 = np.rot90(img, k=3)
    mask3 = np.rot90(mask, k=1)
    imgs = [img, img1, img2, img3]
    masks = [mask, mask1, mask2, mask3]
    for o in range(4):
        seed = np.random.randint(2147483647)
        np.random.seed(seed)
        torch.manual_seed(seed)
        imgs.append(m(imgs[o]))
        masks.append(m(masks[o]))
        imgs[o] = n(imgs[o].copy())
        masks[o] = n(masks[o].copy())
    torchvision.utils.save_image(imgs[o], "./TCGA/imgs/train/new/" + i)
    torchvision.utils.save_image(masks[o], "./TCGA/masks/train/new/" + i)
    i = i.replace(".tif", "")
    for o in range(1, 8):
        torchvision.utils.save_image(imgs[o], "./TCGA/imgs/train/new/" + i + "_" + str(o) + ".tif")
        torchvision.utils.save_image(masks[o], "./TCGA/masks/train/new/" + i + "_" + str(o) + ".tif")
