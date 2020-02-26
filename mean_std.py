import os
from skimage import io
import numpy as np

imgs = os.listdir("./TCGA/imgs/train")
imgs1 = os.listdir("./TCGA/imgs/test")
images = [io.imread("./TCGA/imgs/train/" + i)/255 for i in imgs]
images += [io.imread("./TCGA/imgs/test/" + i)/255 for i in imgs1]
images = np.stack(images)
print(images.shape)
print("Mean:", np.mean(images, axis=(0,1,2)))
print("Std:", np.std(images, axis=(0,1,2)))

