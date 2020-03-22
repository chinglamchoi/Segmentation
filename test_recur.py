import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms as transforms
import numpy as np
import os
from skimage.segmentation import find_boundaries
#from skimage.metrics import adapted_rand_error
from sklearn.metrics import adjusted_rand_score
#from loss import DiceLoss
from skimage import io
import unet_multi
import unet
import torch.nn as nn

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score


class BrainTumour(data.Dataset):

    def __init__(self, img_dir, mask_dir, lbs, transform=transforms.ToTensor(), img_transform = transforms.Normalize([0.5], [0.5])):
        self.img_dir, self.mask_dir = img_dir, mask_dir
        self.ids = os.listdir(img_dir)
        self.transform, self.img_transform = transform, img_transform
        self.lbs = lbs.tolist()                                                                       
        img, mask = [], []
        assert ((len(self.ids) == 759) or (len(self.ids) == 25360))
        for i in self.ids:
            img.append(io.imread(self.img_dir + i))
            mask.append(io.imread(self.mask_dir + i))
        self.img = np.array(img)
        self.mask = np.array(mask)
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, index):
        idx = self.ids[index]
        lb = torch.tensor([self.lbs[index]], dtype=torch.float)
        img1, mask1 = self.transform(self.img[index]), self.transform(self.mask[index])
        img1 = self.img_transform(img1)
        return (img1, mask1, lb, idx)

def DiceLoss(a,b):
    tot = 256*256
    smooth = 1.
    a = a.view(-1)
    b = b.view(-1)
    intersection = (a*b).sum()
    return 1 - ((2. * intersection + smooth) / (a.sum() + b.sum() + smooth))

"""
def PixelLoss(a,b):
    a = (a > 0.33).float()
    c = find_boundaries(a.cpu().numpy(), background=0)
    d = find_boundaries(b.cpu().numpy(), background=0)
    return np.linalg.norm(np.subtract(c,d, dtype=np.float32)) #precompute for trainset
"""
class UNet2(nn.Module):
    def __init__(self):
        super(UNet2, self).__init__()
        self.outc1 = nn.AdaptiveAvgPool2d((1,1))
        self.outc2 = nn.Linear(1,1)
    def forward(self, x):
        out = self.outc1(x)
        out = self.outc2(out.view(out.size(0), -1))
        return x, out

def run_cnn2():
    return UNet2()

def RandLoss(a,b):
    a = (a >= 0.35).float()
    a = a.reshape((256, 256))
    b = b.reshape((256,256))
    a = a.cpu().numpy().flatten()
    b = b.cpu().numpy().flatten()
    c = adjusted_rand_score(a,b)
    c = (c+1)/2
    return 1 - c


def hook_feature(module, input, output):
    feature_blobs.append(output.cpu().data.numpy())

def returnCAM(feature_conv, weight):
    _, nc, h, w = feature_conv.shape
    output_cam = []
    for i in range(_):
        cam = weight.dot(feature_conv[i].reshape((nc, h*w)))
        cam = (torch.from_numpy(cam.reshape(h, w))).reshape(1, h, w)
        output_cam.append(cam)
    return torch.stack(output_cam)

#torch.multiprocessing.freeze_support()
if __name__ == "__main__":
    px_avg = np.array([23.31432984, 21.05975591, 22.50303842])
    test_label_file = np.load("./test_labels.npy")
    testset = BrainTumour(img_dir='./TCGA/imgs/test/', mask_dir='./TCGA/masks/test/', lbs=test_label_file)
    testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)
    
    a = "cuda:0"
    device = torch.device(a if torch.cuda.is_available() else "cpu")

    net1 = unet.run_cnn()
    net2 = run_cnn2()

    pretrain = input("File path of pretrained model: ")
    checkpoint = torch.load("report/new/" + pretrain + ".pt", map_location="cuda:0")
    net1 = net1.eval()
    net2 = net2.eval()

    net = nn.Sequential(net1, net2)
    net.load_state_dict(checkpoint["net"])
    net.to(device)
    net._modules["0"]._modules.get("conv").register_forward_hook(hook_feature)
    net = net.eval()

    path_ = "./TCGA/mask_pred/" + pretrain + "/"
    try:
        os.mkdir(path_)
    except:
        pass

    tot = 0.0
    tot2 = 0.0
    tot_rand = 0.0
    with torch.no_grad():
        tp, fp, tn, fn = 0,0,0,0
        tp1, fp1, tn1, fn1 = 0,0,0,0
        for img, mask, lb, idx in testloader:
            weights = np.squeeze(list(net.parameters())[-2].cpu().data.numpy())
            feature_blobs = []
            img, mask = img.to(device), mask.to(device)
            mask_pred, _ = net(img)
            CAMs = returnCAM(feature_blobs[0], weights).to(device)
            CAMs_ = torch.sigmoid(CAMs)
            cnt = 0
            mask_prop = torch.zeros(CAMs_.size())
            for sample in CAMs_:
                img_ = img[cnt, :, :, :]
                img_ = img_.reshape(1, 3, 256, 256)
                sample = sample.reshape(256,256)
                cnt1 = 0
                erase = (torch.zeros(sample.size())>1).to(device)
                #while len(sample[sample >= 0.5])>0 and not ((sample >= 0.5) == erase).all():
                for i in range(2):
                    print(cnt1)
                    if cnt1 == 19:
                        break
                    erase = sample >= 0.5
                    img_[0,0,:,:][erase] = px_avg[0]
                    img_[0,1,:,:][erase] = px_avg[1]
                    img_[0,2,:,:][erase] = px_avg[2]
                    mask_prop[0,0,:,:][erase] = 1
                    feature_blobs = []
                    sample, _ = net(img_)
                    sample = returnCAM(feature_blobs[0], weights).to(device)
                    sample = torch.sigmoid(sample)
                    sample = sample.reshape(256,256)
                    cnt1 += 1
                cnt += 1
            tot += DiceLoss(mask_prop.to(device), mask).item()
            if len(mask_prop[mask_prop>=0.5]) > 0:
                tp = tp + 1 if lb.item() >= 0.5 else tp
                fp = fp + 1 if lb.item() < 0.5 else fp
            else:
                tn = tn + 1 if lb.item() < 0.5 else tn
                fn = fn + 1 if lb.item() >= 0.5 else fn
            """
            _ = adapted_rand_error(mask.cpu().numpy().astype(int), t_.cpu().numpy().astype(int))
            tot_rand[0] += _[0]
            tot_rand[1] += _[1]
            tot_rand[2] += _[2]
            """
            tot_rand += RandLoss(mask_prop, mask)
            mask_prop.to("cpu")
            torchvision.utils.save_image(mask_prop, path_ + idx[0])
        """
        print("ROC-AUC Score: ", roc_auc_score(y_true, y_scores))
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        print("Ideal Threshold:", _[np.argmax(tpr-fpr)])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel("True Positive Rate")
        plt.title("AUC-ROC")
        plt.savefig("AUROC.png")
        """
    print(tp,fp, tn, fn)
    try:
        precision = tp/(tp+fp)
    except:
        precision = 0
    try:
        recall = tp/(tp+fn)
    except:
        recall = 0
    try:
        print("F1 score:", 2*precision*recall/(precision+recall))
    except:
        print("F1 score:", 0)
    print("Dice Loss:", tot/759) #dice loss
    print("Rand Score:", tot_rand/759)
