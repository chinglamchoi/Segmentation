import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms as transforms
import numpy as np
import os
from skimage import io

import unet_GAP
import unet_multi
import unet_model

import torch.nn as nn
import torch.optim as optim
from dice_loss import dice_coeff
import matplotlib.pyplot as plt
import argparse

import random
random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


class BrainTumour(data.Dataset):

    def __init__(self, img_dir, mask_dir, lbs, scale=1, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5])]), transform_mask=transforms.Compose([transforms.ToTensor()])):
        self.img_dir, self.mask_dir = img_dir, mask_dir
        self.scale = scale
        self.ids = [i for i in os.listdir(img_dir)]
        self.transform, self.transform_mask = transform, transform_mask        
        temp = []
        with open(lbs, "r") as f:
            for i in f:
                i = i.replace("\n", "")
                temp.append(float(i))
        self.lbs = temp
#Which way you choose

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        idx = self.ids[index]
        lb = torch.tensor([self.lbs[index]])
        mask_file = self.mask_dir + idx
        img_file = self.img_dir + idx
        print(img_file)
        img, mask = io.imread(img_file), io.imread(mask_file)
        img, mask = self.transform(img), self.transform_mask(mask)
        return (img, mask, lb, idx)


#torch.multiprocessing.freeze_support()
if __name__ == "__main__":

    trainset = BrainTumour(img_dir='./TCGA/imgs/train/', mask_dir='./TCGA/masks/train/', lbs="./TCGA/train_labels.txt")
    testset = BrainTumour(img_dir='./TCGA/imgs/test/', mask_dir='./TCGA/masks/test/', lbs="./TCGA/test_labels.txt")
    trainloader = data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)
    testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", metavar="LR", type=float, default=0.001, dest="lr")
    parser.add_argument("-eps", metavar="E", type=int, default=200, dest="eps")
    parser.add_argument("-wd", metavar="WD", type=float, default=1e-8, dest="wd")
    parser.add_argument("-m", metavar="M", type=float, default=0, dest="m")
    parser.add_argument("-split", metavar="SP", type=float, default=0.1, dest="sp")
    parser.add_argument("-opt", metavar="OPT", type=str, default="Adam", dest="opt")
    parser.add_argument("-mul", metavar="MUL", type=bool, default=False, dest="mul")
    parser.add_argument("-gap", metavar="GAP", type=bool, default=False, dest="gap")
    args = parser.parse_args()

    a = "cuda:0"
    device = torch.device(a if torch.cuda.is_available() else "cpu")
    criterion1 = nn.BCEWithLogitsLoss().to(device) #nn.CrossEntropyLoss() if class > 1
    if args.mul:
        net = unet_GAP.run_cnn() if args.gap else unet_multi.run_cnn()
        criterion2 = nn.BCELoss().to(device)
    else:
        net = unet_model.run_cnn()
        args.gap = False #does not support non-multi-task loss with GAP
    net.to(device)

    epochs, dice_cof_, alpha, best_dice = args.eps, 100, args.lr, 0.0
    if args.opt == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=alpha, weight_decay=args.wd)
    elif args.opt == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=alpha, weight_decay=args.wd, momentum=args.m)
    else:
        optimizer = optim.RMSprop(net.parameters(), lr=alpha, weight_decay=args.wd, momentum=args.m)

    train_loss, test_loss, val_loss = [], [], []

    for epoch in range(1, epochs+1):
        net = net.train()
        epoch_loss = 0.0
        for img, mask, lb, idx in trainloader:
            mask_type = torch.float32 #long if classes > 1
            img, mask = (img.to(device), mask.to(device, dtype=mask_type))
            if args.mul:
                mask_pred, decision = net(img)
                lb = lb.to(device)
                loss1 = criterion1(mask_pred, mask)
                loss2 = criterion2(decision,lb)
                loss = (1-args.sp)*loss1+args.sp*loss2
            else:
                mask_pred = net(img)
                loss1 = criterion1(mask_pred, mask)
                loss = loss1
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss.append(epoch_loss/3170)
        print("Epoch" + str(epoch) + " Train Loss:", epoch_loss/3170) 
        
        net = net.eval()
        tot = 0
        tot_val = 0.0
        for img, mask, lb, idx in testloader:
            mask_type = torch.float32
            img, mask = (img.to(device), mask.to(device, dtype=mask_type))
            if args.mul:
                mask_pred, decision = net(img)
                lb = lb.to(device)
                loss1 = criterion1(mask_pred, mask)
                loss2 = criterion2(decision,lb)
                tot_val += (1-args.sp)*loss1.item()+args.sp*loss2.item()
            else:
                mask_pred = net(img)
                loss1 = criterion1(mask_pred, mask)
                tot_val += loss1.item()
            # Test accuracy with dice coeff
            for tm, pred in zip(mask, mask_pred):
                pred = (pred > 0.5).float().to(device)
                tm = tm.to(device)
                tot += dice_coeff(pred, tm.squeeze(dim=1)).item()
        dice_cof_ = tot/759
        tot_val /= 759
        print("Epoch" + str(epoch) + " Val Loss:", tot_val)
        if dice_cof_ > best_dice:
            valid = True
            print("New best test loss:", dice_cof_)
            best_dice = dice_cof_
        else:
            valid = False
        test_loss.append(dice_cof_)
        val_loss.append(tot_val)
        print("Epoch " + str(epoch) + " Dice Coeff:", dice_cof_)
        print("\n")
        
        if valid:
            for param_group in optimizer.param_groups:
                alpha_ = param_group["lr"]
            state = {
                "net":net.state_dict(),
                "dice_cof": dice_cof_,
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "alpha": alpha_
            }
            path_ = "./TCGA/models_" + str(args.lr) + "_" + args.opt #alpha may change
            path_ = path_ + "_multi" if args.mul else path_
            path_ = path_ + "_gap" if args.gap else path_
            path_ += "/"
            try:
                os.mkdir(path_)
            except:
                pass
            """
            try:
                os.mkidir(path_)
            except:
                pass
            """
            torch.save(state, str(path_ + "best_" + str(epoch) + "_" + "{0:.2f}".format(dice_cof_) + ".pt")) #can overwrite but ok
        if epoch == epochs:
            fig = plt.figure()
            plt.plot(train_loss, label="Train")
            plt.plot(val_loss, label="Val")
            plt.xlabel("Epochs")
            plt.ylabel("BCE with Logits")
            plt.title("Train-Val Loss")
            fig.savefig(path_+ "train.png")
            fig = plt.figure()
            plt.plot(test_loss)
            plt.xlabel("Epochs")
            plt.ylabel("Dice Coefficient")
            plt.title("Test Accuracy")
            fig.savefig(path_ + "test.png")
            print("Saved plots")




