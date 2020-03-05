import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms as transforms
import numpy as np
import os
import unet
import unet_multi #is actually updated unet_GAP
import unet_multi_recur
from sklearn.metrics import adjusted_rand_score

from skimage import io
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import random
"""
random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
"""

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
        #seed = np.random.randint(2147483647) #31 bit
        #random.seed(seed)
        #np.random.seed(seed)
        #torch.manual_seed(seed)
        img1, mask1 = self.transform(self.img[index]), self.transform(self.mask[index])
        img1 = self.img_transform(img1)
        return (img1, mask1, lb, idx)

def lr_change(alpha):
    global optimizer
    alpha /= 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = alpha
    print("Lr changed to " + str(alpha))
    return alpha

def DiceLoss(a,b):
    tot = 256*256
    smooth = 1.
    a = a.view(-1)
    b = b.view(-1)
    intersection = (a*b).sum()
    return 1 - ((2. * intersection + smooth) / (a.sum() + b.sum() + smooth))

def RandLoss(a,b):
    a = (a>0.5).float()
    a = a.reshape((256, 256))
    b = b.reshape((256,256))
    a = a.cpu().numpy().flatten()
    b = b.cpu().numpy().flatten()
    c = adjusted_rand_score(a,b)
    c = (c+1)/2
    return 1-c

#torch.multiprocessing.freeze_support()
if __name__ == "__main__":
    class_counts = [16424.0, 8936.0]
    train_label_file = np.load("./train_labels.npy")
    weights_per_class = [25360.0/class_counts[i] for i in range(len(class_counts))]
    weights = [weights_per_class[train_label_file[i]] for i in range(25360)]
    sampler = data.sampler.WeightedRandomSampler(torch.DoubleTensor(weights), 25360)
    mean, std = [0.5], [0.5]
    """
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=[0, 360]),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
        ])
    img_transform = transforms.Normalize(mean, std)
    """
    trainset = BrainTumour(img_dir='./TCGA/imgs/train/', mask_dir='./TCGA/masks/train/', lbs=train_label_file)
    test_label_file = np.load("./test_labels.npy")
    testset = BrainTumour(img_dir='./TCGA/imgs/test/', mask_dir='./TCGA/masks/test/', lbs=test_label_file)
    trainloader = data.DataLoader(trainset, batch_size=24, shuffle=False, sampler=sampler, num_workers=4)
    testloader = data.DataLoader(testset, batch_size=24, shuffle=False, num_workers=4)

    parser = argparse.ArgumentParser()
    parser.add_argument("-pre", metavar="PRE", type=str, default=None, dest="pre")
    parser.add_argument("-lr", metavar="LR", type=float, default=0.001, dest="lr")
    parser.add_argument("-eps", metavar="E", type=int, default=400, dest="eps")
    parser.add_argument("-wd", metavar="WD", type=float, default=1e-8, dest="wd")
    parser.add_argument("-m", metavar="M", type=float, default=0, dest="m")
    parser.add_argument("-sp", metavar="SP", type=float, default=0.5, dest="sp")
    parser.add_argument("-opt", metavar="OPT", type=str, default="Adam", dest="opt")
    parser.add_argument("-cuda", metavar="CUDA", type=int, default=0, dest="cuda")
    parser.add_argument("-mul", metavar="MUL", type=bool, default=False, dest="mul") #same thing as GAP
    parser.add_argument("-rec", metavar="REC", type=bool, default=False, dest="rec")
    parser.add_argument("-stp", metavar="STP", type=int, default=100, dest="stp")
    #parser.add_argument("-rep", metavar="REP", type=int, default=2, dest="rep")
    parser.add_argument("-thresh", metavar="THRESH", type=float, default=0.5, dest="thresh") #label thresh
    #parser.add_argument("-thresh2", metvar="THRESH2", type=float, default=0.5, dest="thresh2")
    args = parser.parse_args()

    a = "cuda:" + str(args.cuda)
    device = torch.device(a if torch.cuda.is_available() else "cpu")
    criterion1 = nn.BCEWithLogitsLoss().to(device)
    if args.mul:
        if args.rec:
            net = unet_multi_recur.run_cnn()
        else:
            net = unet_multi.run_cnn()
        criterion2 = nn.BCEWithLogitsLoss().to(device)
    else:    
        net = unet.run_cnn()
    vall = False
    if args.pre is not None:
        checkpoint = torch.load(args.pre)
        net.load_state_dict(checkpoint["net"])
        vall = True
    net.to(device)

    #dsc_loss = DiceLoss()
    #best_val_dsc = 100
    best_loss = checkpoint["loss"] if vall else 100

    alpha = checkpoint["alpha"] if vall else args.lr
    if args.opt == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=alpha, weight_decay=args.wd)
    elif args.opt == "AdamW":
        optimizer = optim.AdamW(net.parameters(), lr=alpha, weight_decay=args.wd)
    elif args.opt == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=alpha, weight_decay=args.wd, momentum=args.m)
    else:
        optimizer = optim.RMSprop(net.parameters(), lr=alpha, weight_decay=args.wd, momentum=args.m)
    if vall:
        optimizer.load_state_dict(checkpoint["optimizer"])
    train_loss, val_loss = [], []
    #tpr, fpr = [], [] #y, x
    start_ = checkpoint["epoch"] if vall else 1 
    epochs = checkpoint["epoch"]+args.eps if vall else args.eps
    for epoch in range(start_, epochs+1):
        if epoch % args.stp == 0 and epoch != epochs:
            alpha = lr_change(alpha)
        net = net.train()
        epoch_loss = 0.0
        for img, mask, lb, idx in trainloader:
            mask_type = torch.float32 #long if classes > 1
            img, mask = (img.to(device), mask.to(device, dtype=mask_type))
            if args.mul:
                lb = lb.to(device)
                mask_pred, pred_lb = net(img)
                loss2 = criterion2(pred_lb, lb)
                loss1 = criterion1(mask_pred, mask)
                loss = args.sp * loss2 + (1-args.sp) * loss1
            else:
                mask_pred = net(img)
                loss = criterion1(mask_pred, mask)
            t = torch.sigmoid(mask_pred)
            epoch_loss += DiceLoss(t, mask).item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss.append(epoch_loss/25360)
        print("Epoch" + str(epoch) + " Train Loss:", epoch_loss/25360) 
        
        net = net.eval()
        tot = 0
        tot_val = 0.0
        #tp, fn, fp, tn = 0,0,0,0
        for img, mask, lb, idx in testloader:
            mask_type = torch.float32
            img, mask = (img.to(device), mask.to(device, dtype=mask_type))
            if args.mul:
                lb = lb.to(device)
                mask_pred, pred_lb = net(img)
            else:
                mask_pred = net(img)
            t = torch.sigmoid(mask_pred)
            tot_val += DiceLoss(t, mask).item()
            """
            Test accuracy with dice coeff
            for tm, pred in zip(mask, mask_pred):
                pred = (pred > 0.5).float().to(device)
                tm = tm.to(device)
                tot += dice_coeff(pred, tm.squeeze(dim=1)).item()
            """
        loss_ = tot_val/759
        #dice_cof_ = tot_val/759
        #tpr.append(tp/(tp+fn))
        #fpr.append(fp/(fp+tn))
        #print("Epoch" + str(epoch) + " Val Loss:", dice_cof_)
        print("Epoch" + str(epoch) + " Val Loss:", loss_)
        #if dice_cof_ < best_val_dsc:
        if loss_ < best_loss:
            valid = True
            print("New best test loss!")
            #best_val_dsc = dice_cof_
            best_loss = loss_
        else:
            valid = False
        #val_loss.append(dice_cof_)
        val_loss.append(loss_)
        print("\n")
        
        if valid:
            for param_group in optimizer.param_groups:
                alpha_ = param_group["lr"]
            if alpha_ != args.lr:
                print('yes')
            state = {
                "net":net.state_dict(),
                #"dice_cof": dice_cof_,
                "loss": loss_,
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "alpha": alpha_
            }
            path_ = "./TCGA/gap_lr" + str(args.lr) + "_sp" + str(args.sp) + "_" + args.opt + "_m" + str(args.m) #alpha may change
            path_ += "/"
            try:
                os.mkdir(path_)
            except:
                pass
            torch.save(state, str(path_ + "best.pt"))
        if epoch == epochs:
            fig = plt.figure()
            plt.plot(train_loss, label="Train")
            plt.plot(val_loss, label="Val")
            plt.xlabel("Epochs")
            plt.ylabel("Dice Loss")
            plt.title("Train-Val Loss")
            fig.savefig(path_+ "train.png")
            print("Saved plots")
