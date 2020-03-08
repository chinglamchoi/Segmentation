import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms as transforms
import numpy as np
import os
import unet
import unet_multi #is actually updated unet_GAP

from loss import DiceLoss
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

def DiceLoss(a, b):
    tot = 256*256
    smooth = 1.
    a = a.view(-1)
    b = b.view(-1)
    intersection = (a*b).sum()
    return ((2. * intersection + smooth) / (a.sum() + b.sum() + smooth))


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


#torch.multiprocessing.freeze_support()
if __name__ == "__main__":
    px_avg = np.array([23.31432984, 21.05975591, 22.50303842])
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
    parser.add_argument("-eps", metavar="E", type=int, default=200, dest="eps")
    parser.add_argument("-wd", metavar="WD", type=float, default=1e-8, dest="wd")
    parser.add_argument("-m", metavar="M", type=float, default=0, dest="m")
    parser.add_argument("-split", metavar="SP", type=float, default=0.5, dest="sp")
    parser.add_argument("-opt", metavar="OPT", type=str, default="Adam", dest="opt")
    parser.add_argument("-cuda", metavar="CUDA", type=int, default=0, dest="cuda")
    parser.add_argument("-mul", metavar="MUL", type=bool, default=False, dest="mul") #same thing as GAP
    parser.add_argument("-rec", metavar="REC", type=bool, default=False, dest="rec")
    parser.add_argument("-stp", metavar="STP", type=int, default=50, dest="stp")
    #parser.add_argument("-rep", metavar="REP", type=int, default=2, dest="rep")
    parser.add_argument("-thresh", metavar="THRESH", type=float, default=0.5, dest="thresh") #label thresh
    parser.add_argument("-thresh2", metavar="THRESH2", type=float, default=0.4, dest="thresh2")
    args = parser.parse_args()

    a = "cuda:" + str(args.cuda)
    device = torch.device(a if torch.cuda.is_available() else "cpu")
    criterion1 = nn.BCELoss().to(device)
    if args.mul:
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
    best_loss = checkpoint["loss"] if vall else 0.0

    alpha = checkpoint["alpha"] if vall else args.lr
    if args.opt == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=alpha, weight_decay=args.wd)
    elif args.opt == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=alpha, weight_decay=args.wd, momentum=args.m)
    else:
        optimizer = optim.RMSprop(net.parameters(), lr=alpha, weight_decay=args.wd, momentum=args.m)
    if vall:
        optimizer.load_state_dict(checkpoint["optimizer"])
    train_loss, val_loss, test_loss = [], [], []
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
                if args.rec:
                    mask_pred = torch.sigmoid(mask_pred)
                    pred_lbs = torch.sigmoid(pred_lb)
                    pred_lb_ = pred_lbs
                    mask_prop = torch.zeros(mask_pred.size())
                    cnt = 0
                    for sample in pred_lbs:
                        img_ = img[cnt, :, :, :]
                        img_ = img_.reshape(1, 3, 256, 256)
                        mask_pred_ = mask_pred[cnt, :, :, :] #3D
                        save_size = mask_pred.size()
                        mask_pred_ = mask_pred_.reshape(1, 1, 256, 256)
                        #save_size = mask_pred_.size()
                        while sample >= args.thresh2:
                            erase = np.array(np.unravel_index(torch.topk(mask_pred_.view(-1), k=51).indices.cpu().numpy(), save_size)) #0.2
                            img_[erase] = px_avg[0]
                            mask_prop[erase] = 1
                            erase[1] = np.ones(len(erase[1]))
                            img_[erase] = px_avg[1]
                            erase[1] = erase[1] + np.ones(len(erase[1]))
                            img_[erase] = px_avg[2]
                            mask_pred_, sample = net(img_)
                            sample = torch.sigmoid(sample)
                            mask_pred_ = torch.sigmoid(mask_pred_)
                        cnt += 1
                else:
                    mask_prop = torch.sigmoid(mask_pred)
                loss1 = criterion1(mask_prop.to(device), mask)
                loss2 = criterion2(pred_lb, lb)
                loss = args.sp * loss2 + (1-args.sp) * loss1
            else:
                mask_pred = net(img)
                mask_prop = torch.sigmoid(mask_pred)
                loss = criterion1(mask_prop, mask)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss.append(epoch_loss/25360)
        print("Epoch" + str(epoch) + " Train Loss:", epoch_loss/25360) 
        
        net = net.eval()
        tot = 0
        tot_val = 0.0
        test_acc = 0.0
        for img, mask, lb, idx in testloader:
            mask_type = torch.float32
            img, mask = (img.to(device), mask.to(device, dtype=mask_type))
            if args.mul:
                lb = lb.to(device)
                mask_pred, pred_lb = net(img)
                if args.rec:
                    mask_pred = torch.sigmoid(mask_pred)
                    pred_lbs = torch.sigmoid(pred_lb)
                    pred_lb_ = pred_lbs
                    mask_prop = torch.zeros(mask_pred.size())
                    cnt = 0
                    for sample in pred_lbs:
                        img_ = img[cnt, :, :, :]
                        img_ = img_.reshape(1, 3, 256, 256)
                        mask_pred_ = mask_pred[cnt, :, :, :]
                        save_size = mask_pred.size()
                        mask_pred_ = mask_pred_.reshape(1, 1, 256, 256)
                        while sample >= args.thresh2:
                            erase = np.array(np.unravel_index(torch.topk(mask_pred_.view(-1), k=51).indices.cpu().numpy(), save_size))
                            img_[erase] = px_avg[0]
                            mask_prop[erase] = 1 #1.0?
                            erase[1] = np.ones(len(erase[1]))
                            img_[erase] = px_avg[1]
                            erase[1] = erase[1] + np.ones(len(erase[1]))
                            img_[erase] = px_avg[2]
                            mask_pred_, sample = net(img_)
                            sample = torch.sigmoid(sample)
                            mask_pred_ = torch.sigmoid(mask_pred_)
                        cnt += 1
                else:
                    mask_prop = torch.sigmoid(mask_pred)
                loss1 = criterion1(mask_prop.to(device), mask)
                loss2 = criterion2(pred_lb, lb)
                loss = args.sp * loss2 + (1-args.sp) * loss1
            else:
                mask_pred = net(img)
                mask_prop = torch.sigmoid(mask_pred)
                loss = criterion1(mask_prop, mask)
            mask_prop = (mask_prop >= 0.5).float()
            test_acc += DiceLoss(mask_prop, mask).item()
            tot_val += loss.item()
        loss_ = tot_val/759
        test_acc /= 759
        print("Epoch" + str(epoch) + " Val Loss:", loss_)
        print("Epoch" + str(epoch) + " Test Accuracy:", test_acc)
        if test_acc > best_acc:
            valid = True
            print("New best test loss!")
            best_loss = test_acc
        else:
            valid = False
        val_loss.append(loss_)
        test_loss.append(test_acc)
        print("\n")
        
        if valid:
            for param_group in optimizer.param_groups:
                alpha_ = param_group["lr"]
            if alpha_ != args.lr:
                print('yes')
            state = {
                "net":net.state_dict(),
                #"dice_cof": dice_cof_,
                "acc": test_acc,
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
            plt.ylabel("BCE with Logits")
            plt.title("Train-Val Loss")
            fig.savefig(path_+ "train.png")

            fig = plt.figure()
            plt.plot(test_loss)
            plt.xlabel("Epochs")
            plt.ylabel("Test Accuracy")
            fig.savefig(path_ + "test.png")
            print("Saved plots")
