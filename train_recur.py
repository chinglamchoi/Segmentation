import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms as transforms
import numpy as np
import os
import unet
import unet_multi #is actually updated unet_GAP

from skimage import io
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import random

def DiceLoss(a, b):
    tot = 256*256
    smooth = 1.
    a = a.view(-1)
    b = b.view(-1)
    intersection = (a*b).sum()
    return ((2. * intersection + smooth) / (a.sum() + b.sum() + smooth))


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

class UNet2(nn.Module):
    def __init__(self):
        super(UNet2, self).__init__()
        self.outc1 = nn.AdaptiveAvgPool2d((1,1))
        self.outc2 = nn.Linear(1,1)
    def forward(self, x):
        out = self.outc1(x)
        out = self.outc2(out.view(out.size(0), -1))
        return x, out

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

def lr_change(alpha):
    global optimizer
    alpha /= 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = alpha
    print("Lr changed to " + str(alpha))
    return alpha


if __name__ == "__main__":
    px_avg = np.array([23.31432984, 21.05975591, 22.50303842])
    #px_avg = np.array([0.0,0.0,0.0])
    class_counts = [16424.0, 8936.0]
    train_label_file = np.load("./train_labels.npy")
    weights_per_class = [25360.0/class_counts[i] for i in range(len(class_counts))]
    weights = [weights_per_class[train_label_file[i]] for i in range(25360)]
    sampler = data.sampler.WeightedRandomSampler(torch.DoubleTensor(weights), 25360)
    mean, std = [0.5], [0.5]
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
    parser.add_argument("-opt", metavar="OPT", type=str, default="Adam", dest="opt")
    parser.add_argument("-cuda", metavar="CUDA", type=int, default=0, dest="cuda")
    parser.add_argument("-mul", metavar="MUL", type=bool, default=False, dest="mul") #same thing as GAP
    parser.add_argument("-rec", metavar="REC", type=bool, default=False, dest="rec")
    parser.add_argument("-stp", metavar="STP", type=int, default=100, dest="stp")
    #parser.add_argument("-rep", metavar="REP", type=int, default=2, dest="rep")
    args = parser.parse_args()

    a = "cuda:" + str(args.cuda)
    device = torch.device(a if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss().to(device)
    try:
        net = unet_multi.run_cnn()
        if args.pre is not None:
            checkpoint = torch.load(args.pre)
            net.load_state_dict(checkpoint["net"])
        net._modules.get("conv").register_forward_hook(hook_feature)
    except:
        net1 = unet.UNet()
        net = nn.Sequential(net1, UNet2())
        if args.pre is not None:
            checkpoint = torch.load(args.pre)
            net.load_state_dict(checkpoint["net"])
        net._modules["0"]._modules.get("conv").register_forward_hook(hook_feature)
    net.to(device)

    best_acc = 0.0

    alpha = args.lr
    if args.opt == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=alpha, weight_decay=args.wd)
    elif args.opt == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=alpha, weight_decay=args.wd, momentum=args.m)
    else:
        optimizer = optim.RMSprop(net.parameters(), lr=alpha, weight_decay=args.wd, momentum=args.m)
    train_loss, val_loss, test_loss = [], [], []
    epochs = args.eps
    for epoch in range(1, epochs+1):
        if epoch % args.stp == 0 and epoch != epochs:
            alpha = lr_change(alpha)
        net = net.train()
        losses = []
        for img, mask, lb, idx in trainloader:
            weights = np.squeeze(list(net.parameters())[-2].cpu().data.numpy())
            feature_blobs = []
            mask_type = torch.float32 #long if classes > 1
            img, mask = (img.to(device), mask.to(device, dtype=mask_type))
            mask_pred, pred_lb = net(img)
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
                
                while len(sample[sample >= 0.5])>0 and not ((sample >= 0.5) == erase).all():
                    if cnt1 == 19:
                        break
                    #erase = np.array(np.unravel_index(torch.topk(sample.view(-1), k=51).indices.cpu().numpy(), save_size))
                   
                #while cnt1 < 3:
                    erase = (sample >= 0.5).to(device)
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
                ##losses.append(DiceLoss(mask_prop.to(device), mask))
                losses.append(criterion(mask_prop.to(device), mask))
            optimizer.zero_grad()
            loss = sum(losses)
            loss.requires_grad = True
            loss.backward()
            optimizer.step()
        epoch_loss = loss.item()/1056.7
        train_loss.append(epoch_loss)
        print("Epoch" + str(epoch) + " Train Loss:", epoch_loss) 
        
        net = net.eval()
        tot = 0
        test_acc = 0.0
        loss = 0.0
        with torch.no_grad():
            for img, mask, lb, idx in testloader:
                weights = np.squeeze(list(net.parameters())[-2].cpu().data.numpy())
                feature_blobs = []
                mask_type = torch.float32
                img, mask = (img.to(device), mask.to(device, dtype=mask_type))
                mask_pred, pred_lb = net(img)
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
                    while len(sample[sample >= 0.5])>0 and not ((sample >= 0.5) == erase).all():
                        if cnt1 == 19:
                            break
                    #while cnt1 < 3:
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
                    mask_prop = mask_prop.to(device)
                    ##loss += DiceLoss(mask_prop.to(device), mask).item()
                    loss += criterion(mask_prop, mask).item()
                    test_acc += DiceLoss(mask_prop, mask).item()
        loss /= 31.6
        test_acc /= 31.6
        print("Epoch" + str(epoch) + " Val Loss:", loss)
        print("Epoch" + str(epoch) + " Test Accuracy:", test_acc)
        if test_acc > best_acc:
        ##if loss < best_loss:
            valid = True
            print("New best test accuracy!")
            ##best_loss = loss
            best_acc = test_acc
        else:
            valid = False
        val_loss.append(loss)
        test_loss.append(test_acc)
        print("\n")
        
        if valid:
            for param_group in optimizer.param_groups:
                alpha_ = param_group["lr"]
            state = {
                "net":net.state_dict(),
                "acc": test_acc,
                ##"acc": loss,
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "alpha": alpha_
            }
            path_ = "./TCGA/recurgap_lr" + str(args.lr) + "_" + args.opt + "_m" + str(args.m) #alpha may change
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
            fig = plt.figure()
            plt.plot(test_loss)
            plt.xlabel("Epochs")
            plt.ylabel("Test Accuracy")
            fig.savefig(path_ + "test.png")
            print("Saved plots")
