import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms as transforms
import numpy as np
import os
import unet
#import unet_multi ##ADD PRETRAINED MULTI MODE
from sklearn.metrics import adjusted_rand_score

from skimage import io
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import random

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

def DiceLoss(a,b):
    tot = 256*256
    smooth = 1.
    a = a.view(-1)
    b = b.view(-1)
    intersection = (a*b).sum()
    return 1 - ((2. * intersection + smooth) / (a.sum() + b.sum() + smooth))

class UNet2(nn.Module):
    def __init__(self):
        super(UNet2, self).__init__()
        self.outc1 = nn.AdaptiveAvgPool2d((1,1))
        self.outc2 = nn.Linear(1,1)
    def forward(self, x):
        out = self.outc1(x)
        out = self.outc2(out.view(out.size(0), -1))
        return x, out

feature_blobs = []
def hook_feature(module, input, output):
    feature_blobs.append(output.cpu().data.numpy())

def returnCAM(feature_conv, weight):
#def returnCAM(feature_conv, weight, class_idx):
    _, nc, h, w = feature_conv.shape
    output_cam = []
    for i in range(_):
        #weights 0 not class_idx[i] cuz binary
        cam = weight.dot(feature_conv[i].reshape((nc, h*w)))
        cam = (torch.from_numpy(cam.reshape(h, w))).reshape(1, h, w)
        output_cam.append(cam)
    return torch.stack(output_cam)

if __name__ == "__main__":
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
    parser.add_argument("-stp", metavar="STP", type=int, default=100, dest="stp")
    parser.add_argument("-thresh", metavar="THRESH", type=float, default=0.5, dest="thresh") #label thresh
    args = parser.parse_args()

    a = "cuda:" + str(args.cuda)
    device = torch.device(a if torch.cuda.is_available() else "cpu")
    vall = True
    
    net = unet.UNet()
    criterion2 = nn.BCEWithLogitsLoss().to(device)
    checkpoint = torch.load(args.pre)
    net.load_state_dict(checkpoint["net"])
    UNet2().to(device)
    """
    for child in net.children():
        for param in child.parameters():
            param.requires_grad = False
    """
    net.to(device)
    model = nn.Sequential(net, UNet2())
    model.to(device)
    model._modules["0"]._modules.get("conv").register_forward_hook(hook_feature)

    best_loss = 100.0
    alpha = args.lr

    if args.opt == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=alpha, weight_decay=args.wd)
        #optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=alpha, weight_decay=args.wd)
    elif args.opt == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=alpha, weight_decay=args.wd)
    elif args.opt == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=alpha, weight_decay=args.wd, momentum=args.m)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=alpha, weight_decay=args.wd, momentum=args.m)
    train_loss, val_loss, test_loss = [], [], []
    #tpr, fpr = [], [] #y, x
    start_ = 1 
    epochs = args.eps

    for epoch in range(start_, epochs+1):
        if epoch % args.stp == 0 and epoch != epochs:
            alpha = lr_change(alpha)
        model = model.train()
        epoch_loss = 0.0
        for img, mask, lb, idx in trainloader:
            weights = np.squeeze(list(model.parameters())[-2].cpu().data.numpy())
            feature_blobs = []

            img = img.to(device)
            mask = mask.to(device)
            mask_pred, pred_lb = model(img)
            
            """
            h_x = torch.sigmoid(pred_lb)
            h_x = [pred_lb_i.data.squeeze() for pred_lb_i in pred_lb] 
            h_x_ = [h_x_i.sort(0, True) for h_x_i in h_x]
            probs, idx = [], []
            for temp_hx in h_x_:
                probs.append(temp_hx[0])
                idx.append([temp_hx[1].item()])
            CAMs = returnCAM(feature_blobs[0], weights, idx) #need idx/item() as list
            """
            CAMs = returnCAM(feature_blobs[0], weights).to(device)
            loss2 = criterion2(CAMs, mask)
            loss2.requires_grad = True
            epoch_loss += loss2.item()
            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()
        train_loss.append(epoch_loss/25360)
        print("Epoch" + str(epoch) + " Train Loss:", epoch_loss/25360) 
        
        model = model.eval()
        tot = 0
        tot_val = 0.0
        for img, mask, lb, idx in testloader:
            weights = np.squeeze(list(model.parameters())[-2].cpu().data.numpy())
            feature_blobs = []
            img = img.to(device)
            lb = lb.to(device)
            mask = mask.to(device)
            mask_pred, pred_lb = model(img)
            CAMs = returnCAM(feature_blobs[0], weights).to(device)
            CAMs_sig = (torch.sigmoid(CAMs) >= 0.5).float()
            tot += DiceLoss(CAMs_sig, mask).item()
            tot_val += criterion2(CAMs, mask).item()
        loss_ = tot_val/759
        print("Epoch" + str(epoch) + " Val Loss:", loss_)
        val_loss.append(loss_)
        loss = tot/759
        print("Epoch" + str(epoch) + " Test Loss:", loss_)
        if loss_ < best_loss:
            valid = True
            print("New best test loss!")
            best_loss = loss_
        else:
            valid = False
        #val_loss.append(dice_cof_)
        test_loss.append(loss_)
        print("\n")
        
        if valid:
            for param_group in optimizer.param_groups:
                alpha_ = param_group["lr"]
            state = {
                "net":model.state_dict(),
                #"dice_cof": dice_cof_,
                "loss": loss_,
                "epoch": epoch,
                "alpha": alpha_
            }
            path_ = "./TCGA/gapgap_lr" + str(args.lr) + "_" + args.opt + "_m" + str(args.m) #alpha may change
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
            plt.ylabel("BCE Loss")
            plt.title("Train-Val Loss")
            fig.savefig(path_+ "train.png")
            fig = plt.figure()
            plt.plot(test_loss)
            plt.xlabel("Epochs")
            plt.ylabel("Dice Loss")
            plt.title("Dice Test Loss")
            fig.savefig(path_+ "test.png")
            print("Saved plots")
