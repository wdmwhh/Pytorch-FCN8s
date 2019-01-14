# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 19:55:09 2018
FCN-VGG16-8s
@author: WDMWHH
"""

from __future__ import print_function, division
from torch.optim import lr_scheduler
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy import ndimage
from tqdm import tqdm
import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

datadir = input('data directory: ') # public server
voc_root = os.path.join(datadir, 'VOC2012')

def read_images(root_dir, train):
    txt_fname = root_dir + '/Segmentation/' + ('sbdtrain.txt' if train else 'seg11valid.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    data_list = [os.path.join(root_dir, 'JPEGImages', i+'.jpg') for i in images]
    label_list = [os.path.join(root_dir, 'SegmentationClass', i+'.png') for i in images]
    return data_list, label_list

class VOCDataset(Dataset):
    """ VOC2012 Dataset. """
    
    def __init__(self, root_dir=voc_root, train=True, trsf=None):
        self.root_dir = root_dir
        self.trsf = trsf
        self.data_list, self.label_list = read_images(root_dir, train)
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        image, label = self.data_list[idx], self.label_list[idx]
        image, label = Image.open(image).convert('RGB'), Image.open(label)
        sample = {'image': image, 'label': label}
        if self.trsf:
            sample = self.trsf(sample)
        return sample
    
class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = transforms.ToTensor()(image)
        label = torch.from_numpy(np.array(label, dtype='int'))
        return {'image': image, 'label': label}

class Normalize(object):
    def __init__(self, mean = [0., 0., 0.], std = [1., 1., 1.]):
        self.mean = mean
        self.std = std
    def __call__(self, sample):        
        image, label = sample['image'], sample['label']
        image = transforms.Normalize(self.mean, self.std)(image)
        return {'image': image, 'label': label}

# 定义 bilinear kernel
def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)
    
class FCN_vgg16(nn.Module):
    def __init__(self, num_category):
        super(FCN_vgg16, self).__init__()

        model_ft = models.vgg16(pretrained=True)
        features = list(model_ft.features.children())
        conv1 = nn.Conv2d(3, 64, 3, 1, 100)
        conv1.weight.data = features[0].weight.data
        conv1.bias.data = features[0].bias.data
        features[0] = conv1
        features[4] = nn.MaxPool2d(2, 2, ceil_mode=True)
        features[9] = nn.MaxPool2d(2, 2, ceil_mode=True)
        features[16] = nn.MaxPool2d(2, 2, ceil_mode=True)
        features[23] = nn.MaxPool2d(2, 2, ceil_mode=True)
        features[30] = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage1 = nn.Sequential(*features[:17]) # 第一段
        self.stage2 = nn.Sequential(*features[17:24]) # 第二段
        self.stage3 = nn.Sequential(*features[24:]) # 第三段

        #fc6, fc7
        fc = list(model_ft.classifier.children())
        fc6 = nn.Conv2d(512, 1024, 7) 
        fc7 = nn.Conv2d(1024, 1024, 1)
        fc[0] = fc6
        fc[3] = fc7
        self.fc = nn.Sequential(*fc[:6])
        


        self.scores1 = nn.Conv2d(1024, num_category, 1) #
        self.scores2 = nn.Conv2d(512, num_category, 1)
        self.scores3 = nn.Conv2d(256, num_category, 1)
        for layer in [self.scores1, self.scores2, self.scores3]:
            nn.init.kaiming_normal_(layer.weight, a=1)
            nn.init.constant_(layer.bias, 0)

        self.upsample_8x = nn.ConvTranspose2d(num_category, num_category, 16, 8, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_category, num_category, 16) # 使用双线性 kernel

        self.upsample_4x = nn.ConvTranspose2d(num_category, num_category, 4, 2, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_category, num_category, 4) # 使用双线性 kernel

        self.upsample_2x = nn.ConvTranspose2d(num_category, num_category, 4, 2, bias=False)   
        self.upsample_2x.weight.data = bilinear_kernel(num_category, num_category, 4) # 使用双线性 kernel


    def forward(self, x):
        h = self.stage1(x)
        s1 = h # 1/8

        h = self.stage2(h)
        s2 = h # 1/16

        h = self.stage3(h)
        h = self.fc(h)
        s3 = h # 1/32
               
        s3 = self.scores1(s3)
        s3 = self.upsample_2x(s3)
        s2 = self.scores2(s2*1e-2)
        s2 = s2[:, :, 5:5+s3.size()[2], 5:5+s3.size()[3]].contiguous()
        s2 = s2 + s3

        s2 = self.upsample_4x(s2)
        s1 = self.scores3(s1*1e-4)
        s1 = s1[:, :, 9:9+s2.size()[2], 9:9+s2.size()[3]].contiguous()
        s = s1 + s2

        s = self.upsample_8x(s)
        s = s[:, :, 31:31+x.size()[2], 31:31+x.size()[3]].contiguous()

        return s

    def get_params(self, split):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                if split == 'weight':
                    yield layer.weight
                else:
                    yield layer.bias
            elif isinstance(layer, nn.ConvTranspose2d) and split == 'weight':
                yield layer.weight
            

def fast_hist(label_pred, label_gt, num_category):
    mask = (label_gt >= 0) & (label_gt < num_category) # include background
    hist = np.bincount(
        num_category * label_pred[mask] + label_gt[mask].astype(int),
        minlength=num_category ** 2).reshape(num_category, num_category)
    return hist


def evaluation_metrics(label_preds, label_gts, num_category):
    """Returns evaluation result.
      - pixel accuracy
      - mean accuracy
      - mean IoU
      - frequency weighted IoU
    """
    hist = np.zeros((num_category, num_category))
    for p, g in zip(label_preds,label_gts):
        tmp = (g < 21)
        hist += fast_hist(p[tmp], g[tmp], num_category)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        macc = np.diag(hist) / hist.sum(axis=0)
    macc = np.nanmean(macc)
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.diag(hist) / (hist.sum(axis=0) + hist.sum(axis=1) - np.diag(hist))
    miou = np.nanmean(iou)
    freq = hist.sum(axis=0) / hist.sum()
    fwiou = (freq[freq > 0] * iou[freq > 0]).sum()
    return acc, macc, miou, fwiou

#%%
def main():
    #%% Initialize
    transforms_train = transforms.Compose([
            ToTensor(), 
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transforms_val = transforms.Compose([
            ToTensor(), 
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    voc_data = {'train': VOCDataset(root_dir=voc_root, train=True,
                              trsf=transforms_train), 
                'val': VOCDataset(root_dir=voc_root, train=False,
                              trsf=transforms_val)}
    dataloaders = {'train': DataLoader(voc_data['train'], batch_size=1,
                                             shuffle=True, num_workers=4), 
                   'val': DataLoader(voc_data['val'], batch_size=1,
                                             shuffle=False, num_workers=4)} #
    dataset_sizes = {x: len(voc_data[x]) for x in ['train', 'val']}
    
    
    num_category = 20 + 1 #
    
    myfcn = FCN_vgg16(num_category) #
    num_epoch = 20 #
    criterion = nn.NLLLoss(ignore_index=255)
    
    # Observe that all parameters are being optimized
    train_params = [{'params': myfcn.get_params('weight'), 'lr': 1e-4, 'weight_decay': 5e-4},
     {'params': myfcn.get_params('bias'), 'lr': 2e-4, 'weight_decay': 0}] #
    optimizer = optim.SGD(train_params, momentum=0.99) #
    
    # (LR) Decreased  by a factor of 10 every 2000 iterations
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.9) # 
    myfcn = nn.DataParallel(myfcn).cuda()
    
    since = time.time()
    #%% Train
    for t in range(num_epoch): #

        myfcn.train()  # Set model to training mode
        tbar = tqdm(dataloaders['train'])
        running_loss = 0

        # Iterate over data.
        for i, sample in enumerate(tbar):
            exp_lr_scheduler.step()
            inputs, labels = sample['image'], sample['label']
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # forward
                outputs = myfcn(inputs)
                outputs = F.log_softmax(outputs, dim=1)
                loss = criterion(outputs, labels.long())
                # backward + optimize
                loss.backward()
                optimizer.step()
            
            # statistics
            running_loss += loss.item() * inputs.size(0)
        train_loss = running_loss / dataset_sizes['train']
        print('Training Results({}): '.format(t))
        print('Loss: {:4f}'.format(train_loss))
        
        
    
    #%% Save model
    state = {'net':myfcn.state_dict(), 'optimizer':optimizer.state_dict(), 'num_epoch':num_epoch}
    torch.save(state, os.path.join(datadir, 'myfcn.pth'))
    
    #%% Evaluate
    
    myfcn.eval()   # Set model to evaluate mode
    running_acc = 0
    running_macc = 0
    running_miou = 0
    running_fwiou = 0
    for sample in tqdm(dataloaders['val']):
        inputs, labels = sample['image'], sample['label']
        inputs = inputs.cuda()
        labels = labels.cuda()
        # forward
        outputs = myfcn(inputs)
        outputs = F.log_softmax(outputs, dim=1)
        preds = outputs.data.cpu().numpy()
        labels = labels.data.cpu().numpy()
        h, w = labels.shape[1:]
        ori_h, ori_w = preds.shape[2:]
        preds = np.argmax(ndimage.zoom(preds, (1., 1., 1.*h/ori_h, 1.*w/ori_w), order=1), axis=1)
        for pred, label in zip(preds, labels):
            acc, macc, miou, fwiou = evaluation_metrics(pred, label, num_category)
            running_acc += acc
            running_macc += macc
            running_miou += miou
            running_fwiou += fwiou
        
    val_acc = running_acc / dataset_sizes['val']
    val_macc = running_macc / dataset_sizes['val']
    val_miou = running_miou / dataset_sizes['val']
    val_fwiou = running_fwiou / dataset_sizes['val']
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
    print('Validation Results: ')
    print('Pixel accuracy: {:4f}'.format(val_acc))
    print('Mean accuracy: {:4f}'.format(val_macc))
    print('Mean IoU: {:4f}'.format(val_miou))    
    print('frequency weighted IoU: {:4f}'.format(val_fwiou))
        
    #%% Visualize
    # RGB color for each class
    colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128],[255, 255, 255]]
    cm = np.array(colormap, dtype='uint8')
    
    _, figs = plt.subplots(6, 3, figsize=(12, 10))
    for t in range(6):
        
        val_sample = voc_data['val'][t]
        val_image = val_sample['image'].cuda()
        val_label = val_sample['label']
        val_output = myfcn(val_image.unsqueeze(0))
        val_pred = val_output.max(dim=1)[1].squeeze(0).data.cpu().numpy()
        val_label = val_label.long().data.numpy()
        val_image = val_image.squeeze().data.cpu().numpy().transpose((1, 2, 0))
        val_image = val_image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        val_image *= 255
        val_image = val_image.astype(np.uint8)
        val_pred = cm[val_pred]
        val_label[val_label==255] = 21
        val_label = cm[val_label]
        figs[t, 0].imshow(val_image)
        figs[t, 0].axes.get_xaxis().set_visible(False)
        figs[t, 0].axes.get_yaxis().set_visible(False)
        figs[t, 1].imshow(val_label)
        figs[t, 1].axes.get_xaxis().set_visible(False)
        figs[t, 1].axes.get_yaxis().set_visible(False)
        figs[t, 2].imshow(val_pred)
        figs[t, 2].axes.get_xaxis().set_visible(False)
        figs[t, 2].axes.get_yaxis().set_visible(False)
    plt.savefig('val0_6.jpg')
    
        

#%%
if __name__ == '__main__':
    main()
