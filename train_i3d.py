import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-max_steps', type=int)
parser.add_argument('-batch_size', type=int)
parser.add_argument('-train_split', type=str)
parser.add_argument('-num_classes', type=int)
args = parser.parse_args()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms

import numpy as np

from pytorch_i3d import InceptionI3d
from charades_dataset import Charades as Dataset

def run(init_lr=0.1, max_steps=64e3, mode='rgb', root='/ssd/Charades_v1_rgb', train_split='charades/charades.json', batch_size=8*5, save_model='', num_classes=157):
    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip()])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    dataset = Dataset(train_split, 'training', root, mode, train_transforms, num_classes)
    print("Length of training data: " + str(len(dataset)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    val_dataset = Dataset(train_split, 'testing', root, mode, test_transforms, num_classes)
    print("Length of testing data: " + str(len(val_dataset)))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset} 
    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    i3d.replace_logits(num_classes)
    #i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])


    num_steps_per_update = 4 # accum gradient
    steps = 0
    # train it
    while steps < max_steps:#for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, max_steps))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
          if phase == 'train':
              i3d.train(True)
          else:
              i3d.train(False)  # Set model to evaluate mode
              
          tot_loss = 0.0
          tot_loc_loss = 0.0
          tot_cls_loss = 0.0
          num_iter = 0
          optimizer.zero_grad()
          
          # Iterate over data.
          for data in dataloaders[phase]:
            num_iter += 1
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs = Variable(inputs.cuda())
            # inputs = Variable(inputs)
            t = inputs.size(2)
            labels = Variable(labels.cuda())
            # labels = Variable(labels)
            per_frame_logits = i3d(inputs)
            # upsample to input size
            per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear')
            # compute localization loss
            loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
            tot_loc_loss += loc_loss.data
            # compute classification loss (with max-pooling along time B x C x T)
            cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
            tot_cls_loss += cls_loss.data
            loss = (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update
            tot_loss += loss.data
            loss.backward()

            if num_iter == num_steps_per_update and phase == 'train':
              steps += 1
              num_iter = 0
              optimizer.step()
              optimizer.zero_grad()
              lr_sched.step()
              if steps % 10 == 0:
                print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/(10*num_steps_per_update), tot_cls_loss/(10*num_steps_per_update), tot_loss/10))
                # save model
                torch.save(i3d.module.state_dict(), save_model+str(steps).zfill(6)+'.pt')
                tot_loss = tot_loc_loss = tot_cls_loss = 0.
          if phase == 'val':
            print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter))
          # del loss
          # del per_frame_logits
    
def _run(**kwargs):
   print(kwargs)


if __name__ == '__main__':
    # need to add argparse
    #run(mode=args.mode, root=args.root, max_steps=int(args.max_steps), batch_size=int(args.batch_size), train_split=args.train_split, save_model=args.save_model, num_classes=args.num_classes)
    # print(args)
    _run(**vars(args))