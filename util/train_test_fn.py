"""
The following code is adapted from: https://github.com/ignacio-rocco/cnngeometric_pytorch.
"""

from __future__ import print_function, division
import torch
from skimage import io
from collections import OrderedDict
from image.normalization import NormalizeImageDict, normalize_image
from geotnf.transformation import GeometricTnf
import torch
import torch.nn as nn

def train(epoch,model,loss_fn,optimizer,dataloader,pair_generation_tnf,use_cuda=True,log_interval=50):
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        tnf_batch = pair_generation_tnf(batch)
        theta = model(tnf_batch)
        loss = loss_fn(theta,tnf_batch['theta_GT'],tnf_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.data.cpu().numpy()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                epoch, batch_idx , len(dataloader),
                100. * batch_idx / len(dataloader), loss.data))
    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.6f}'.format(train_loss))
    return train_loss

def test(model,loss_fn,dataloader,pair_generation_tnf,use_cuda=True,geometric_model='affine'):
    model.eval()
    test_loss = 0
    dice = 0
    for batch_idx, batch in enumerate(dataloader):
        tnf_batch = pair_generation_tnf(batch)
        theta = model(tnf_batch)
        loss = loss_fn(theta,tnf_batch['theta_GT'],tnf_batch)
        test_loss += loss.data.cpu().numpy()
        
        
        I = tnf_batch['target_mask']
        geometricTnf = GeometricTnf(geometric_model, 240, 240, use_cuda = use_cuda)

        if geometric_model == 'affine':
            theta = theta.view(-1,2,3)
        J = geometricTnf(tnf_batch['source_mask'],theta)
        
        if use_cuda:
            I = I.cuda()
            J = J.cuda()
        
        numerator = 2 * torch.sum(torch.sum(torch.sum(I * J,dim=3),dim=2),dim=1)
        denominator = torch.sum(torch.sum(torch.sum(I + J,dim=3),dim=2),dim=1)
        dice = dice + torch.sum(numerator/(denominator + 0.00001))/I.shape[0]

    test_loss /= len(dataloader)
    dice /=len(dataloader)
    
    print('Test set: Average loss: {:.6f}'.format(test_loss))
    print('Test set: Dice: {:.6f}'.format(dice))
    return test_loss
