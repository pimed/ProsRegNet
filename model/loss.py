from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from geotnf.point_tnf import PointTnf
from geotnf.transformation import GeometricTnf
from skimage import io

class SSDLoss(nn.Module):
    def __init__(self, geometric_model='affine', use_cuda=True):
        super(SSDLoss, self).__init__()
        self.geometric_model = geometric_model
        self.use_cuda = use_cuda

    def forward(self, theta, theta_GT, tnf_batch):
        ### compute square root of ssd
        A = tnf_batch['target_image']
        geometricTnf = GeometricTnf(self.geometric_model, 240, 240, use_cuda = self.use_cuda)
        
        B = geometricTnf(tnf_batch['source_image'],theta)
        
        ssd = torch.sum(torch.sum(torch.sum(torch.pow(A - B,2),dim=3),dim=2),dim=1)
        ssd = torch.sum(ssd)/(A.shape[0]*A.shape[1]*A.shape[2]*A.shape[3])
        ssd = torch.sqrt(ssd)

        
        return  ssd 
