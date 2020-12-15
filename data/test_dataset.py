  
"""
The following code is adapted from: https://github.com/ignacio-rocco/cnngeometric_pytorch.
"""


from __future__ import print_function, division
import os
import torch
from torch.autograd import Variable
from skimage import io
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from geotnf.transformation import GeometricTnf

class TestDataset(Dataset):
    
    """
    
    Test image dataset
    

    Args:
        csv_file (string): Path to the csv file with image names and transformations.
        training_image_path (string): Directory with the images.
        output_size (2-tuple): Desired output size
        transform (callable): Transformation for post-processing the training pair (eg. image normalization)
        
    """

    def __init__(self, csv_file, training_image_path,output_size=(240,240),transform=None):

        self.out_h, self.out_w = output_size
        self.train_data = pd.read_csv(csv_file)
        self.source_image_names = self.train_data.iloc[:,0]
        self.target_image_names = self.train_data.iloc[:,1]
        self.training_image_path = training_image_path         
        self.transform = transform
        # no cuda as dataset is called from CPU threads in dataloader and produces confilct
        self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda = False) 
              
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        # get pre-processed images
        source_image,moving_img_size = self.get_image(self.source_image_names,idx)
        target_image,fixed_img_size = self.get_image(self.target_image_names,idx)
                
        sample = {'source_image': source_image, 'target_image': target_image, 'moving_im_size': moving_img_size, 'fixed_im_size': fixed_img_size}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_image(self,img_name_list,idx):
        img_name = os.path.join(self.training_image_path, img_name_list[idx])
        image = io.imread(img_name)
        
        
        if len(image.shape)== 2:
            image_rgb = np.zeros(image.shape[0],image.shape[1],3)
            img_rgb[:,:,0] =  image
            img_rgb[:,:,1] =  image
            img_rgb[:,:,2] =  image
            image = img_rgb
        
        # get image size
        im_size = np.asarray(image.shape)

        # convert to torch Variable
        image = np.expand_dims(image.transpose((2,0,1)),0)
        image = torch.Tensor(image.astype(np.float32))
        image_var = Variable(image,requires_grad=False)
        
        # Resize image using bilinear sampling with identity affine tnf
        image = self.affineTnf(image_var).data.squeeze(0)
        
        im_size = torch.Tensor(im_size.astype(np.float32))
        
        return (image, im_size)
    
