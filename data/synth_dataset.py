from __future__ import print_function, division
import torch
import os
from os.path import exists, join, basename
from skimage import io
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from geotnf.transformation import GeometricTnf
from torch.autograd import Variable

class SynthDataset(Dataset):
    """
    
    Synthetically transformed pairs dataset for training with strong supervision
    
    Args:
            csv_file (string): Path to the csv file with image names and transformations.
            training_image_path (string): Directory with all the images.
            transform (callable): Transformation for post-processing the training pair (eg. image normalization)
            
    Returns:
            Dict: {'source_image': source_image, 'target_image': target_image, 'theta': desired transformation}
            
    """

    def __init__(self, csv_file, training_image_path, output_size=(240,240), geometric_model='affine', transform=None,
                 random_sample=False, random_t=0.5, random_s=0.5, random_alpha=1/6, random_t_tps=0.4):
        # random_sample is used to indicate whether deformation coefficients are randomly generated?
        self.random_sample = random_sample
        self.random_t = random_t
        self.random_t_tps = random_t_tps
        self.random_alpha = random_alpha
        self.random_s = random_s
        self.out_h, self.out_w = output_size
        # read csv file
        self.train_data = pd.read_csv(csv_file)
        self.img_A_names = self.train_data.iloc[:,0]
        self.img_B_names = self.train_data.iloc[:,1]
        self.theta_array = self.train_data.iloc[:, 2:].values.astype('float')
        # copy arguments
        self.training_image_path = training_image_path
        self.transform = transform
        self.geometric_model = geometric_model
        # affine transform used to rescale images
        self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda = False) 
        
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        # read image
        img_A_name = os.path.join(self.training_image_path, self.img_A_names[idx])
        image_A = io.imread(img_A_name)

        img_B_name = os.path.join(self.training_image_path, self.img_B_names[idx])
        image_B = io.imread(img_B_name)
        
        # read theta
        if self.random_sample==False:
            theta = self.theta_array[idx, :]

            if self.geometric_model=='affine':
                # reshape theta to 2x3 matrix [A|t] where 
                # first row corresponds to X and second to Y
                theta = theta[[3,2,5,1,0,4]].reshape(2,3)
            elif self.geometric_model=='tps':
                theta = np.expand_dims(np.expand_dims(theta,1),2)            
        
        # make arrays float tensor for subsequent processing
        image_A = torch.Tensor(image_A.astype(np.float32))
        image_B = torch.Tensor(image_B.astype(np.float32))
        theta = torch.Tensor(theta.astype(np.float32))
        
        # permute order of image to CHW
        image_A = image_A.transpose(1,2).transpose(0,1)
        image_B = image_B.transpose(1,2).transpose(0,1)
                
        # Resize image using bilinear sampling with identity affine tnf
        if image_A.size()[0]!=self.out_h or image_A.size()[1]!=self.out_w:
            image_A = self.affineTnf(Variable(image_A.unsqueeze(0),requires_grad=False)).data.squeeze(0)

        if image_B.size()[0]!=self.out_h or image_B.size()[1]!=self.out_w:
            image_B = self.affineTnf(Variable(image_B.unsqueeze(0),requires_grad=False)).data.squeeze(0)
                
        sample = {'image_A': image_A, 'image_B': image_B, 'theta': theta}
        
        if self.transform:
            sample = self.transform(sample)

        return sample
