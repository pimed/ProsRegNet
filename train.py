  
"""
If you use this code, please cite the following papers:
(1) Shao, Wei, et al. "ProsRegNet: A Deep Learning Framework for Registration of MRI and Histopathology Images of the Prostate." Medical Image Analysis. 2020.
(2) Rocco, Ignacio, Relja Arandjelovic, and Josef Sivic. "Convolutional neural network architecture for geometric matching." Proceedings of CVPR. 2017.

The following code is adapted from: https://github.com/ignacio-rocco/cnngeometric_pytorch.
"""


from __future__ import print_function, division
import argparse
import os
from os.path import exists, join, basename
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model.ProsRegNet_model import ProsRegNet
from model.loss import SSDLoss
from data.synth_dataset import SynthDataset
from geotnf.transformation import SynthPairTnf
from image.normalization import NormalizeImageDict
from util.train_test_fn import train, test
from util.torch_util import save_checkpoint, str_to_bool
import numpy as np
from collections import OrderedDict
from torch.optim.lr_scheduler import StepLR


# Argument parsing
parser = argparse.ArgumentParser(description='ProsRegNet PyTorch implementation')
# Paths
parser.add_argument('--training-tnf-csv', type=str, default='', help='path to training transformation csv folder')
parser.add_argument('--training-image-path', type=str, default='', help='path to folder containing training images')
parser.add_argument('--trained-models-dir', type=str, default='trained_models', help='path to trained models folder')
parser.add_argument('--trained-models-fn', type=str, default='CombinedLoss', help='trained model filename')

parser.add_argument('--pretrained-model-aff', type=str, default='', help='path to a pretrained affine network')
parser.add_argument('--pretrained-model-tps', type=str, default='', help='path to a pretrained tps network')


# Optimization parameters 
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.95, help='gamma')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum constant')
parser.add_argument('--num-epochs', type=int, default=50, help='number of training epochs')
parser.add_argument('--batch-size', type=int, default=64,help='training batch size')
parser.add_argument('--weight-decay', type=float, default=0, help='weight decay constant')
parser.add_argument('--seed', type=int, default=1, help='Pseudo-RNG seed')
# Model parameters
parser.add_argument('--geometric-model', type=str, default='affine', help='geometric model to be regressed at output: affine or tps')
parser.add_argument('--use-mse-loss', type=str_to_bool, nargs='?', const=True, default=False, help='Use MSE loss on tnf. parameters')
parser.add_argument('--feature-extraction-cnn', type=str, default='resnet101', help='Feature extraction architecture: vgg/resnet101')
# Synthetic dataset parameters
parser.add_argument('--random-sample', type=str_to_bool, nargs='?', const=True, default=False, help='sample random transformations')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()

print("Use Cuda? ", use_cuda)

torch.cuda.set_device(0)
print("cuda:", torch.cuda.current_device())


do_aff = not args.pretrained_model_aff==''
do_tps = not args.pretrained_model_tps==''


# Seed
if use_cuda:
    torch.cuda.manual_seed(args.seed)

if args.training_image_path == '':
    args.training_image_path = 'datasets/training/'
if args.training_tnf_csv == '' and args.geometric_model=='affine':
    args.training_tnf_csv = 'training_data/affine'
elif args.training_tnf_csv == '' and args.geometric_model=='tps':
        args.training_tnf_csv = 'training_data/tps'

# CNN model and loss
print('Creating CNN model...')

model = ProsRegNet(use_cuda=use_cuda,geometric_model=args.geometric_model,feature_extraction_cnn=args.feature_extraction_cnn)

if args.geometric_model == 'affine' and do_aff:
    checkpoint = torch.load(args.pretrained_model_aff, map_location=lambda storage, loc: storage)
    checkpoint['state_dict'] = OrderedDict([(k.replace(args.feature_extraction_cnn, 'model'), v) for k, v in checkpoint['state_dict'].items()])
    model.load_state_dict(checkpoint['state_dict'])
        
if args.geometric_model == 'tps' and do_tps:
    checkpoint = torch.load(args.pretrained_model_tps, map_location=lambda storage, loc: storage)
    checkpoint['state_dict'] = OrderedDict([(k.replace(args.feature_extraction_cnn, 'model'), v) for k, v in checkpoint['state_dict'].items()])
    model.load_state_dict(checkpoint['state_dict'])


if args.use_mse_loss:
    print('Using MSE loss...')
    loss = nn.MSELoss()
else:
    print('Using SSD loss...')
    loss = SSDLoss(use_cuda=use_cuda,geometric_model=args.geometric_model)


# Dataset and dataloader
dataset = SynthDataset(geometric_model=args.geometric_model,
                       csv_file=os.path.join(args.training_tnf_csv,'train.csv'),
                       training_image_path=args.training_image_path,
                       transform=NormalizeImageDict(['image_A','image_B']),
                       random_sample=args.random_sample)

dataloader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)

dataset_test = SynthDataset(geometric_model=args.geometric_model,
                            csv_file=os.path.join(args.training_tnf_csv,'test.csv'),
                            training_image_path=args.training_image_path,
                            transform=NormalizeImageDict(['image_A','image_B']),
                            random_sample=args.random_sample)

dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)

pair_generation_tnf = SynthPairTnf(geometric_model=args.geometric_model,use_cuda=use_cuda)

# Optimizer
optimizer = optim.Adam(model.FeatureRegression.parameters(), lr=args.lr)

scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

# Train
if args.use_mse_loss:
    checkpoint_name = os.path.join(args.trained_models_dir,
                                   args.trained_models_fn + '_' + args.geometric_model + '_mse_loss' + args.feature_extraction_cnn + '.pth.tar')

else:
    checkpoint_name = os.path.join(args.trained_models_dir,
                                   args.trained_models_fn + '_' + args.geometric_model  + '_' + args.feature_extraction_cnn + '.pth.tar')

best_test_loss = float("inf")

print('Starting training...')

epochArray = np.zeros(args.num_epochs)
trainLossArray = np.zeros(args.num_epochs)
testLossArray = np.zeros(args.num_epochs)


for epoch in range(1, args.num_epochs+1):
    train_loss = train(epoch,model,loss,optimizer,dataloader,pair_generation_tnf,log_interval=10)
    test_loss = test(model,loss,dataloader_test,pair_generation_tnf,use_cuda=use_cuda, geometric_model=args.geometric_model)

    scheduler.step()
    
    epochArray[epoch-1] = epoch
    trainLossArray[epoch-1] = train_loss
    testLossArray[epoch-1] = test_loss

    # remember best loss
    is_best = test_loss < best_test_loss
    best_test_loss = min(test_loss, best_test_loss)
    save_checkpoint({
      'epoch': epoch + 1,
      'args': args,
      'state_dict': model.state_dict(),
      'best_test_loss': best_test_loss,
      'optimizer' : optimizer.state_dict(),
    }, is_best,checkpoint_name)
print('Done!')

if args.use_mse_loss:
    np.savetxt(os.path.join(args.trained_models_dir,
                                   args.trained_models_fn + '_' + args.geometric_model + '_mse_loss' + args.feature_extraction_cnn + '.csv'), np.transpose((epochArray, trainLossArray, testLossArray)), delimiter=',')
else:
    np.savetxt(os.path.join(args.trained_models_dir,
                                   args.trained_models_fn + '_' + args.geometric_model + args.feature_extraction_cnn + '.csv'), np.transpose((epochArray, trainLossArray, testLossArray)), delimiter=',')
