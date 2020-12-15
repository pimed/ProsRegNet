import json
import numpy as np
import cv2 
import SimpleITK as sitk 
from collections import OrderedDict
from geotnf.transformation import GeometricTnf
import torch 
from image.normalization import NormalizeImageDict, normalize_image
import os 
from matplotlib import pyplot as plt


def transformAndSaveRegion(preprocess_moving_dest, case, slice, s, region, theta, dH, dW, h, w,x,y,x_offset,y_offset): 
    rotated = np.zeros((w + 2*x_offset, h + 2*y_offset, 3))   
    try:
        path = s['regions'][region]['filename']
        ann = cv2.imread(path) #annotation
        # if flip is 1, flip image horizontally
        try: 
            if s['transform']['flip'] == 1: 
                ann = cv2.flip(ann, 1)
        except: 
            pass 
        
        ann = np.pad(ann,((ann.shape[0],ann.shape[0]),(ann.shape[1],ann.shape[1]),(0,0)),'constant', constant_values=0)
        
        rows, cols, channels = ann.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
        rotated_ann = cv2.warpAffine(ann,M,(cols,rows))

        
        # find edge and downsample
        ann = cv2.resize(rotated_ann, (dH, dW), interpolation=cv2.INTER_CUBIC)
        ann[ann > 5] = 1

        # set edge to outline 
        region3d= np.zeros((w + 2*x_offset, h + 2*y_offset, 3))

        region3d[x_offset:w + x_offset, y_offset:h + y_offset,:] = (ann[x:x+w,y:y+h]>0)*255
        
        rotated = region3d
    except: 
        pass
    
    try: 
        os.mkdir(preprocess_moving_dest + case)
    except: 
        pass 
    
    outputPath = preprocess_moving_dest + case + '\\' + region + '_' + case + '_' + slice +'.png'
    cv2.imwrite(outputPath, rotated)


# preprocess_hist into hist slices here
def preprocess_hist(moving_dict, pre_process_moving_dest, case): 
    for slice in moving_dict:
        s = moving_dict[slice]
        
        # Read image
        img = cv2.imread(s['filename'], )

        # multiply by mask
        prosPath = s['regions']['region00']['filename']
        region00 = cv2.imread(prosPath)
        img = img*(region00/255)
        # if flip is 1, flip image horizontally
        try: 
            if s['transform']['flip'] == 1: 
                img = cv2.flip(img, 1)
                region00 = cv2.flip(region00, 1)
        except: 
            pass
        # rotate image
        try: 
            theta = -s['transform']['rotation_angle']
        except: 
            theta = 0
        
        
        img = np.pad(img,((img.shape[0],img.shape[0]),(img.shape[1],img.shape[1]),(0,0)),'constant', constant_values=0)
        region00 = np.pad(region00,((region00.shape[0],region00.shape[0]),(region00.shape[1],region00.shape[1]),(0,0)),'constant', constant_values=0)
        
        
        rows, cols, channels = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
        rotated_hist = cv2.warpAffine(img,M,(cols,rows),borderValue = (0,0,0) )
        rotated_region00 = cv2.warpAffine(region00,M,(cols,rows))
        

        dH = int(rotated_hist.shape[1]/4)
        dW = int(rotated_hist.shape[0]/4)

        # downsample image, this has to be consistent with the size of MRI
      #  dSize = rotated_hist.shape[0]/720 ; # downsample size
      #  dH = int(rotated_hist.shape[1]/dSize) #downsampled height
      #  dW = int(rotated_hist.shape[0]/dSize) #downsampled width
      #  dH = 3000
      #  dW = 3000
      #  imgResize = cv2.resize(rotated_hist, (dH, dW), interpolation=cv2.INTER_CUBIC)
        
        
        rotated_hist = cv2.resize(rotated_hist, (dH, dW), interpolation=cv2.INTER_CUBIC)
        rotated_region00 = cv2.resize(rotated_region00, (dH, dW), interpolation=cv2.INTER_CUBIC)
        

        # create a bounding box around slice
        points = np.argwhere(rotated_region00[:,:,0] != 0)
        points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
        y, x, h, w = cv2.boundingRect(points) # create a rectangle around those points
        
        crop = rotated_hist[x:x+w, y:y+h,:]
        
        if h>w:
            y_offset = int(h*0.15)
            x_offset = int((h - w + 2*y_offset)/2)
        else:
            y_offset = int(h*0.2)
            x_offset = int((h - w + 2*y_offset)/2)
            
        transformAndSaveRegion(pre_process_moving_dest, case, slice, s, 'region00', theta, dH, dW, h, w,x,y,x_offset,y_offset)
        transformAndSaveRegion(pre_process_moving_dest, case, slice, s, 'region01', theta, dH, dW, h, w,x,y,x_offset,y_offset)
        transformAndSaveRegion(pre_process_moving_dest, case, slice, s, 'region10', theta, dH, dW, h, w,x,y,x_offset,y_offset)
        transformAndSaveRegion(pre_process_moving_dest, case, slice, s, 'region09', theta, dH, dW, h, w,x,y,x_offset,y_offset)
        
        # pad image
        h = h + 2*y_offset
        w = w + 2*x_offset
        

  
        padHist = np.zeros((w, h, 3)) 
      
        padHist[x_offset:crop.shape[0]+x_offset, y_offset:crop.shape[1]+y_offset, :] = crop

        # Write images, with new filename
        cv2.imwrite(pre_process_moving_dest + case + '\\hist_' + case + '_' + slice +'.png', padHist)

#preprocess mri mha files to slices here
def preprocess_mri(fixed_img_mha, fixed_seg, pre_process_fixed_dest, coord, case):     
    imMri = sitk.ReadImage(fixed_img_mha)
    imMri = sitk.GetArrayFromImage(imMri)
    imMriMask = sitk.ReadImage(fixed_seg)
    #### resample mri mask to be the same size as mri
    if imMri.shape[1]!=sitk.GetArrayFromImage(imMriMask).shape[1] | imMri.shape[2]!=sitk.GetArrayFromImage(imMriMask).shape[2]:
        mri_ori = sitk.ReadImage(fixed_img_mha)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(mri_ori)
        imMriMask = resampler.Execute(imMriMask)
        print("input mri and mri mask have different sizes")
    
    imMriMask = sitk.GetArrayFromImage(imMriMask)
    
    coord[case] = {}
    coord[case]['x_offset'] = []
    coord[case]['y_offset'] = []
    coord[case]['x'] = []
    coord[case]['y'] = []
    coord[case]['h'] = []
    coord[case]['w'] = []
    coord[case]['slice']  = []
    
    for slice in range(imMri.shape[0]):
        if np.sum(np.ndarray.flatten(imMriMask[slice, :, :])) == 0: 
            continue
        
        mri = imMri[slice, :, :]*imMriMask[slice, :, :]
        
        mri_mask = imMriMask[slice, :, :] * 255
        
        # create a bounding box around slice
        points = np.argwhere(mri_mask != 0)
        points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
        y, x, h, w = cv2.boundingRect(points) # create a rectangle around those points
        
        

        imMri[slice, :, :] = imMri[slice, :, :] / int(np.max(imMri[slice, :, :]) / 255)
   
        if h>w:
            y_offset = int(h*0.15)
            x_offset = int((h - w + 2*y_offset)/2)
        else:
            y_offset = int(h*0.2)
            x_offset = int((h - w + 2*y_offset)/2)
        
        coord[case]['x'].append(x)
        coord[case]['y'].append(y)
        coord[case]['h'].append(h)
        coord[case]['w'].append(w)
        coord[case]['slice'].append(slice) 
        coord[case]['x_offset'].append(x_offset)
        coord[case]['y_offset'].append(y_offset)  
        
        crop = mri[x - x_offset:x+w+x_offset, y - y_offset:y+h +y_offset]
        
        h = h + 2*y_offset
        w = w + 2*x_offset
        
        crop = crop*25.5/(np.max(crop)/10)
        
        # upsample slice to approx 500 px in width
        ups = 1; 
        upsHeight = int(h*ups)
        upsWidth = int(w*ups)
        
        upsMri = cv2.resize(crop.astype('float32'), (upsHeight,  upsWidth), interpolation=cv2.INTER_CUBIC)
        
                 # save x, y, x_offset, y_offset, h, w for each slice in dictionary 'coord' (coordinates)
        
        try: 
            os.mkdir(pre_process_fixed_dest + case)
        except: 
            pass 
        
        # write to a file        
        cv2.imwrite(pre_process_fixed_dest + case + '\\mri_' + case + '_' + str(slice).zfill(2) +'.jpg', upsMri)

        
        cv2.imwrite(pre_process_fixed_dest + case + '\\mriUncropped_' + case + '_' + str(slice).zfill(2) +'.jpg', imMri[slice, :, :])
        cv2.imwrite(pre_process_fixed_dest + case + '\\mriMask_' + case + '_' + str(slice).zfill(2) +'.jpg', np.uint8(mri_mask))

    coord = OrderedDict(coord)
    
    return coord