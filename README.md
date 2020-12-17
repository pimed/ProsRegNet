# ProsRegNet: A Deep Learning Framework for Registration of MRI and Histopathology Images of the Prostate.

![](pictures/pipeline.png)

This is the PyTorch implementation of the following paper:

Shao, Wei, et al. "ProsRegNet: A Deep Learning Framework for Registration of MRI and Histopathology Images of the Prostate."  [[Accepted to Medical Image Analysis (MedIA)](https://arxiv.org/pdf/2012.00991)]


### Introduction
Our source code has been modified from [cnngeometric_pytorch](https://github.com/ignacio-rocco/cnngeometric_pytorch), and have been tested successfully on Linux Mint, Cuda 10.0, RTX 2080 Ti, Anaconda Python 3.7, PyTorch 1.3.0.

The code is only for research purposes.

### Dependencies
PyTorch 1.3.0
Cuda 10.0
Anaconda Python 3.7
SimpleITK
cv2
skimage

### Usage
1. Clone the repository:
```
git clone https://github.com/pimed/ProsRegNet.git
cd ProsRegNet
```
2. Download the [[training dataset](https://drive.google.com/file/d/1W3eV50pDGBKKz1XX6o6Fi7wzgAHZZBlr/view?usp=sharing)]:
```
uzip the compressed folder named "datasets", this folder contains two subfolders: "training" and "testing". 
```
The small training dataset consists MRI and histopathology image slices of 25 subjects from [[The Cancer Imaging Archive PROSTATE-MRI dataset](https://wiki.cancerimagingarchive.net/display/Public/PROSTATE-MRI)]. The small testing dataset consists of one subject from [[The Cancer Imaging Archive Prostate Fused-MRI-Pathology dataset](https://wiki.cancerimagingarchive.net/display/Public/Prostate+Fused-MRI-Pathology)].


3. Training the affine and deformable registration models (optional):
```
python train.py --geometric-model affine
python train.py --geometric-model tps
```

4. Evaluation:
```
run the registration_pipeline.ipynb jupyter notebok
```

### Models trained with larger dataset, see details in our [MedIA paper](https://arxiv.org/pdf/2012.00991)
[[Trained ProsRegNet affine model](https://drive.google.com/file/d/1REqMqNVLHRnFfuqzJIWrqQgctnaauSO1/view?usp=sharing)]
[[Trained ProsRegNet deformable model](https://drive.google.com/file/d/1j1ai3RG6blpE6Zz9fmazoMsTyCQvGR9z/view?usp=sharing)]

### BibTeX

If you use this code, please cite the following papers:

```bibtex
@article{shao2020prosregnet,
  title={ProsRegNet: A Deep Learning Framework for Registration of MRI and Histopathology Images of the Prostate},
  author={Wei Shao and Linda Banh and Christian A. Kunder and Richard E. Fan and Simon J. C. Soerensen and Jeffrey B. Wang and Nikola C. Teslovich and Nikhil Madhuripan and Anugayathri Jawahar and Pejman Ghanouni and James D. Brooks and Geoffrey A. Sonn and Mirabela Rusu},
  journal={Medical Image Analysis},
  year={2020}
}
```

and

```bibtex
@InProceedings{Rocco17,
  author = {Rocco, I. and Arandjelovi\'c, R. and Sivic, J.},
  title  = {Convolutional neural network architecture for geometric matching},
  booktitle = {{Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition}},
  year = {2017},
}
```
