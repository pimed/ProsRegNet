# ProsRegNet: A Deep Learning Framework for Registration of MRI and Histopathology Images of the Prostate.

![](pictures/pipeline.png)

This is the PyTorch implementation of the following paper:

Shao, Wei, et al. "ProsRegNet: A Deep Learning Framework for Registration of MRI and Histopathology Images of the Prostate."  [[Accepted to Medical Image Analysis](https://arxiv.org/pdf/2012.00991)]


### Introduction
Our source code has been modified from [cnngeometric_pytorch](https://github.com/ignacio-rocco/cnngeometric_pytorch), and have been tested successfully on Linux Mint, Cuda 10.0, RTX 2080 Ti, Anaconda Python 3.7, PyTorch 1.3.0.

The code is only for research purposes.

### Usage
1. Clone the repository:
```
git clone https://github.com/pimed/ProsRegNet.git
```
2. Download the training dataset:
```
To be added
```

3. Training the affine and deformable registration models (optional):
```
python train.py --geometric-model affine
python train.py --geometric-model tps
```

4. Evaluation:
```
run the registration_pipeline.ipynb jupyter notebok
```

### Models trained in our [MedIA paper](https://arxiv.org/pdf/2012.00991)
[[Trained ProsRegNet affine model](http://pimed-synology1.stanford.edu:5000/sharing/78V4Qp6ZS)]
[[Trained ProsRegNet deformable model](http://pimed-synology1.stanford.edu:5000/sharing/GCMIX0IHG)]

### BibTeX

If you use this code, please cite the following papers:

```bibtex
@article{shao2020prosregnet,
  title={ProsRegNet: A Deep Learning Framework for Registration of MRI and Histopathology Images of the Prostate},
  author={Shao, Wei and Banh, Linda and Kunder, Christian A and Fan, Richard E and Soerensen, Simon JC and Wang, Jeffrey B and Teslovich, Nikola C and Madhuripan, Nikhil and Jawahar, Anugayathri and Ghanouni, Pejman and others},
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
