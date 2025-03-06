# Installation

These are modified instructions for creating a compatible conda environment that uses a more recent version of pytorch and cuda (more compatible with RTX 4090). 

## Create Conda Environment

Install anaconda and run the following in a terminal:
```
conda create --name frnet -c pytorch python==3.8.18 pytorch==1.10.0 cudatoolkit=11.3 torchvision -y
```

Then activate the new environment:
```
conda activate frnet
```

## Install Pip and Mim Dependencies
Inside the conda environment:
```
pip install -U openmim
mim install mmengine mmcv==2.1 mmdet==3.2 mmdet3d
git clone https://github.com/open-mmlab/mmdetection3d
cd mmdetection3d
pip install -v -e .
conda install -c pyg pytorch-scatter
pip install nuscenes-devkit
```

## Setup Semantic KITTI Dataset and Train Example
In the conda environment, run:
```
python tools/create_semantickitti.py
```

Then run:
```
python train.py configs/frnet/frnet-semantickitti_seg.py
```