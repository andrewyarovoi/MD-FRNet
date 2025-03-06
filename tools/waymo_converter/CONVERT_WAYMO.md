# Converting Waymo to Semantic Kitti Format

## Setup
Create a conda environment:
```
conda create -n waymo-kitti python=3.10 tensorflow=2.12 -c conda-forge
conda activate waymo-kitti
```

Install Waymo's dataset loader:
```
pip install waymo-open-dataset-tf-2-12-0
```

## Usage

First, download the 1.4 Waymo datasets (training, validation, and  (optionally) testing) from [here](https://waymo.com/open/download/).

Place them in the following folder structure:
```
waymo
|--waymo_format
|  |--training
|  |--validation
|  |--testing
|--kitti_format
```

Then navigate into FRNet\waymo_converter and run the python script:
```
cd FRNet/waymo_converter
python waymo_to_sem_kitti.py /path/to/waymo/waymo_format/ /path/to/waymo/kitti_format/ --num_proc 16
```

Make sure to replace `/path/to` to with actual full path.

The script will take a while, but at the end, you should have all the waymo files converted to semantic Kitti format:

```
waymo
|--waymo_format
|  |--training
|  |--validation
|  |--testing
|--kitti_format
|  |--sequences
|  |  |--0000
|  |  |--0001
|  |  |--etc.
|  |--waymo_infos_train.pkl
|  |--waymo_infos_trainval.pkl
|  |--waymo_infos_val.pkl
```