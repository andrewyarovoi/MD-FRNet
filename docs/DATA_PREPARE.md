# Data Preparation

## SemanticKITTI

### Download
Download the SemanticKITTI dataset from: https://semantic-kitti.org/dataset.html#download. Unzip and organize the files
into the following structure:

```
FRNet
├── data
│   ├── semantickitti
│   │   ├── sequences
│   │   │   ├── 00
│   │   │   │   ├── labels
│   │   │   │   ├── velodyne
│   │   │   ├── 01
│   │   │   ├── ..
│   │   │   ├── 21
```

### Semantic KITTI

We will also need to create `.pkl` info files for SemanticKITTI following the instruction of MMDetection3D. Alternatively, we also support a simplified version by running:

```bash
python tools/create_semantickitti.py --root-path ${PATH_TO_SEMANTICKITTI}
```

## Waymo Open Dataset

See [CONVERT_WAYMO.md](../waymo_converter/CONVERT_WAYMO.md) for instructions for converting the waymo dataset into the SemanticKITTI dataset format.

## Target Dataset

Unfortunately, we cannot provide the target dataset used in the paper (due to lack of ownership). Thus, you will need your own dataset. We used an Ouster OS1-64 LiDAR, but any LiDAR should work. You will need to convert your LiDAR point clouds to the SemanticKITTI dataset format. If you can convert your point clouds to pcd files, with a field for index labels, you can adapt our [pcd_to_kitti_format.py](..\tools\target_converter\pcd_to_kitti_format.py) to convert the point clouds to numpy files, and [create_info_file.py](..\tools\target_converter\create_info_file.py) to generate the info files. Note, your pcd cloud will need to store the labels under the field name "label" for the script to work.