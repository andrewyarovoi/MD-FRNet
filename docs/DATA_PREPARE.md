# Data Preparation

## SemanticKITTI

### Structure

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

We need to create `.pkl` info files for SemanticKITTI following the instruction of MMDetection3D. Meanwhile, we also support a simplified version by running:

```bash
python tools/create_semantickitti.py --root-path ${PATH_TO_SEMANTICKITTI}
```

### Waymo Open Dataset

See [CONVERT_WAYMO.md](../waymo_converter/CONVERT_WAYMO.md) for instructions for converting the waymo dataset into the semantic kitti format.