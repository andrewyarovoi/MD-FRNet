# Multi-Dataset Frustrum Range Net (MD-FRNet)

This repo is an adaptation of the [FRNet](https://github.com/Xiangxu-0103/FRNet) repo, integrating [Point Prompt Training](https://arxiv.org/abs/2308.09718) (PPT), [Manifold Mixup](https://arxiv.org/abs/1806.05236) (MM), and ambient values into the FRNet model. This repo also enables multi-dataset pre-training and single-dataset model finetuning. This work primarily served to satisfy a project for CS 8751 at Georgia Institute of Technology. 

Please see our ArXiv publication for more details regarding the implementation:
[https://arxiv.org/abs/2508.20135](https://arxiv.org/abs/2508.20135)

## Setup
To use this code base:
1. Install all dependencies by following the [INSTALL.md](docs\INSTALL.md).
2. Follow [DATA_PREPARE.md](docs\DATA_PREPARE.md) to setup the SemanticKITTI and Waymo Open Dataset. Optionally setup your Target dataset as well.
3. Modify the config files to match your datasets. You may need to adjust the dataset FOVs set in [frnet-mixed_seg.py](configs\frnet\frnet-mixed_seg.py) and will need to adjust the fov and the dataloaders set in [mixed_seg.py](configs\_base_\datasets\mixed_seg.py) for your datasets. You can comment out datasets in the train_dataloader["dataset"]["datasets"] and val_dataloader["dataset"]["datasets"] to disable them for a specific run.
4. Follow [GET_STARTED.md](docs\GET_STARTED.md) to run the training and perform testing.

## License

This work is under the [Apache 2.0 license](LICENSE).

## Citation

If you find this work helpful, please kindly consider citing our paper:
```bibtex
@misc{yarovoi2025mdfrnet,
      title={Data-Efficient Point Cloud Semantic Segmentation Pipeline for Unimproved Roads}, 
      author={Andrew Yarovoi and Christopher R. Valenta},
      year={2025},
      eprint={2508.20135},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2508.20135}, 
}
```

You may also cite the original paper for FRNet:

```bibtex
@article{xu2025frnet,
    title = {FRNet: Frustum-Range Networks for Scalable LiDAR Segmentation},
    author = {Xu, Xiang and Kong, Lingdong and Shuai, Hui and Liu, Qingshan},
    journal = {IEEE Transactions on Image Processing},
    year = {2025}
}
```

## Acknowledgements

This work is developed based on the [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) codebase.

> <img src="https://github.com/open-mmlab/mmdetection3d/blob/main/resources/mmdet3d-logo.png" width="30%"/><br>
> MMDetection3D is an open-source object detection toolbox based on PyTorch, towards the next-generation platform for general 3D perception. It is a part of the OpenMMLab project developed by MMLab.

We acknowledge the use of the following public resources during the course of this work: <sup>1</sup>[SemanticKITTI](http://www.semantic-kitti.org), <sup>2</sup>[SemanticKITTI-API](https://github.com/PRBonn/semantic-kitti-api), <sup>3</sup>[Waymo Open Dataset](https://waymo.com/open/), <sup>4</sup>[waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset), <sup>5</sup>[FRNet](https://github.com/Xiangxu-0103/FRNet).
