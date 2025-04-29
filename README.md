<h1 align="center">ICG-MVSNet: Learning Intra-view and Cross-view Relationships for Guidance in Multi-View Stereo (ICME 2025)</h1>

<div align="center">
    <a href="https://yuhsihu.github.io" target='_blank'>Yuxi Hu</a>, 
    <a href="https://halajun.github.io/" target='_blank'>Jun Zhang</a>,  
    <a href="https://www.doublez.site" target='_blank'>Zhe Zhang</a>, 
    <a href="https://www.tugraz.at/institute/icg/research/team-fraundorfer/people/rafael-weilharter" target='_blank'>Rafael Weilharter</a>, 
    <a href="https://yuchenrao.github.io/" target='_blank'>Yuchen Rao</a>, 
    <a href="https://easonchen99.github.io/Homepage/" target='_blank'>Kuangyi Chen</a>, 
    <a href="https://scholar.google.com/citations?user=Qf-_DhUAAAAJ&hl=en" target='_blank'>Runze Yuan</a>, 
    <a href="https://www.tugraz.at/institute/icg/research/team-fraundorfer/people/friedrich-fraundorfer/" target='_blank'>Friedrich Fraundorfer</a>*
</div>

<br />

<div align="center">

![Publication](https://img.shields.io/badge/2025-ICME-2978b5)
[![Paper](http://img.shields.io/badge/arxiv-arxiv.2503.21525-B31B1B?logo=arXiv&logoColor=green)](https://arxiv.org/abs/2503.21525)
![Pytorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)

</div>

##  📌 Introduction
This repository contains the official implementation of **ICG-MVSNet: Learning Intra-view and Cross-view Relationships for Guidance in Multi-View Stereo**. 

## 🚀 Pipeline
![Pipeline](assets/pipeline.png)

## 🔧 Setup

### 1.1 Requirements

Use the following commands to build the `conda` environment.

```bash
conda create -n icgmvsnet python=3.10.8
conda activate icgmvsnet
pip install -r requirements.txt
```

### 1.2 Datasets

Download the following datasets and modify the corresponding local path in `scripts/data_path.sh`.

#### DTU Dataset

**Training data**. We use the same DTU training data as mentioned in MVSNet and CasMVSNet, please refer to [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) and [Depth raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) for data download. You should download the [Recitfied raw](http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip) if you want to train the model in raw image resolution. Unzip and organize them as:

```
dtu_training/
├── Cameras
├── Depths
├── Depths_raw
├── Rectified
└── Rectified_raw (optional)
```

**Testing data**. Download [DTU testing data](https://drive.google.com/file/d/135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_/view). Unzip it as:


```
dtu_test/
├── scan1
├── scan4
├── ...
```

#### BlendedMVS Dataset

Download the low image resolution version of [BlendedMVS dataset](https://drive.google.com/file/d/1ilxls-VJNvJnB7IaFj7P0ehMPr7ikRCb/view) and unzip it as:

```
blendedmvs/
└── dataset_low_res
    ├── ...
    └── 5c34529873a8df509ae57b58
```

#### Tanks and Temples Dataset

Download the intermediate and advanced subsets of Tanks and Temples dataset. We use the camera parameters of short depth range version, you can download processed data [here](https://drive.google.com/file/d/17mTgTzjPV1KsazabRIU0J3p0_ogufi5R/view?usp=sharing) and change `cams_1` to `cams`.

```
tanksandtemples/
├── advanced
│   ├── ...
│   └── Temple
│       ├── cams
│       ├── images
│       ├── pair.txt
│       └── Temple.log
└── intermediate
    ├── ...
    └── Train
        ├── cams
        ├── cams_train
        ├── images
        ├── pair.txt
        └── Train.log
```

## 🧠 Training

You can train ICG-MVSNet from scratch on DTU dataset and then fine-tune on BlendedMVS dataset. Please make sure to set the dataset path in `scripts/data_path.sh` before running training or testing.

### 2.1 DTU

To train ICG-MVSNet on DTU dataset, you can refer to `scripts/dtu/train_dtu.sh`, and run:

```bash
bash scripts/dtu/train_dtu.sh exp_name
```

### 2.2 BlendedMVS

To fine-tune the model on BlendedMVS dataset, you can refer to `scripts/blend/train_bld_ft.sh`, and also specify `THISNAME`, `BLD_CKPT_FILE`, and run:

```bash
bash scripts/blend/train_bld_ft.sh expname
```

## 📊 Testing

### 3.1 DTU

For DTU testing, we use model trained on DTU training dataset. You can perform *depth map estimation, point cloud fusion, and result evaluation* according to the following steps.
1. Depth map estimation and point cloud fusion. Run:

```
bash scripts/dtu/test_dtu.sh exp_name
```

2. Download the [ObsMask](http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip) and [Points](http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip) of DTU GT point clouds from the official website and organize them as:

```
evaluation/
    ├── ObsMask
    └── Points
```

3. Result evaluation. Setup `Matlab` in command line mode, and run `bash scripts/dtu/matlab_quan_dtu.sh`. You can adjust the `num_at_once` config according to your machine's CPU and memory ceiling. After quantitative evaluation, you will get `[FUSION_METHOD]_quantitative/` and `[THISNAME].log` just store the quantitative results.

### 3.2 Tanks and Temples

For testing on [Tanks and Temples benchmark](https://www.tanksandtemples.org/leaderboard/), you can use any of the following configurations:
- Only train on DTU training dataset.
- Only train on BlendedMVS dataset.
- Pretrained on DTU training dataset and finetune on BlendedMVS dataset. (Recommend)

After your training, please follow these steps:
1. To generate point cloud results, run:

```
bash scripts/tnt/test_tnt_inter.sh exp_name
```
```
bash scripts/tnt/test_tnt_adv.sh exp_name
``` 

2. Follow the *Upload Instructions* on the [Tanks and Temples official website](https://www.tanksandtemples.org/submit/) to make online submissions.

### 3.3 Custom Data

ICG-MVSNet can also reconstruct on custom data. You can refer to [MVSNet](https://github.com/YoYo000/MVSNet#file-formats) to organize your data, and run:

```
bash scripts/custom/test_custom.sh exp_name
```

## 🎯 Results  
### Qualitative Results  
![Results](assets/dtu-visual.png)  

### Quantitative Results  
Our results on DTU and Tanks and Temples (T&T) Dataset are listed in the tables.

| DTU | Acc. ↓ | Comp. ↓ | Overall ↓ |
| ----------- | ------ | ------- | --------- |
| Ours   | 0.327 | 0.251  | 0.289    |

| T&T (Intermediate) | Mean ↑ | Family | Francis | Horse | Lighthouse | M60   | Panther | Playground | Train |
| ------------------ | ------ | ------ | ------- | ----- | ---------- | ----- | ------- | ---------- | ----- |
| Ours          | 65.53  | 81.73  | 68.92   | 56.59 | 66.10      | 64.86 | 64.41   | 62.33      | 59.26 |

You can download point clouds [here](https://cloud.tugraz.at/index.php/s/bfC3ykYt7BszG8C).
## 🔗 Citation
If you find this work useful in your research, please consider citing the following preprint:
```
@article{hu2025icg,
  title={ICG-MVSNet: Learning Intra-view and Cross-view Relationships for Guidance in Multi-View Stereo},
  author={Hu, Yuxi and Zhang, Jun and Zhang, Zhe and Weilharter, Rafael and Rao, Yuchen and Chen, Kuangyi and Yuan, Runze and Fraundorfer, Friedrich},
  journal={arXiv preprint arXiv:2503.21525},
  year={2025}
}
```

## ❤️ Acknowledgements
This repository builds upon the great work of the following projects:
- [MVSNet](https://github.com/YoYo000/MVSNet)
- [MVSNet-pytorch](https://github.com/xy-guo/MVSNet_pytorch)
- [CVP-MVSNet](https://github.com/JiayuYANG/CVP-MVSNet)
- [CasMVSNet](https://github.com/alibaba/cascade-stereo)
- [MVSTER](https://github.com/JeffWang987/MVSTER)
- [GeoMVSNet](https://github.com/doubleZ0108/GeoMVSNet)
- [ET-MVSNet](https://github.com/TQTQliu/ET-MVSNet).

We sincerely thank the authors for their contributions to the MVS community.