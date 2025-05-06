# Hierarchical Context Learning of Object Components for Unsupervised Semantic Segmentation
[Dong Bao](https://scholar.google.com/citations?user=ZRZYhssAAAAJ&hl=en&oi=ao),
[Jun Zhou](https://scholar.google.com/citations?user=6hOOxw0AAAAJ&hl=en&oi=ao),
[Gervase Tuxworth](https://scholar.google.com/citations?user=gKB12I4AAAAJ&hl=en),
[Jue Zhang](https://scholar.google.com/citations?user=K5sULxUAAAAJ&hl=en&oi=ao),
[Yongsheng Gao](https://scholar.google.com/citations?user=IqazXu4AAAAJ&hl=en)

[[Paper]](https://www.sciencedirect.com/science/article/pii/S0031320325003735/pdfft?md5=e66df299a16c22511c4226eb0f984d76&pid=1-s2.0-S0031320325003735-main.pdf)

[![PWC](https://img.shields.io/badge/Ranked_%231-Unsupervised_Semantic_Segmentation_on_COCO--Stuff_27_(Linear_Classifier_[mIoU]_metric)-0EA7A8?logo=paperswithcode)](https://paperswithcode.com/paper/hierarchical-context-learning-of-object)
[![PWC](https://img.shields.io/badge/Ranked_%231-Unsupervised_Semantic_Segmentation_on_COCO--Stuff_27_(Linear_Classifier_[Accuracy]_metric)-0EA7A8?logo=paperswithcode)](https://paperswithcode.com/paper/hierarchical-context-learning-of-object)
[![PWC](https://img.shields.io/badge/Ranked_%231-Unsupervised_Semantic_Segmentation_on_PASCAL_VOC_2012_val_(Linear_Classifier_[mIoU]_metric)-0EA7A8?logo=paperswithcode)](https://paperswithcode.com/paper/hierarchical-context-learning-of-object)


![profile](/imgs/Figure_1.png)

## :book: Contents
<!--ts-->
   * [Requirements](#zap-requirements)
   * [Datasets](#dog-datasets)
      * [COCO-Stuff](#coco-stuff)
      * [Pascal VOC](#pascal-voc)
   * [Checkpoints](#bread-checkpoints)
   * [Training](#running-training)
   * [Evaluation](#koala-evaluation)
      * [Linear Classifier Evaluation](#linear-classifier-evaluation)
      * [Overclustering Evaluation](#overclustering-evaluation)
   * [Understanding HCL](#understanding-hcl)
      * [PM-ViT](#pm-vit)
      * [HCL Architecture](#hcl-architecture)
      * [Evaluation Results](#evaluation-results)
      * [Unsupervised Foreground Extraction](#unsupervised-foreground-extraction)
      * [Learned Object Component Visualization](#learned-object-component-visualization)
   * [Citation](#citation)
<!--te-->

## :zap: Requirements
Install the following packages.
```commandline
- python >= 3.10
- pytorch >= 2.0
- faiss-gpu >= 1.7.4
- torchvision >= 0.15.2
- torchmetrics >= 1.4.0
- opencv >= 4.6.0
- pydensecrf = 1.0rc3
- scikit-learn >= 1.1.3
- scikit-image >= 0.21.0
- einops >= 0.3.2
```

## :dog: Datasets
Please download the data and follow the structure below. 
### COCO-Stuff
```
dataset root.
└───stuffthingmaps_trainval2017
│   └───train2017
│       │   *.png
│       │   ...
│   └───val2017
│       │   *.png
│       │   ...
└───train2017
│   │   *.jpg
│   │   ...
└───val2017
│   │   *.jpg
│   │   ...
└───Coco164kFull_Stuff_Coarse.txt
└───Coco164kFull_Stuff_Coarse_7.txt
└───cocostuff10k.txt
```

### Pascal VOC
```
dataset root.
└───SegmentationClass
│   │   *.png
│   │   ...
└───SegmentationClassAug # contains segmentation masks from trainaug extension 
│   │   *.png
│   │   ...
└───JPEGImages
│   │   *.jpg
│   │   ...
└───ImageSets
|   └───Segmentation
│       │   train.txt
│       │   trainaug.txt
│       │   val.txt
```

## :bread: Checkpoints
We release the weights on trained HCL. The backbone of HCL is PM-ViT, which is fixed during the model training.
For PM-ViT-S/16, we load <a href="https://github.com/facebookresearch/dino?tab=readme-ov-file">Dino-pretrained</a> 
ViT weights "dino_deitsmall16_pretrain.pth" (you can download either from the link in the table below or through the Dino git repo).
For PM-ViT-S/8, we load Dino-pretrained ViT weights "dino_deitsmall8_pretrain.pth". Seghead and linear classifier weights are provided.

<table style="margin: auto">
  <tr>
    <th>Dataset</th>
    <th>Backbone</th>
    <th>Pretrained ViT</th>
    <th>Seghead</th>
    <th>Linear Classifier</th>
  </tr>
  <tr>
    <td align="center">PVOC</td>
    <td align="center">PM-ViT-S/16</td>
    <td align="center"><a href="https://drive.google.com/file/d/1qa8R59ksBTpzD6yA4zrV1pyazVY4A-g0/view?usp=sharing">link</a></td>
    <td align="center"><a href="https://drive.google.com/file/d/12hQ8uRFIKYzoVw_Un-hAWPj8CZrkedhh/view?usp=sharing">link</a></td>
    <td align="center"><a href="https://drive.google.com/file/d/14Va-uzVUWbjk8fDqNraOAfS9NtKzcaHg/view?usp=sharing">link</a></td>
  </tr>
  <tr>
    <td align="center">PVOC</td>
    <td align="center">PM-ViT-S/8</td>
    <td align="center"><a href="https://drive.google.com/file/d/12ULYfr7u3aa1RPElVLHE2fY0V1SCXBvW/view?usp=sharing">link</a></td>
    <td align="center"><a href="https://drive.google.com/file/d/1p5Brn2MzoyZVlH-T9dBL8K26LCCA0Rdf/view?usp=sharing">link</a></td>
    <td align="center"><a href="https://drive.google.com/file/d/1f6U9jmP7PkC8fJ4fQ6gKQc36xDCCnBS2/view?usp=sharing">link</a></td>
  </tr>
  <tr>
    <td align="center">COCO-Stuff</td>
    <td align="center">PM-ViT-S/16</td>
    <td align="center"></td>
    <td align="center"><a href="https://drive.google.com/file/d/1jdLVOB02mRf5iEE6mZit56_LeY6T_K6R/view?usp=sharing">link</a></td>
    <td align="center"><a href="https://drive.google.com/file/d/1rNloiIYBbAz84aaRT6jg5M28DTCgAzHM/view?usp=sharing">link</a></td>
  </tr>
  <tr>
    <td align="center">COCO-Stuff</td>
    <td align="center">PM-ViT-S/8</td>
    <td align="center"></td>
    <td align="center"><a href="https://drive.google.com/file/d/1nkuAacHXcUBEMZhLrTC5U2ulfhhl3TJn/view?usp=sharing">link</a></td>
    <td align="center"><a href="https://drive.google.com/file/d/1Nvvc2akwkOSXbatAC-Z6K-N30rzvRQfe/view?usp=sharing">link</a></td>
  </tr>
</table>

Create a folder "weights" in the root folder with following structure:
```
weights
|── linear_classifier_weights
|── pretrain
└── seghead_weights
```
Then download these check points. Put Dino-pretrained weights to "pretrain" folder, 
put Seghead weights to "seghead_weights" folder, and put linear classifier weights to "linear_classifier_weights" folder.


## :running: Training
To train HCL, go to main_hcl.py, change the corresponding hyperparameters.
Then please run:

```shell script
python main_hcl.py --epochs 10 --batch-size 64 --dist-url 'tcp://0.0.0.0:10001' --multiprocessing-distributed --world-size 1 --rank 0
```
## :koala: Evaluation
### Linear Classifier Evaluation
To evaluate linear classifier, go to "linear_eval.py", select a configuration from "eval_config", and then modify "selected_config". In the end, please run:
```shell script
python linear_eval.py --batch-size 16 --gpu 0
```

### Overclustering Evaluation
To evaluate linear classifier, go to "overclustering_eval.py", select a configuration from "eval_config", and then modify "selected_config". In the end, please run:
```shell script
python overclustering_eval.py --batch-size 16 --gpu 0
```

## Understanding HCL
### PM-ViT
Parallel Multilevel Vision Transformer (PM-ViT), a specially designed backbone that captures multi-level object granularities and aggregates hierarchical contextual information into unified object component tokens.
![pm-vit](/imgs/Figure_3.png)

### HCL Architecture
Hierarchical Context Learning (HCL) of object components for USS, which focuses on learning discriminative spatial token embeddings by enhancing semantic consistency 
through hierarchical context. At the core of HCL is PM-ViT, a specially designed backbone that integrates multi-level hierarchical contextual information into unified token 
representations. To uncover the intrinsic semantic structures of objects, we introduce Momentum-based Global Foreground-Background Clustering (MoGoClustering). Leveraging DINO’s
foreground extraction capability, MoGoClustering clusters foreground and background object components into coherent semantic groups. It initializes cluster centroids and iteratively 
refines them during the optimization process to achieve robust semantic grouping. Furthermore, coupled with a dense prediction loss, we design a Foreground-Background-Aware (FBA) 
contrastive loss based on MoGoClustering to ensure that the learned dense representations are compact and consistent across views.
![arch](/imgs/Figure_2.png)

### Evaluation Results
We evaluate the HCL on the PVOC and COCO-Stuff datasets.
![arch](/imgs/Figure_4.png)

### Unsupervised Foreground Extraction
We compare the unsupervised foreground extraction results (in green) of HCL with DINO cls token attention maps (in red). 
![arch](/imgs/fg.png)


### Learned Object Component Visualization
Object component representation visualization on the PVOC dataset using PM-ViT-S/16. The locations with a cross on the image are the query tokens, e.g., there is a cross on
the bus wheel in the top left image. The query token is assigned a cluster ID from Cfg or Cbg,
then other tokens with the same cluster ID from other images are visualized and presented
on the right side of the query images. There are eight query tokens with different cluster IDs
included: 1) left 1: bus wheel; 2) left 2: car glass; 3) left 3: car wheel; 4) left 4: human upper
face; 5) right 1: human mouth and jaw; 6) right 2: human hand; 7) right 3: cat ear; 8) right
4: dog nose and mouth.
![arch](/imgs/Figure_6.png)

## Citation
If you believe this project is useful, please consider starring or citing, cheers.
```commandline
@article{bao2025hierarchical,
  title={Hierarchical Context Learning of object components for unsupervised semantic segmentation},
  author={Bao, Dong and Zhou, Jun and Tuxworth, Gervase and Zhang, Jue and Gao, Yongsheng},
  journal={Pattern Recognition},
  pages={111713},
  year={2025},
  publisher={Elsevier}
}
```