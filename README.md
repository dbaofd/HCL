# Hierarchical Context Learning of Object Components for Unsupervised Semantic Segmentation
[Dong Bao](https://scholar.google.com/citations?user=ZRZYhssAAAAJ&hl=en&oi=ao),
[Jun Zhou](https://scholar.google.com/citations?user=6hOOxw0AAAAJ&hl=en&oi=ao),
[Gervase Tuxworth](https://scholar.google.com/citations?user=gKB12I4AAAAJ&hl=en),
[Jue Zhang](https://scholar.google.com/citations?user=K5sULxUAAAAJ&hl=en&oi=ao),
[Yongsheng Gao](https://scholar.google.com/citations?user=IqazXu4AAAAJ&hl=en)
![profile](/imgs/Figure_1.png)

## :book: Contents
<!--ts-->
   * [Installation](#installation)
   * [Training](#training)
   * [Evaluation](#evaluation)
   * [Understanding HCL](#understanding-hcl)
      * [PM-ViT](#pm-vit)
      * [HCL Architecture](#hcl-architecture)
      * [Evaluation Results](#evaluation-results)
      * [Learned Object Component Visualization](#learned-object-component-visualization)
<!--te-->

## :zap: Installation

## :running: Training

## :koala: Evaluation

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

### Learned Object Component Visualization
Object component representation visualization on the PVOC dataset using PM-ViT-S/16. The locations with a cross on the image are the query tokens, e.g., there is a cross on
the bus wheel in the top left image. The query token is assigned a cluster ID from Cfg or Cbg,
then other tokens with the same cluster ID from other images are visualized and presented
on the right side of the query images. There are eight query tokens with different cluster IDs
included: 1) left 1: bus wheel; 2) left 2: car glass; 3) left 3: car wheel; 4) left 4: human upper
face; 5) right 1: human mouth and jaw; 6) right 2: human hand; 7) right 3: cat ear; 8) right
4: dog nose and mouth.
![arch](/imgs/Figure_6.png)