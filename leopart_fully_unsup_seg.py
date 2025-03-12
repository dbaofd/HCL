# Adopted from https://github.com/MkuuWaUjinga/leopart
import click
import optuna
import joblib
import torch
from torch import nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torchvision.transforms import GaussianBlur
from collections import defaultdict
from torch.utils.data import DataLoader
from skimage.feature import graycomatrix
from skimage.measure import label
from typing import List, Tuple
from infomap import Infomap

from utils.eval_utils import PredsmIoU, cluster, normalize_and_transform
from data import eval_transforms
from data.data_module import EvalPascal
import build
def process_attentions(attentions: torch.Tensor, spatial_res: int, threshold: float = 0.5, blur_sigma: float = 0.6) \
        -> torch.Tensor:
    """
    Process [0,1] attentions to binary 0-1 mask. Applies a Guassian filter, keeps threshold % of mass and removes
    components smaller than 3 pixels.
    The code is adapted from https://github.com/facebookresearch/dino/blob/main/visualize_attention.py but removes the
    need for using ground-truth data to find the best performing head. Instead we simply average all head's attentions
    so that we can use the foreground mask during training time.
    :param attentions: torch 4D-Tensor containing the averaged attentions
    :param spatial_res: spatial resolution of the attention map
    :param threshold: the percentage of mass to keep as foreground.
    :param blur_sigma: standard deviation to be used for creating kernel to perform blurring.
    :return: the foreground mask obtained from the ViT's attention.
    """
    # Blur attentions
    attentions = GaussianBlur(7, sigma=(blur_sigma))(attentions)
    attentions = attentions.reshape(attentions.size(0), 1, spatial_res ** 2)
    # Keep threshold% of mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=-1, keepdim=True)
    cumval = torch.cumsum(val, dim=-1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    th_attn[:, 0] = torch.gather(th_attn[:, 0], dim=1, index=idx2[:, 0])
    th_attn = th_attn.reshape(attentions.size(0), 1, spatial_res, spatial_res).float()
    # Remove components with less than 3 pixels
    for j, th_att in enumerate(th_attn):
        labelled = label(th_att.cpu().numpy())
        for k in range(1, np.max(labelled) + 1):
            mask = labelled == k# 1 x 28 x 28
            if np.sum(mask) <= 2:
                th_attn[j, 0][mask[0]] = 0# Mistake, corrected, was [mask]
    return th_attn.detach()
def eval_jac(gt: torch.Tensor, pred_mask: torch.Tensor, with_boundary: bool = True) -> float:
    """
    Calculate Intersection over Union averaged over all pictures. with_boundary flag, if set, doesn't filter out the
    boundary class as background.
    """
    jacs = 0
    for k, mask in enumerate(gt):
        if with_boundary:
            gt_fg_mask = (mask != 0).float()
        else:
            gt_fg_mask = ((mask != 0) & (mask != 255)).float()
        intersection = gt_fg_mask * pred_mask[k]
        intersection = torch.sum(torch.sum(intersection, dim=-1), dim=-1)
        union = (gt_fg_mask + pred_mask[k]) > 0
        union = torch.sum(torch.sum(union, dim=-1), dim=-1)
        jacs += intersection / union
    res = jacs / gt.size(0)
    print(res)
    return res.item()

def process_and_store_attentions(attns: List[torch.Tensor], threshold: float, spatial_res: int, split: str,
                                 experiment_folder: str):
    # Concat and average attentions over all heads.
    attns_processed = torch.cat(attns, dim = 0)
    attns_processed = sum(attns_processed[:, i] * 1 / attns_processed.size(1) for i in range(attns_processed.size(1)))
    attns_processed = attns_processed.reshape(-1, 1, spatial_res, spatial_res)
    # Transform attentions to binary fg mask
    th_attns = process_attentions(attns_processed, spatial_res, threshold=threshold, blur_sigma=0.6)
    torch.save(th_attns, os.path.join(experiment_folder, f'attn_{split}.pt'))


def compute_features(loader: DataLoader, model: nn.Module, device: str, spatial_res: int) -> \
        Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    # Compute ViT model features on all data provided by loader. Also return attentions and gt masks.
    model.to(device)
    feats = []
    all_masks = []
    attns = []
    for i, (imgs, mask) in enumerate(loader):
        if i%2==0:
            print(i)
            print(imgs.shape, mask.shape)
        bs = imgs.size(0)
        #assert torch.max(mask).item() <= 1 and torch.min(mask).item() >= 0
        gt = mask #* 255
        # Get backbone embeddings for batch
        with torch.no_grad():
            embeddings, attn = model.forward_backbone(imgs.to(device), last_self_attention=True)
            #embeddings = embeddings[:, 1:].reshape(bs * spatial_res**2, model.embed_dim)
            embeddings = embeddings.reshape(bs * spatial_res ** 2, 768)
        attns.append(attn.cpu())# attn:15 x 6 x 784
        feats.append(embeddings.cpu())
        all_masks.append(gt.cpu())
    model.cpu()
    return feats, all_masks, attns

def store_and_compute_features(train_loader: DataLoader, val_loader: DataLoader, pca_dim: int, model: nn.Module, device: str,
                               spatial_res: int, experiment_folder: str, gt_save_folder: str = None,
                               save_attn : bool = True):
    train_feats, train_gt, train_attns = compute_features(train_loader,
                                                          model, device, spatial_res)
    print("computed train features", torch.cat(train_feats, dim=0).shape, torch.cat(train_gt, dim=0).shape, torch.cat(train_attns, dim=0).shape)
    val_feats, val_gt, val_attns = compute_features(val_loader,
                                               model, device, spatial_res)
    print("computed val features")

    transformed_feats = normalize_and_transform(torch.cat((
        torch.cat(train_feats, dim=0), torch.cat(val_feats, dim=0)), dim=0), pca_dim)
    transformed_feats = transformed_feats.reshape(len(train_loader.dataset) + len(val_loader.dataset),
                                                  spatial_res**2, pca_dim)
    print(f"Normalized and PCA to {pca_dim} dims with shape {transformed_feats.size()}")

    # Store to disk
    print("Storing to disk")
    os.makedirs(experiment_folder, exist_ok=True)
    torch.save(transformed_feats[:len(train_loader.dataset)], os.path.join(experiment_folder, "all_pascal_train.pt"))
    torch.save(transformed_feats[len(train_loader.dataset):], os.path.join(experiment_folder, "all_pascal_val.pt"))

    if gt_save_folder is not None:
        train_path = os.path.join(gt_save_folder, "all_gt_masks_train_voc12.pt")
        if not os.path.exists(train_path):
            torch.save(torch.cat(train_gt), train_path)
        val_path = os.path.join(gt_save_folder, "all_gt_masks_val_voc12.pt")
        if not os.path.exists(val_path):
            torch.save(torch.cat(val_gt), val_path)

    # postprocess attentions
    if save_attn:
        process_and_store_attentions(train_attns, 0.65, spatial_res, "train", experiment_folder)
        process_and_store_attentions(val_attns, 0.65, spatial_res, "val", experiment_folder)
def get_cluster_precs(cluster: torch.Tensor, mask: torch.Tensor, k: int) -> List[float]:
    # Calculate attention foreground precision for each cluster id.
    # Note this doesn't use any gt but rather takes the ViT attention as noisy ground-truth for foreground.
    assert cluster.size(0) == mask.size(0)
    cluster_id_to_oc_count = defaultdict(int)
    cluster_id_to_cum_jac = defaultdict(float)
    for img_id in range(cluster.size(0)):
        img_attn = mask[img_id].flatten()
        img_clus = cluster[img_id].flatten()
        for cluster_id in torch.unique(img_clus):
            tmp_attn = (img_attn == 1)
            tmp_clust = (img_clus == cluster_id)
            tp = torch.sum(tmp_attn & tmp_clust).item()
            fp = torch.sum(~tmp_attn & tmp_clust).item()
            prec = float(tp) / max(float(tp + fp), 1e-8)  # Calculate precision
            cluster_id_to_oc_count[cluster_id.item()] += 1
            cluster_id_to_cum_jac[cluster_id.item()] += prec
    assert len(cluster_id_to_oc_count.keys()) == k and len(cluster_id_to_cum_jac.keys()) == k
    # Calculate average precision values
    precs = []
    for cluster_id in sorted(cluster_id_to_oc_count.keys()):
        precs.append(cluster_id_to_cum_jac[cluster_id] / cluster_id_to_oc_count[cluster_id])
    return precs


def find_good_threshold(train_clusters: torch.Tensor, train_gt: torch.Tensor, precs: List[float], k: int) -> \
        List[Tuple[float, int, float]]:
    jacs = []
    sorted_precs = np.sort(precs)
    sorted_args = np.argsort(precs)
    for start in range(int(0.55 * k), int(0.75 * k)): # try out cuts between assigning 55% to 75% of clusters to bg
        fg_ids = sorted_args[start:]
        cbfe_mask = torch.zeros_like(train_clusters)
        for i in fg_ids:
            cbfe_mask[train_clusters == i] = 1
        jacs.append((sorted_precs[start], start, eval_jac(train_gt, cbfe_mask, with_boundary=True)))
        print(
            f"for {start} % fg cluster train is {torch.sum(cbfe_mask).item() / cbfe_mask.flatten().size(0)} "
            f"with {sorted_precs[start]}")
    return sorted(jacs, key=lambda x: x[2])  # return sorted by IoU


def cluster_all(transformed_feats: torch.Tensor, seed: int, K: List[int], spatial_res: int, experiment_folder: str,
                pca_dim: int, train_len: int, mask: torch.Tensor = None, interpolate_embeddings: bool = False,
                masks_interpolation_size: int = 100, spherical: bool = False):
    # Create cluster subdir
    os.makedirs(os.path.join(experiment_folder, "clusters"), exist_ok=True)
    if interpolate_embeddings:
        # Interpolate embeddings to masks_interpolation_size instead of interpolating the cluster assignments
        transformed_feats = transformed_feats.reshape(-1, spatial_res, spatial_res, pca_dim).permute(0, 3, 1, 2)
        transformed_feats = F.interpolate(transformed_feats,
                                          mode='nearest',
                                          size=(masks_interpolation_size, masks_interpolation_size))
        transformed_feats = transformed_feats.permute(0, 2, 3, 1).reshape(-1, pca_dim)
        spatial_res = masks_interpolation_size
    if mask is not None:
        # Apply mask to embeddings to get foreground embeddings
        mask = F.interpolate(mask, size=(spatial_res, spatial_res))
        transformed_feats = transformed_feats[mask.flatten().bool()]
    # Cluster with granularities K
    for k in K:
        root_cluster_folder = os.path.join(experiment_folder, "clusters")
        if interpolate_embeddings:
            train_cluster_path = os.path.join(root_cluster_folder, f"clusters_train_{k}_{seed}_interembTrue.pt")
            val_cluster_path = os.path.join(root_cluster_folder, f"clusters_val_{k}_{seed}_interembTrue.pt")
        else:
            train_cluster_path = os.path.join(root_cluster_folder, f"clusters_train_{k}_{seed}.pt")
            val_cluster_path = os.path.join(root_cluster_folder, f"clusters_val_{k}_{seed}.pt")
        if os.path.exists(train_cluster_path):
            print(f"Already computed clusters {k}")
            continue
        clusters = cluster(pca_dim, transformed_feats.numpy(), spatial_res, k=k, seed=seed, mask=mask,
                           spherical=spherical)
        torch.save(clusters[:train_len], train_cluster_path)
        torch.save(clusters[train_len:], val_cluster_path)


def cluster_based_fg_extraction(save_folder: str, overclustering_eval_size: int, experiment_folder: str,
                                k_fg_extraction: int, seed_fg_extraction: int, masks_eval_size: int):
    # Load noisy fg mask and clusters for train data
    noisy_fg_train = torch.load(os.path.join(experiment_folder, "attn_train.pt"))# Mistake, corrected, used to be save_folder, wrong pt name
    noisy_fg_train = nn.functional.interpolate(noisy_fg_train,
                                               size=(overclustering_eval_size, overclustering_eval_size),
                                               mode='nearest')
    train_clusters = torch.load(os.path.join(experiment_folder, "clusters",
                                             f"clusters_train_{k_fg_extraction}_{seed_fg_extraction}.pt"))
    train_clusters = nn.functional.interpolate(train_clusters.float(),
                                               size=(overclustering_eval_size, overclustering_eval_size),
                                               mode='nearest')
    print(torch.unique(noisy_fg_train), noisy_fg_train.shape, train_clusters.shape, "+++++++++_____________")
    # calculate cluster-attn-precisions and find good precision threshold
    train_precs = get_cluster_precs(train_clusters, noisy_fg_train, k_fg_extraction)
    gt = torch.load(os.path.join(save_folder, "all_gt_masks_train_voc12.pt"))# gt shape N x 448 x 448
    gt =gt.unsqueeze(dim = 1)# mistake, I added this line to prepare gt for interpolation. N x 1 x 448 x 448
    gt_interpolate = nn.functional.interpolate(gt.float(),
                                               size=(overclustering_eval_size, overclustering_eval_size),
                                               mode='nearest')
    res = find_good_threshold(train_clusters, gt_interpolate, train_precs, k_fg_extraction)
    # pick precision value of best performing split and round it to nearest 0.05 boundary.
    threshold = min(np.arange(0, 1, 0.05), key=lambda x: abs(x - res[-1][0]))
    print(f"Found threshold {threshold}")

    # Apply threshold to train data and evaluate
    start_idx = np.where((np.sort(train_precs) >= threshold) == True)[0][0]
    fg_ids = np.argsort(train_precs)[start_idx:]
    attn_mask_soft = torch.zeros_like(train_clusters)
    for i in fg_ids:
        attn_mask_soft[train_clusters == i] = 1
    print(f"% fg cluster train is {torch.sum(attn_mask_soft).item() / attn_mask_soft.flatten().size(0)}")
    eval_jac(gt,
             F.interpolate(attn_mask_soft, size=(masks_eval_size, masks_eval_size), mode='nearest'),
             with_boundary=True)
    # Save train mask
    torch.save(attn_mask_soft,
               os.path.join(experiment_folder, f"cluster_saliency_train_{k_fg_extraction}_{seed_fg_extraction}.pt"))

    # Apply threshold to val data
    noisy_fg_val = torch.load(os.path.join(experiment_folder, "attn_val.pt"))# Mistake, corrected
    noisy_fg_val = nn.functional.interpolate(noisy_fg_val, size=(overclustering_eval_size, overclustering_eval_size),
                                             mode='nearest')
    val_clusters = torch.load(os.path.join(experiment_folder, "clusters",
                                           f"clusters_val_{k_fg_extraction}_{seed_fg_extraction}.pt"))
    val_clusters = nn.functional.interpolate(val_clusters.float(),
                                             size=(overclustering_eval_size, overclustering_eval_size),
                                             mode='nearest')
    val_precs = get_cluster_precs(val_clusters, noisy_fg_val, k_fg_extraction)
    start_idx = np.where((np.sort(val_precs) >= threshold) == True)[0][0]
    fg_ids = np.argsort(val_precs)[start_idx:]
    attn_mask_soft = torch.zeros_like(val_clusters)
    for i in fg_ids:
        attn_mask_soft[val_clusters == i] = 1
    print(f"% fg cluster val is {torch.sum(attn_mask_soft).item() / attn_mask_soft.flatten().size(0)}")
    print(attn_mask_soft.size())
    # Save val mask
    torch.save(attn_mask_soft,
               os.path.join(experiment_folder, f"cluster_saliency_val_{k_fg_extraction}_{seed_fg_extraction}.pt"))

    # Evaluate val mask
    gt_val = torch.load(os.path.join(save_folder, 'all_gt_masks_val_voc12.pt'))
    gt_val = gt_val.unsqueeze(dim=1)  # mistake, I added this line to prepare gt for interpolation. N x 1 x 448 x 448
    gt_val_interpolate = nn.functional.interpolate(gt_val.float(), size=(masks_eval_size, masks_eval_size), mode='nearest')
    eval_jac(gt_val_interpolate,
             nn.functional.interpolate(attn_mask_soft, size=(masks_eval_size, masks_eval_size), mode='nearest'),
             with_boundary=True)


def evaluate_clustering(k: int, seed: int, split: str, experiment_folder: str, save_folder: str,
                        interpolate_embeddings: bool = False, masks_eval_size: int = 100, used_mask: bool = False):
    # Load cluster from disk
    cluster_path = os.path.join(experiment_folder, "clusters", f"clusters_{split}_{k}_{seed}.pt")
    if interpolate_embeddings:
        cluster_path = os.path.join(experiment_folder, "clusters", f"clusters_{split}_{k}_{seed}_interembTrue.pt")
    cluster_preds = torch.load(cluster_path)
    cluster_preds = F.interpolate(cluster_preds.float(), size=(masks_eval_size, masks_eval_size), mode='nearest')

    # Load gt
    gt = torch.load(os.path.join(save_folder, f"all_gt_masks_{split}_voc12.pt"))
    gt = nn.functional.interpolate(gt, size=(masks_eval_size, masks_eval_size), mode='nearest')

    # Evaluate clustering
    if used_mask:
        # Clustering was only done on foreground. Thus background class is masked-out background.
        metric = PredsmIoU(k + 1, 21)
    else:
        metric = PredsmIoU(k, 21)
    metric.update(gt[gt != 255], cluster_preds[gt != 255])
    many_to_one = True
    precision_based = True
    if k == 21 or (k + 1 == 21 and used_mask):
        many_to_one = False
        precision_based = False
    return metric.compute(True, many_to_one=many_to_one, precision_based=precision_based)


def create_matrix(clusters: torch.Tensor, num_clusters: int, distances: List[int]):
    # Create co-occurrence matrix based on normalized per image co-occurrences of each cluster
    assert num_clusters <= 256, "skimage's graycomatrix() only works with level up to 256"
    assert torch.max(clusters) <= 256
    co_occurrence_matrix = torch.zeros(num_clusters, num_clusters).int()
    cluster_appearances_per_image = np.zeros(num_clusters)
    for i in list(range(clusters.size(0))):
        clusters_in_image = clusters[i]
        # Calculate co-occurrence counts for clusters in image i
        co_mat = graycomatrix(clusters_in_image.squeeze().numpy(),
                              distances=distances,
                              levels=num_clusters,
                              angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 4 * np.pi / 3, 3 * np.pi / 2,
                                      7 * np.pi / 4])  # Take 8-neighborhood for co-occurrence
        counts = np.sum(co_mat, axis=(2, 3))  # Sum over distances and angles
        np.fill_diagonal(counts, 0)  # Cancel same cluster co-occurrences
        # Normalize over total counts per cluster
        row_sum = np.sum(counts, axis=1)
        repeat_row_sum = np.repeat(row_sum, num_clusters).reshape(num_clusters, num_clusters)
        counts_normalized = np.divide(counts, repeat_row_sum, out=np.zeros_like(counts, dtype=float),
                                      where=repeat_row_sum != 0)
        cluster_appearances_per_image += (row_sum > 0).astype(int)
        co_occurrence_matrix += counts_normalized
    com_row_stochastic = co_occurrence_matrix / np.repeat(cluster_appearances_per_image, num_clusters).reshape(
        num_clusters, num_clusters)
    return com_row_stochastic


def start_unsup_seg(patch_size: int, arch: str, experiment_name: str, batch_size: int, input_size: int,
                    save_folder: str, data_dir: str, pca_dim: int, k_fg_extraction: int,
                    clustering_eval_size: int, evaluate_cbfe: bool, clustering_seed: int,
                    num_objects_pvoc: int, k_community: int, markov_time: float, weight_threshold: float,
                    num_runs: int, compute_upper_bound: bool, split_cd: str, weight_path:str):
    # Derive some important vars
    method = "ours"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    spatial_res = input_size // patch_size
    experiment_folder = os.path.join(save_folder, method, experiment_name)

    # Data loading code
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    # we scale mean value by 255 is because the values of image pixels will be from 0 - 255 after ToTensor.
    # This ToTensor is implemented by us, it won't rescale the value to 0 -1 like a standard ToTensor function.
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    val_transforms = eval_transforms.Compose([
        eval_transforms.Resize((input_size, input_size)),  # (448,448)
        eval_transforms.ToTensor(),
        eval_transforms.Normalize(mean=mean, std=std)])
    # Init data module
    train_dataset = EvalPascal(root_path="/workspace/VOCdevkit/VOC2012/", split="train", transform=val_transforms)
    val_dataset = EvalPascal(root_path="/workspace/VOCdevkit/VOC2012/", split="val", transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True
    )

    # Load model, compute features and store them
    train_features_path = os.path.join(save_folder, method, experiment_name, "all_pascal_train.pt")
    val_features_path = os.path.join(save_folder, method, experiment_name, "all_pascal_val.pt")
    if not os.path.exists(train_features_path):
        # model = vits.__dict__["vit_small"](
        #     patch_size=patch_size,
        #     pretrained=True,
        #     pretrain_weights="leopart",
        #     use_projector=False,
        # )
        model = build.MoGoSegEval(
            vit_arch="vit_small",
            pretrained_path=weight_path,#"mogoseg_vit_epoch9_pascal_15.pth.tar",
            checkpoint_key="online_seg_head",
            patch_size=patch_size,
            embed_dim=384,
            seg_head_type="transformer_block",
            apply_dropout=False
        )

        store_and_compute_features(train_loader, val_loader, pca_dim, model, device, spatial_res, experiment_folder,
                                   gt_save_folder=save_folder, save_attn=True)

    # Cluster features to k_fg_extraction for cluster-based foreground extraction (CBFE)
    root_cluster_folder = os.path.join(experiment_folder, "clusters")
    if not os.path.exists(os.path.join(root_cluster_folder, f"clusters_train_{k_fg_extraction}_{clustering_seed}.pt")):
        print(f"Clustering with granularity {k_fg_extraction}")
        print("Loading embeddings from disk")
        val_emb = torch.load(val_features_path)
        train_emb = torch.load(train_features_path)
        cluster_all(torch.cat((train_emb, val_emb), dim=0).reshape(-1, pca_dim),
                    K=[k_fg_extraction], seed=clustering_seed, spatial_res=spatial_res,
                    experiment_folder=experiment_folder,
                    pca_dim=pca_dim, train_len=len(train_loader.dataset), spherical=True)

    # Run CBFE to get fg masks
    train_fg_path = os.path.join(experiment_folder, f"cluster_saliency_train_{k_fg_extraction}_{clustering_seed}.pt")
    val_fg_path = os.path.join(experiment_folder, f"cluster_saliency_val_{k_fg_extraction}_{clustering_seed}.pt")
    if not os.path.exists(train_fg_path):
        print("#" * 10 + " cluster-based foreground extraction " + "#" * 10)
        cluster_based_fg_extraction(save_folder, clustering_eval_size, experiment_folder, k_fg_extraction,
                                    clustering_seed, masks_eval_size=input_size)
        print("Done for cbfe")

    # Cluster foreground only, using CBFE masks
    if not os.path.exists(
            os.path.join(root_cluster_folder, f"clusters_train_{k_community}_{clustering_seed}_interembTrue.pt")):
        print("#" * 10 + "cluster foreground only" + "#" * 10)
        print("Loading embeddings from disk")
        val_emb = torch.load(val_features_path)
        train_emb = torch.load(train_features_path)
        fg_mask_val, fg_mask_train = torch.load(val_fg_path), torch.load(train_fg_path)
        cluster_all(torch.cat((train_emb, val_emb)).reshape(-1, pca_dim),
                    K=[k_community], mask=torch.cat((fg_mask_train, fg_mask_val)), spatial_res=spatial_res,
                    experiment_folder=experiment_folder, pca_dim=pca_dim, train_len=len(train_loader.dataset),
                    interpolate_embeddings=True, seed=clustering_seed)
        if evaluate_cbfe:
            for i in range(5):
                cluster_all(torch.cat((train_emb, val_emb)).reshape(-1, pca_dim),
                            K=[num_objects_pvoc], mask=torch.cat((fg_mask_train, fg_mask_val)), spatial_res=spatial_res,
                            experiment_folder=experiment_folder, pca_dim=pca_dim, train_len=len(train_loader.dataset),
                            interpolate_embeddings=True, seed=i)
        print("Done")

    # Evaluate fg clustering with k_gt = k_clus to get CBFE semantic segmentation performance.
    if evaluate_cbfe:
        print("Evaluate k=20 performance")
        preds = []
        for i in range(5):
            preds.append(evaluate_clustering(num_objects_pvoc, i, "val", experiment_folder, save_folder,
                                             interpolate_embeddings=True, used_mask=True))
        res = num_objects_pvoc, [p[0] for p in preds], np.argmax([p[0] for p in preds])
        print(res)
        print(np.mean(res[1]))
        print("Done")

    print("#" * 10 + " Start unsupervised overclustering through community detection " + "#" * 10)
    if compute_upper_bound:
        print(f"Upper bound for cd with k={k_community + 1} is: ")
        print(evaluate_clustering(k_community, clustering_seed, "val", experiment_folder, save_folder,
                                  interpolate_embeddings=True, used_mask=True)[0])

    # Construct undirected graph with clusters as nodes and edges derived from co-occurrence probabilities.
    print("Constructing graph")
    train_clusters = torch.load(os.path.join(experiment_folder, "clusters",
                                             f"clusters_train_{k_community}_{clustering_seed}_interembTrue.pt")).int()
    val_clusters = torch.load(os.path.join(experiment_folder, "clusters",
                                           f"clusters_val_{k_community}_{clustering_seed}_interembTrue.pt")).int()
    num_clusters = len(torch.unique(train_clusters))
    print(num_clusters, "okokok", len(torch.unique(val_clusters)))
    assert num_clusters == (k_community + 1) and len(torch.unique(val_clusters)) == (k_community + 1)
    clusters = torch.cat((train_clusters, val_clusters))#B x 1 x 100 x100
    adj_mat = create_matrix(clusters, num_clusters, [1, 2])# 150 x 150
    edges = [(i, j, min(adj_mat[i, j].item(), adj_mat[j, i].item()))
             for i in range(num_clusters)
             for j in range(i, num_clusters)
             if adj_mat[i, j] >= weight_threshold and adj_mat[j, i] >= weight_threshold]
    print(
        f"Adding {len(edges)} edges with {len(set([edge[0] for edge in edges]).union(set([edge[1] for edge in edges])))}"
        f"nodes out of {num_clusters} clusters")
    # Run community detection for num_runs different seeds
    comms = []
    for i in range(num_runs):
        im = Infomap(directed=False, two_level=True, seed=i+1, markov_time=markov_time, silent=True,
                     preferred_number_of_modules=num_objects_pvoc)# mistake corrected. seed in Infomap can only be positive, not 0.
        im.add_links(edges)
        im.run()
        cluster_id_to_merged = {}
        for node in im.tree:
            if node.is_leaf:
                cluster_id_to_merged[node.node_id] = node.module_id
        print(f"Found {len(set(cluster_id_to_merged.values()))} comms")
        comms.append(cluster_id_to_merged)
    # Dump communities
    joblib.dump(comms, os.path.join(experiment_folder, "comms.pkl"))

    # Run evaluation
    all_res = []
    if split_cd == "val":
        clusters_cd = val_clusters
    elif split_cd == "train":
        clusters_cd = train_clusters
    else:
        raise ValueError()
    # Load gt
    gt = torch.load(os.path.join(save_folder, f'all_gt_masks_{split_cd}_voc12.pt'))
    gt = gt.unsqueeze(dim=1)  # mistake, I added this line to prepare gt for interpolation. N x 1 x 448 x 448
    gt = nn.functional.interpolate(gt.float(), size=(clustering_eval_size, clustering_eval_size), mode='nearest')
    # Merge clusters using node to community map
    for i, com in enumerate(comms):
        abc=[]
        merged_clusters = torch.zeros_like(clusters_cd)#N x 1 x 100 x 100
        for cluster_id, merged_id in com.items():
            abc.append(cluster_id)
            merged_clusters[clusters_cd == int(cluster_id)] = merged_id
        abc.sort()
        print(abc, len(abc))
        torch.save(merged_clusters, os.path.join(experiment_folder,
                                                 f"cd_{k_community}_{i}_{split_cd}_{weight_threshold}_{markov_time}.pt"))
        # calculate mIoU
        merged_clusters_flat = merged_clusters[gt != 255]
        gt_wo_boundary = gt[gt != 255]
        assert merged_clusters_flat.size(0) == gt_wo_boundary.size(0)
        preds_miou_protos_clus = PredsmIoU(num_objects_pvoc + 1, num_objects_pvoc + 1)
        preds_miou_protos_clus.update(gt_wo_boundary, merged_clusters_flat)
        res = preds_miou_protos_clus.compute(True, many_to_one=False)
        print("current miou: ", res[0], "acc: ", res[1])
        all_res.append(res)
    print([r[0] for r in all_res])
    print([r[1] for r in all_res])
    print(f"Mean iou of {split_cd} set performance is {np.mean([r[0] for r in all_res])}")
    print(f"mIoU Std of {split_cd} set performance is {np.std([r[0] for r in all_res])}")
    print(f"Mean acc of {split_cd} set performance is {np.mean([r[1] for r in all_res])}")
    print(f"Acc Std of {split_cd} set performance is {np.std([r[1] for r in all_res])}")
    return np.mean([r[0] for r in all_res])

weight_path = "mogoseg_vit_epoch9_pascal_15.pth.tar"
patch_size = 16
batch_size=15
clustering_seed=2
embedding_folder="/workspace/MoGoSeg/embeddings_p16_pascal_100"
best_k=189
best_mt=1
best_wt=0.03
# weight_path = "mogoseg_vit_epoch9_pascal_16_p8.pth.tar"
# patch_size = 8
# batch_size=8
# clustering_seed=1# seed 0 will cause val features have less clusters than train features
# embedding_folder="/workspace/MoGoSeg/embeddings_p16_pascal_100"#"/workspace/MoGoSeg/embeddings_p8_pascal_100"
# best_k=179
# best_mt=1#1.8000000000000003#0.95
# best_wt=0.03#0.03
pca_dim=100

@click.command()
@click.option('--experiment_name', required=True, help="")
@click.option('--save_folder', default=embedding_folder)
# Model vars
@click.option('--patch_size', default=patch_size, help="")
@click.option('--arch', default="vit-small", help="")
#@click.option('--ckpt_path', required=True, help="")
# Data vars
@click.option('--batch_size', default=batch_size)
@click.option('--input_size', default=448)
@click.option('--data_dir', default="/tmp/voc")
@click.option('--pca_dim', default=pca_dim)#50
# Clustering vars
@click.option('--k_fg_extraction', default=200)
@click.option('--clustering_eval_size', default=100)
@click.option('--evaluate_cbfe', default=False)
@click.option('--clustering_seed', default=clustering_seed)
# Community Detection vars
@click.option('--num_runs', default=10)#10
@click.option('--compute_upper_bound', default=False)
@click.option('--weight_path', default=weight_path)
# @click.option('--best_k', default = best_k, type=int)
# @click.option('--best_mt', default = best_mt,  type=float)
# @click.option('--best_wt', default=best_wt, type=float)#0.07

@click.option('--best_k',type=int)
@click.option('--best_mt', type=float)
@click.option('--best_wt', type=float)#0.07
def find_hyperparams(patch_size: int, arch: str, experiment_name: str, batch_size: int, input_size: int,
                     save_folder: str, data_dir: str, pca_dim: int, k_fg_extraction: int,
                     clustering_eval_size: int, evaluate_cbfe: bool, clustering_seed: int,
                     num_runs: int, compute_upper_bound: bool, best_k: int = None,
                     best_mt: float = None, best_wt: float = None, weight_path:str=None) -> None:
    """
    Evaluate fully unsupervised semantic segmentation. Optionally find good hyperparameters for community detection.
    :param patch_size: the patch size of ViT
    :param arch: the architecture of the ViT
    :param ckpt_path: the file path to the checkpoint of the model fine-tuned with leopart
    :param experiment_name: the name for the experiment. A folder within save_folder will be created for the results
    :param batch_size: the batch size used for calculating the embeddings
    :param input_size: the input size of the images
    :param save_folder: the root folder to save to
    :param data_dir: the root directory to the PVOC dataset
    :param pca_dim: the target dimensionality of the embeddings. Transformed by PCA.
    :param k_fg_extraction: the clustering granularity used for cluster-based foreground extraction (CBFE)
    :param clustering_eval_size: the size of the segmentation maps used for evaluation. This is kept to 100 as we have
    not seen any significant distance in performance and we gain a significant evaluation speed-up.
    :param evaluate_cbfe: Flag to indiciate direct segmentation evaluation by just clustering foreground.
    :param clustering_seed: The seed used for all clustering runs.
    :param num_runs: The number of seeds used for evaluating community detection (CD) performance
    :param compute_upper_bound:  Flag to compute the upper bound of performance attainable by community detection.
    :param best_k: clustering granularity used for community detection. Has to be supplied to skip hyperparam tuning.
    :param best_mt: markov time used for community detection. Has to be supplied to skip hyperparam tuning.
    :param best_wt: weight threshold for graph construction. Has to be supplied to skip hyperparam tuning.
    """
    num_objects_pvoc = 20

    # Start hyperparameter search if cd hyperparams are not provided
    if any([best_mt is None, best_wt is None, best_k is None]):
        def objective(trial):
            wt = trial.suggest_float("weight_threshold", 0.01, 0.1, step=0.01)
            mt = trial.suggest_float("markov_time", 0.1, 2, step=0.1)
            k_community = trial.suggest_int("k_community", 99, 249, step=10)
            return start_unsup_seg(patch_size=patch_size, arch=arch,
                                   experiment_name=experiment_name, batch_size=batch_size, input_size=input_size,
                                   save_folder=save_folder, data_dir=data_dir, pca_dim=pca_dim,
                                   k_fg_extraction=k_fg_extraction, clustering_eval_size=clustering_eval_size,
                                   evaluate_cbfe=evaluate_cbfe, clustering_seed=clustering_seed,
                                   num_objects_pvoc=num_objects_pvoc, k_community=k_community,
                                   markov_time=mt, weight_threshold=wt, num_runs=num_runs,
                                   compute_upper_bound=compute_upper_bound, split_cd="val", weight_path=weight_path)

        sampler = optuna.samplers.TPESampler(seed=1000)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        try:
            study.optimize(objective, n_trials=150, catch=(AssertionError,), n_jobs=5)
        except Exception:
            joblib.dump(study, "study.pkl")

        joblib.dump(study, "study.pkl")

        best_params = study.best_params
        best_wt = best_params["weight_threshold"]
        best_mt = best_params["markov_time"]
        best_k = best_params["k_community"]
        print(f"Edge thr: {best_wt}")
        print(f"Markov time: {best_mt}")
        print(f"K community: {best_k}")
        print(study.best_value)

    # Get pvoc12 val results
    start_unsup_seg(patch_size=patch_size, arch=arch,
                    experiment_name=experiment_name, batch_size=batch_size, input_size=input_size,
                    save_folder=save_folder, data_dir=data_dir, pca_dim=pca_dim,
                    k_fg_extraction=k_fg_extraction, clustering_eval_size=clustering_eval_size,
                    evaluate_cbfe=evaluate_cbfe, clustering_seed=clustering_seed,
                    num_objects_pvoc=num_objects_pvoc, k_community=best_k,
                    markov_time=best_mt, weight_threshold=best_wt, num_runs=num_runs,
                    compute_upper_bound=compute_upper_bound, split_cd="val", weight_path=weight_path)


if __name__ == "__main__":
    find_hyperparams()