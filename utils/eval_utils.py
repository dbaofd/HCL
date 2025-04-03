# Adopted from https://github.com/wvangansbeke/Unsupervised-Semantic-Segmentation/blob/main/segmentation/utils/utils.py
import faiss
import numpy as np
import time
import torch
import torch.nn as nn
from collections import defaultdict
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from torchmetrics import Metric
from typing import List, Tuple, Dict, Any
import torch.nn.functional as F
import torchvision.transforms.functional as VF
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
from utils.dist_utils import all_reduce_dict
from datetime import datetime
import os
class SemsegMeter(object):
    def __init__(self, num_classes, class_names, ignore_index=255):
        self.num_classes = num_classes 
        self.class_names = class_names
        self.tp = [0] * self.num_classes
        self.fp = [0] * self.num_classes
        self.fn = [0] * self.num_classes
        assert(ignore_index == 255)
        self.ignore_index = ignore_index

    def update(self, pred, gt):
        valid = (gt != self.ignore_index)

        for i_part in range(0, self.num_classes):
            tmp_gt = (gt == i_part)
            tmp_pred = (pred == i_part)
            self.tp[i_part] += torch.sum(tmp_gt & tmp_pred & valid).item()
            self.fp[i_part] += torch.sum(~tmp_gt & tmp_pred & valid).item()
            self.fn[i_part] += torch.sum(tmp_gt & ~tmp_pred & valid).item()

    def reset(self):
        self.tp = [0] * self.num_classes
        self.fp = [0] * self.num_classes
        self.fn = [0] * self.num_classes
            
    def return_score(self, verbose=True):
        jac = [0] * self.num_classes
        for i_part in range(self.num_classes):
            jac[i_part] = float(self.tp[i_part]) / max(float(self.tp[i_part] + self.fp[i_part] + self.fn[i_part]), 1e-8)

        eval_result = dict()
        eval_result['jaccards_all_categs'] = jac
        eval_result['mIoU'] = np.mean(jac)

        if verbose:
            print('Evaluation of semantic segmentation ')
            print('mIoU is %.2f' %(100*eval_result['mIoU']))
            for i_part in range(self.num_classes):
                print('IoU class %s is %.2f' %(self.class_names[i_part], 100*jac[i_part]))

        return eval_result


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
class ClusterLookup(nn.Module):

    def __init__(self, dim: int, n_classes: int, mode = "cluster_head"):
        super(ClusterLookup, self).__init__()
        assert mode in ["cluster_head", "cls_prototype"]
        self.mode = mode
        self.n_classes = n_classes
        self.dim = dim
        self.clusters = torch.nn.Parameter(torch.randn(n_classes, dim))

    def reset_parameters(self):
        with torch.no_grad():
            self.clusters.copy_(torch.randn(self.n_classes, self.dim))

    def forward(self, x, alpha, log_probs=False, is_direct=False):
        if is_direct:
            inner_products = x
        else:
            normed_clusters = F.normalize(self.clusters, dim=1)
            normed_features = F.normalize(x, dim=1)
            if self.mode == "cluster_head":
                inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)
            elif self.mode == "cls_prototype":
                inner_products = torch.einsum("bc,nc->bn", normed_features, normed_clusters)
        if alpha is None:
            if self.mode == "cluster_head":
                cluster_probs = F.one_hot(torch.argmax(inner_products, dim=1), self.clusters.shape[0]) \
                    .permute(0, 3, 1, 2).to(torch.float32) # bnhw
                #torch.argmax(inner_products, dim=1):b x h x w
                #F.one_hot(torch.argmax(inner_products, dim=1), self.clusters.shape[0]) b x h x w x n_classes
            elif self.mode == "cls_prototype":
                cluster_probs = F.one_hot(torch.argmax(inner_products, dim=1), self.clusters.shape[0]) \
                    .to(torch.float32)#bn
        else:
            cluster_probs = nn.functional.softmax(inner_products * alpha, dim=1)
        cluster_loss = -(cluster_probs * inner_products).sum(1).mean()

        if log_probs:
            return cluster_loss, nn.functional.log_softmax(inner_products * alpha, dim=1)
        else:
            return cluster_loss, cluster_probs


class PredsmIoU(Metric):
    """
    Adopted from leopart github
    Subclasses Metric. Computes mean Intersection over Union (mIoU) given ground-truth and predictions.
    .update() can be called repeatedly to add data from multiple validation loops.
    """
    def __init__(self,
                 num_pred_classes: int,
                 num_gt_classes: int):
        """
        :param num_pred_classes: The number of predicted classes.
        :param num_gt_classes: The number of gt classes.
        """
        super().__init__(dist_sync_on_step=False)#, compute_on_step=False)
        self.num_pred_classes = num_pred_classes
        self.num_gt_classes = num_gt_classes
        self.add_state("gt", [])
        self.add_state("pred", [])
        self.n_jobs = -1

    def update(self, gt: torch.Tensor, pred: torch.Tensor) -> None:
        self.gt.append(gt)
        self.pred.append(pred)

    def compute(self, is_global_zero: bool, many_to_one: bool = False,
                precision_based: bool = False, linear_probe : bool = False) -> Tuple[float, List[np.int64],
                                                                                     List[np.int64], List[np.int64],
                                                                                     List[np.int64], float]:
        """
        Compute mIoU with optional hungarian matching or many-to-one matching (extracts information from labels).
        :param is_global_zero: Flag indicating whether process is rank zero. Computation of metric is only triggered
        if True.
        :param many_to_one: Compute a many-to-one mapping of predicted classes to ground truth instead of hungarian
        matching.
        :param precision_based: Use precision as matching criteria instead of IoU for assigning predicted class to
        ground truth class.
        :param linear_probe: Skip hungarian / many-to-one matching. Used for evaluating predictions of fine-tuned heads.
        :return: mIoU over all classes, true positives per class, false negatives per class, false positives per class,
        reordered predictions matching gt,  percentage of clusters matched to background class. 1/self.num_pred_classes
        if self.num_pred_classes == self.num_gt_classes.
        """
        if is_global_zero:
            pred = torch.cat(self.pred).cpu().numpy().astype(int)
            gt = torch.cat(self.gt).cpu().numpy().astype(int)
            assert len(np.unique(pred)) <= self.num_pred_classes
            assert np.max(pred) <= self.num_pred_classes
            return self.compute_miou(gt, pred, self.num_pred_classes, self.num_gt_classes, many_to_one=many_to_one,
                                     precision_based=precision_based, linear_probe=linear_probe)

    def compute_miou(self, gt: np.ndarray, pred: np.ndarray, num_pred: int, num_gt:int,
                     many_to_one=False, precision_based=False, linear_probe=False) -> Tuple[float, List[np.int64], List[np.int64], List[np.int64],
                                                  List[np.int64], float]:
        """
        Compute mIoU with optional hungarian matching or many-to-one matching (extracts information from labels).
        :param gt: numpy array with all flattened ground-truth class assignments per pixel
        :param pred: numpy array with all flattened class assignment predictions per pixel
        :param num_pred: number of predicted classes
        :param num_gt: number of ground truth classes
        :param many_to_one: Compute a many-to-one mapping of predicted classes to ground truth instead of hungarian
        matching.
        :param precision_based: Use precision as matching criteria instead of IoU for assigning predicted class to
        ground truth class.
        :param linear_probe: Skip hungarian / many-to-one matching. Used for evaluating predictions of fine-tuned heads.
        :return: mIoU over all classes, true positives per class, false negatives per class, false positives per class,
        reordered predictions matching gt,  percentage of clusters matched to background class. 1/self.num_pred_classes
        if self.num_pred_classes == self.num_gt_classes.
        """
        assert pred.shape == gt.shape
        print(f"seg map preds have size {gt.shape}")
        tp = [0] * num_gt
        fp = [0] * num_gt
        fn = [0] * num_gt
        jac = [0] * num_gt

        if linear_probe:
            reordered_preds = pred
            matched_bg_clusters = {}
        else:
            if many_to_one:
                match = self._original_match(num_pred, num_gt, pred, gt, precision_based=precision_based)
                # remap predictions
                reordered_preds = np.zeros(len(pred))
                for target_i, matched_preds in match.items():
                    for pred_i in matched_preds:
                        reordered_preds[pred == int(pred_i)] = int(target_i)
                matched_bg_clusters = len(match[0]) / num_pred
            else:
                match = self._hungarian_match(num_pred, num_gt, pred, gt)
                # remap predictions
                reordered_preds = np.zeros(len(pred))
                for target_i, pred_i in zip(*match):
                    reordered_preds[pred == int(pred_i)] = int(target_i)
                # merge all unmatched predictions to background
                for unmatched_pred in np.delete(np.arange(num_pred), np.array(match[1])):
                    reordered_preds[pred == int(unmatched_pred)] = 0
                matched_bg_clusters = 1/num_gt

        # tp, fp, and fn evaluation
        for i_part in range(0, num_gt):
            tmp_all_gt = (gt == i_part)
            tmp_pred = (reordered_preds == i_part)
            tp[i_part] += np.sum(tmp_all_gt & tmp_pred)
            fp[i_part] += np.sum(~tmp_all_gt & tmp_pred)
            fn[i_part] += np.sum(tmp_all_gt & ~tmp_pred)

        # Calculate IoU per class
        for i_part in range(0, num_gt):
            jac[i_part] = float(tp[i_part]) / max(float(tp[i_part] + fp[i_part] + fn[i_part]), 1e-8)

        print("IoUs computed")
        pixel_acc = float(np.sum(tp))/(np.sum(tp)+np.sum(fp)+np.sum(fn))
        return np.mean(jac), pixel_acc, tp, fp, fn, reordered_preds.astype(int).tolist(), matched_bg_clusters

    @staticmethod
    def get_score(flat_preds: np.ndarray, flat_targets: np.ndarray, c1: int, c2: int, precision_based: bool = False) \
            -> float:
        """
        Calculates IoU given gt class c1 and prediction class c2.
        :param flat_preds: flattened predictions
        :param flat_targets: flattened gt
        :param c1: ground truth class to match
        :param c2: predicted class to match
        :param precision_based: flag to calculate precision instead of IoU.
        :return: The score if gt-c1 was matched to predicted c2.
        """
        tmp_all_gt = (flat_targets == c1)
        tmp_pred = (flat_preds == c2)
        tp = np.sum(tmp_all_gt & tmp_pred)
        fp = np.sum(~tmp_all_gt & tmp_pred)
        if not precision_based:
            fn = np.sum(tmp_all_gt & ~tmp_pred)
            jac = float(tp) / max(float(tp + fp + fn), 1e-8)
            return jac
        else:
            prec = float(tp) / max(float(tp + fp), 1e-8)
            return prec

    def compute_score_matrix(self, num_pred: int, num_gt: int, pred: np.ndarray, gt: np.ndarray,
                             precision_based: bool = False) -> np.ndarray:
        """
        Compute score matrix. Each element i, j of matrix is the score if i was matched j. Computation is parallelized
        over self.n_jobs.
        :param num_pred: number of predicted classes
        :param num_gt: number of ground-truth classes
        :param pred: flattened predictions
        :param gt: flattened gt
        :param precision_based: flag to calculate precision instead of IoU.
        :return: num_pred x num_gt matrix with A[i, j] being the score if ground-truth class i was matched to
        predicted class j.
        """
        print("Parallelizing iou computation")
        start = time.time()
        score_mat = Parallel(n_jobs=self.n_jobs)(delayed(self.get_score)(pred, gt, c1, c2, precision_based=precision_based)
                                                 for c2 in range(num_pred) for c1 in range(num_gt))
        print(f"took {time.time() - start} seconds")
        score_mat = np.array(score_mat)
        return score_mat.reshape((num_pred, num_gt)).T

    def _hungarian_match(self, num_pred: int, num_gt: int, pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray,
                                                                                                      np.ndarray]:
        # do hungarian matching. If num_pred > num_gt match will be partial only.
        iou_mat = self.compute_score_matrix(num_pred, num_gt, pred, gt)
        match = linear_sum_assignment(1 - iou_mat)
        print("Matched clusters to gt classes:")
        print(match)
        return match

    def _original_match(self, num_pred, num_gt, pred, gt, precision_based=False) -> Dict[int, list]:
        score_mat = self.compute_score_matrix(num_pred, num_gt, pred, gt, precision_based=precision_based)
        preds_to_gts = {}
        preds_to_gt_scores = {}
        # Greedily match predicted class to ground-truth class by best score.
        for pred_c in range(num_pred):
            for gt_c in range(num_gt):
                score = score_mat[gt_c, pred_c]
                if (pred_c not in preds_to_gts) or (score > preds_to_gt_scores[pred_c]):
                    preds_to_gts[pred_c] = gt_c
                    preds_to_gt_scores[pred_c] = score
        gt_to_matches = defaultdict(list)
        for k,v in preds_to_gts.items():
            gt_to_matches[v].append(k)
        print("matched clusters to gt classes:")
        return gt_to_matches

class UnsupervisedMetrics(Metric):
    def __init__(self, prefix: str, n_classes: int, extra_clusters: int, compute_hungarian: bool,
                 dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.n_classes = n_classes
        self.extra_clusters = extra_clusters
        self.compute_hungarian = compute_hungarian
        self.prefix = prefix
        self.stats = torch.zeros(n_classes + self.extra_clusters, n_classes,
                                           dtype=torch.int64, device="cuda")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        with torch.no_grad():
            actual = target.reshape(-1) # 32*320*320
            preds = preds.reshape(-1) # 32*320*320

            mask = (actual >= 0) & (actual < self.n_classes) & (preds !=255) & (preds < self.n_classes)
            actual = actual[mask]
            preds = preds[mask]
            self.stats += torch.bincount(
                (self.n_classes + self.extra_clusters) * actual + preds,
                minlength=self.n_classes * (self.n_classes + self.extra_clusters)) \
                .reshape(self.n_classes, self.n_classes + self.extra_clusters).t().to(self.stats.device)

    def map_clusters(self, clusters):
        if self.extra_clusters == 0:
            return torch.tensor(self.assignments[1])[clusters]
        else:
            missing = sorted(list(set(range(self.n_classes + self.extra_clusters)) - set(self.assignments[0])))
            cluster_to_class = self.assignments[1]
            for missing_entry in missing:
                if missing_entry == cluster_to_class.shape[0]:
                    cluster_to_class = np.append(cluster_to_class, -1)
                else:
                    cluster_to_class = np.insert(cluster_to_class, missing_entry + 1, -1)
            cluster_to_class = torch.tensor(cluster_to_class)
            return cluster_to_class[clusters]

    def compute(self):
        if self.compute_hungarian:  # cluster
            self.assignments = linear_sum_assignment(self.stats.detach().cpu(), maximize=True)  # row, col
            if self.extra_clusters == 0:
                self.histogram = self.stats[np.argsort(self.assignments[1]), :]

            if self.extra_clusters > 0:
                self.assignments_t = linear_sum_assignment(self.stats.detach().cpu().t(), maximize=True)
                histogram = self.stats[self.assignments_t[1], :]
                missing = list(set(range(self.n_classes + self.extra_clusters)) - set(self.assignments[0]))
                new_row = self.stats[missing, :].sum(0, keepdim=True)
                histogram = torch.cat([histogram, new_row], axis=0)
                new_col = torch.zeros(self.n_classes + 1, 1, device=histogram.device)
                self.histogram = torch.cat([histogram, new_col], axis=1)
        else:  # linear
            self.assignments = (torch.arange(self.n_classes).unsqueeze(1),
                                torch.arange(self.n_classes).unsqueeze(1))
            self.histogram = self.stats

        tp = torch.diag(self.histogram)
        fp = torch.sum(self.histogram, dim=0) - tp
        fn = torch.sum(self.histogram, dim=1) - tp
        iou = tp / (tp + fp + fn)
        prc = tp / (tp + fn)
        opc = torch.sum(tp) / torch.sum(self.histogram)

        metric_dict = {self.prefix + "mIoU": iou[~torch.isnan(iou)].mean().item(),
                       self.prefix + "Accuracy": opc.item()}
        return {k: 100 * v for k, v in metric_dict.items()}

def get_linear_weights(linear_pretrained_path):
    if os.path.isfile(linear_pretrained_path):
        linear_state_dict = torch.load(linear_pretrained_path, map_location="cpu")["linear_classifier"]
        print("loading weights from, ", linear_pretrained_path)
        for k in list(linear_state_dict.keys()):
            if k.startswith("module."):
                linear_state_dict[k[len("module."):]] = linear_state_dict[k]
            del linear_state_dict[k]
        return linear_state_dict
    else:
        print("=> no checkpoint found at '{}'".format(linear_pretrained_path))
        return 0

def get_metrics(m1: UnsupervisedMetrics, m2: UnsupervisedMetrics) -> Dict[str, Any]:
    metric_dict_1 = m1.compute()
    metric_dict_2 = m2.compute()
    metrics = all_reduce_dict(metric_dict_1, op="mean")
    tmp = all_reduce_dict(metric_dict_2, op="mean")
    metrics.update(tmp)
    return metrics

def get_single_metric(m1: UnsupervisedMetrics) -> Dict[str, Any]:
    metric_dict_1 = m1.compute()
    metrics = all_reduce_dict(metric_dict_1, op="mean")
    return metrics

def time_log() -> str:
    a = datetime.now()
    return f"*" * 48 + f"  {a.year:>4}/{a.month:>2}/{a.day:>2} | {a.hour:>2}:{a.minute:>2}:{a.second:>2}\n"



def normalize_and_transform(feats: torch.Tensor, pca_dim: int) -> torch.Tensor:
    feats = feats.numpy()
    # Iteratively train scaler to normalize data
    bs = 100001
    num_its = (feats.shape[0] // bs) + 1
    scaler = StandardScaler()
    for i in range(num_its):
        scaler.partial_fit(feats[i * bs:(i + 1) * bs])
    print("trained scaler")
    for i in range(num_its):
        feats[i * bs:(i + 1) * bs] = scaler.transform(feats[i * bs:(i + 1) * bs])
    print(f"normalized feats to {feats.shape}")
    # Do PCA
    pca = faiss.PCAMatrix(feats.shape[-1], pca_dim)
    pca.train(feats)
    assert pca.is_trained
    transformed_val = pca.apply_py(feats)
    print(f"val feats transformed to {transformed_val.shape}")
    return torch.from_numpy(transformed_val)

def cluster(pca_dim: int, transformed_feats: np.ndarray, spatial_res: int, k: int, seed: int = 1,
            mask: torch.Tensor = None, spherical: bool = False):
    """
    Adapted from leopart
    Computes k-Means and retrieve assignments for each feature vector. Optionally the clusters are only computed on
    foreground vectors if a mask is provided. In this case tranformed_feats is already expected to contain only the
    foreground vectors.
    """
    print(f"start clustering with {seed}")
    kmeans = faiss.Kmeans(pca_dim, k, niter=100, nredo=5, verbose=True,
                          gpu=False, spherical=spherical, seed=seed)
    kmeans.train(transformed_feats)
    print("kmeans trained")
    _, pred_labels = kmeans.index.search(transformed_feats, 1)
    clusters = pred_labels.squeeze()
    print("index search done")

    # Apply fg mask if provided.
    if mask is not None:
        preds = torch.zeros_like(mask) + k
        preds[mask.bool()] = torch.from_numpy(clusters).float()
    else:
        preds = torch.from_numpy(clusters.reshape(-1, 1, spatial_res, spatial_res))
    return preds


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2

unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
MAX_ITER = 10
POS_W = 3
POS_XY_STD = 1
Bi_W = 4
Bi_XY_STD = 67
Bi_RGB_STD = 3
def dense_crf(image_tensor: torch.FloatTensor, output_logits: torch.FloatTensor):
    image = np.array(VF.to_pil_image(unnorm(image_tensor)))[:, :, ::-1]
    H, W = image.shape[:2]
    image = np.ascontiguousarray(image)

    # output_logits = F.interpolate(output_logits.unsqueeze(0), size=(H, W), mode="bilinear",
    #                               align_corners=False).squeeze()
    output_probs = F.softmax(output_logits, dim=0).cpu().numpy()

    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q


def _apply_crf(tup):
    return dense_crf(tup[0], tup[1])


def batched_crf(img_tensor, prob_tensor):
    batch_size = list(img_tensor.size())[0]
    img_tensor_cpu = img_tensor.detach().cpu()
    prob_tensor_cpu = prob_tensor.detach().cpu()
    out = []
    for i in range(batch_size):
        out_ = dense_crf(img_tensor_cpu[i], prob_tensor_cpu[i])
        out.append(out_)

    return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in out], dim=0)