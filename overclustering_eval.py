import argparse
import builtins
import os
import random
import warnings
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np
import build
from PIL import Image
from data import eval_transforms
from utils.eval_utils import PredsmIoU, cluster, normalize_and_transform, batched_crf
from data.data_module import EvalPascal, EvalCoco
import vit.vision_transformer as vits
model_weights_root = "weights/seghead_weights/"
eval_config = {
    "hcl_p16_overclustering_pascal":{
        "eval_model_name": "hcl",
        "mode":"overclustering",
        "vit_patch_size":16,
        "evaluate_data":"pascal",
        "downsam_size":100,
        "pca_reduce_dim":100,
        "num_repeat_clustering":5,
        "model_check_point":"online_seg_head",
        "model_weights_path":model_weights_root+"hcl_pascal_p16.pth.tar",
    },

    "hcl_p8_overclustering_pascal":{
        "eval_model_name": "hcl",
        "mode":"overclustering",
        "vit_patch_size":8,
        "evaluate_data":"pascal",
        "downsam_size":100,
        "pca_reduce_dim":100,
        "num_repeat_clustering":5,
        "model_check_point":"online_seg_head",
        "model_weights_path":model_weights_root+"hcl_pascal_p8.pth.tar"
    },

    "hcl_p16_overclustering_coco":{
        "eval_model_name": "hcl",
        "mode":"overclustering",
        "vit_patch_size":16,
        "evaluate_data":"coco",
        "downsam_size":100,
        "pca_reduce_dim":100,
        "num_repeat_clustering":5,
        "model_check_point":"online_seg_head",
        "model_weights_path":model_weights_root+"hcl_coco_p16.pth.tar"
    },
    "hcl_p8_overclustering_coco":{
        "eval_model_name": "hcl",
        "mode":"overclustering",
        "vit_patch_size":8,
        "evaluate_data":"coco",
        "downsam_size":100,
        "pca_reduce_dim":100,
        "num_repeat_clustering":5,
        "model_check_point":"online_seg_head",
        "model_weights_path":model_weights_root+"hcl_coco_p8.pth.tar"
    },
}
selected_config = eval_config["hcl_p16_overclustering_pascal"]

eval_model_name = selected_config["eval_model_name"]
mode =selected_config["mode"]#"overclustering"
assert eval_model_name in ["hcl"]
model_weights_path = selected_config["model_weights_path"]
vit_patch_size = selected_config["vit_patch_size"]
downsam_size = selected_config["downsam_size"]
pca_reduce_dim = selected_config["pca_reduce_dim"]
num_repeat_clustering = selected_config["num_repeat_clustering"]
model_check_point = selected_config["model_check_point"]
evaluate_data = selected_config["evaluate_data"]
coco_data_set = "full"
seghead_type = "transformer_block"#mlp
if evaluate_data == "coco":
    if mode == "hungarian":
        inference_image_size = 320
    else:
        inference_image_size = 448
    if coco_data_set == "full":
        n_classes = 27
    elif coco_data_set == "thing":
        n_classes = 12
    elif coco_data_set == "stuff":
        n_classes = 15
elif evaluate_data == "pascal":
    inference_image_size = 448
    n_classes = 21
k = 500 # K for overclustering
print("you choose to do",mode, " evaluation")
print("dataset", evaluate_data)
print("k", k)
print("inference_image_size", inference_image_size)
parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

parser.add_argument("-j", "--workers", default=32, type=int, metavar="N",
                    help="number of data loading workers (default: 32)", )
parser.add_argument(
    "--classes", default=n_classes, type=int, metavar="N", help="number of total classes"
)
parser.add_argument("-b", "--batch-size", default=256, type=int, metavar="N",
                    help="mini-batch size (default: 256), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel")
parser.add_argument("--world-size", default=-1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=-1, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://224.66.41.62:23456", type=str,
                    help="url used to set up distributed training")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--seed", default=7, type=int, help="seed for initializing training. ")
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument("--multiprocessing-distributed", action="store_true",
                    help="Use multi-processing distributed training to launch "
                         "N processes per node, which has N GPUs. This is the "
                         "fastest way to use PyTorch for either single node or "
                         "multi node data parallel training")
parser.add_argument("--pretrained", default=model_weights_path, type=str,
                    help="path to moco pretrained checkpoint")
parser.add_argument("--patch-size", default=vit_patch_size, type=int, help="vit patch size")
parser.add_argument("--spatial-token-dim", default=384, type=int, help="spatial token dimension (default: 384)")
parser.add_argument("--inference-img-size", default=inference_image_size, type=int, help="vit inference image size")
parser.add_argument("--downsample-size", default=downsam_size, type=int, help="vit inference image size")
parser.add_argument("--pca-dim", default=pca_reduce_dim, type=int, help="reduce vit token dimension using pca")
parser.add_argument("--num-seeds", default=num_repeat_clustering, type=int, help="number of time of performing kmeans clustering")
parser.add_argument("--k", default=k, type=int, help="number of clusters for kmeans. 500 overcluster")
parser.add_argument("--ignore-index", default=255, type=int, help="label index to be ignored")
parser.add_argument("--num-classes", default=n_classes, type=int, help="number of classes")
parser.add_argument("--seghead-type", default=seghead_type, type=str, help="distributed backend")
parser.add_argument("--eval-mode", default=mode, type=str, help="eval mode, hungarian matching and overclustering")
parser.add_argument("--checkpoint_key", default=model_check_point, type=str,
                    help='Key to use in the checkpoint (example: "online_seg_head")')
parser.add_argument('--arch', default='vit_small', type=str, choices=['vit_tiny', 'vit_small', 'vit_base'],
                    help="Name of architecture to train. For quick experiments with ViTs,we recommend using vit_tiny or vit_small.")
parser.add_argument("--model-name", default=eval_model_name, type=str, help='eval model name')
best_iou = 0

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_iou
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    # create model
    print("=> creating model vit small")
    if args.model_name == "hcl":
        model = build.HCLEval(
            vit_arch=args.arch,
            pretrained_path=args.pretrained,
            checkpoint_key=args.checkpoint_key,
            patch_size=args.patch_size,
            embed_dim=args.spatial_token_dim,
            seg_head_type=args.seghead_type,
        )
    else:
        model = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            pretrained=True,
            pretrain_weights=args.model_name,
            use_projector=False,
            num_classes=args.classes
        )
        # freeze all layers but the last classification layer of PSPNet.
        for name, param in model.named_parameters():
            if name not in ["head.weight", "head.bias"]:
                param.requires_grad = False
                # print(name)
    print(model)

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    # Data loading code
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    # we scale mean value by 255 is because the values of image pixels will be from 0 - 255 after ToTensor.
    # This ToTensor is implemented by us, it won't rescale the value to 0 -1 like a standard ToTensor function.
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    if evaluate_data == "pascal" or args.eval_mode == "overclustering":
        val_transforms = eval_transforms.Compose([
            eval_transforms.Resize((args.inference_img_size, args.inference_img_size)),  # (448,448)
            eval_transforms.ToTensor(),
            eval_transforms.Normalize(mean=mean, std=std)])
        print("eval data++++++++", evaluate_data, "eval mode ++++++++", args.eval_mode)
    else:
        val_transforms = eval_transforms.Compose([
            eval_transforms.ResizeAndCenterCrop(args.inference_img_size),
            eval_transforms.ToTensor(),
            eval_transforms.Normalize(mean=mean, std=std)])
        print("eval data---------", evaluate_data, "eval mode ++++++++", args.eval_mode)


    if evaluate_data == "coco":
        val_dataset = EvalCoco(root_path="/workspace/coco_2017/", split="val", transform=val_transforms,
                               coarse_labels=True, data_set=coco_data_set, subset="iic_subset_val")#iic_subset_val
    elif evaluate_data == "pascal":
        val_dataset = EvalPascal(root_path="E:/Projects/ssl/Dataset/VOCdevkit/VOC2012/", split="val", transform=val_transforms)

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler
    )

    validate(val_loader, model, args)


def validate(val_loader, model, args):
    model.eval()
    with torch.no_grad():
        masks_list = []
        tokens_list = []
        for i, (images, masks) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            masks = masks.cuda(args.gpu, non_blocking=True)
            # print(masks.shape, args.inference_img_size)
            # compute output
            if args.model_name == "hcl":
                tokens = model.forward_backbone(images)  # N x num_spa_tokens x emdim
            else:
                tokens = model.forward_backbone(images)
            tokens = tokens.reshape(tokens.shape[0] * tokens.shape[1], tokens.shape[2])  # Nnum_spa_tokens x emdim
            if masks.size(2) != args.downsample_size:
                masks = TF.resize(masks, args.downsample_size, Image.NEAREST)
            masks_list.append(masks.cpu())
            tokens_list.append(tokens.cpu())
    all_masks = torch.cat(masks_list, dim=0).unsqueeze(dim=1)  # N x 1 x 100 x 100
    all_tokens = torch.cat(tokens_list, dim=0)
    print(all_masks.shape, all_tokens.shape)
    normalized_feats = normalize_and_transform(all_tokens, args.pca_dim)
    clusterings = []
    spatial_res = args.inference_img_size // args.patch_size  # 28
    for i in range(args.num_seeds):
        clusterings.append(cluster(args.pca_dim, normalized_feats.numpy(), spatial_res, args.k, seed=i))

    miou=[]
    acc=[]
    print(f"Number of pixels ignored is: {torch.sum(all_masks == args.ignore_index)}")
    for clustering in clusterings:
        clustering = F.interpolate(clustering.float(), size=(args.downsample_size, args.downsample_size),
                                   mode='nearest')
        metric = PredsmIoU(args.k, args.num_classes)
        metric.update(all_masks[all_masks != args.ignore_index], clustering[all_masks != args.ignore_index])
        if args.k == args.num_classes:
            results = metric.compute(True, many_to_one=False, precision_based=False)
            miou.append(results[0])
            acc.append(results[1])
        else:
            results = metric.compute(True, many_to_one=True, precision_based=True)
            miou.append(results[0])
            acc.append(results[1])
        print("finish current clustering, miou", results[0], "acc: ", results[1])
    print(miou, acc)
    print("miou: ", np.mean(miou), "acc: ",np.mean(acc))


if __name__ == "__main__":
    main()
