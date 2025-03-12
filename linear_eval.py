import argparse
import builtins
import os
import random
import time
import warnings
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import build as build
from data import eval_transforms
from utils.eval_utils import SemsegMeter, AverageMeter, ProgressMeter, batched_crf,UnsupervisedMetrics, get_single_metric, time_log, get_linear_weights
from data.data_module import EvalPascal, EvalCoco
import vit.vision_transformer as vits

eval_config = {
    "hcl_p16_eval_pascal":{
        "eval_model_name": "hcl",
        "vit_patch_size":16,
        "evaluate_data":"pascal",
        "using_crf": True,
        "model_weights_path":"seghead_weights/hcl_pascal_p16.pth.tar",
        "linear_weights_path":"linear_classifier_weights/linear_classifier_pascal_p16.pth.tar",
    },
    "hcl_p8_eval_pascal":{
        "eval_model_name": "hcl",
        "vit_patch_size":8,
        "evaluate_data":"pascal",
        "using_crf": True,
        "model_weights_path":"seghead_weights/hcl_pascal_p8.pth.tar",
        "linear_weights_path":"linear_classifier_weights/linear_classifier_pascal_p8.pth.tar",
    },
    "hcl_p16_eval_coco":{
        "eval_model_name": "hcl",
        "vit_patch_size":16,
        "evaluate_data":"coco",
        "using_crf": True,
        "model_weights_path":"seghead_weights/hcl_coco_p16.pth.tar",
        "linear_weights_path":"linear_classifier_weights/linear_classifier_coco_p16.pth.tar",
    },
    "hcl_p8_eval_coco":{
        "eval_model_name": "hcl",
        "vit_patch_size":8,
        "evaluate_data":"coco",
        "using_crf": True,
        "model_weights_path":"seghead_weights/hcl_coco_p8.pth.tar",
        "linear_weights_path":"linear_classifier_weights/linear_classifier_coco_p8.pth.tar",
    },

}
selected_config = eval_config["hcl_p16_eval_pascal"]
# **********eval config**********
eval_model_name = selected_config["eval_model_name"]
assert eval_model_name in ["hcl"]
model_weights_path = selected_config["model_weights_path"]
linear_weights_path = selected_config["linear_weights_path"]
vit_patch_size = selected_config["vit_patch_size"]
model_check_point = "online_seg_head"#"momentum_seg_head"#"online_seg_head"
evaluate_data = selected_config["evaluate_data"]  # switch dataset here
coco_data_set = "full"
using_crf = selected_config["using_crf"]
seghead_type = "transformer_block"#mlp

if evaluate_data == "coco":
    inference_image_size = 320
    if coco_data_set == "full":
        n_classes = 27
    elif coco_data_set == "thing":
        n_classes = 12
    elif coco_data_set == "stuff":
        n_classes = 15
elif evaluate_data == "pascal":
    n_classes = 21
    inference_image_size = 448

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("-j", "--workers", default=32, type=int, metavar="N",
                    help="number of data loading workers (default: 32)")
parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N",
                    help="manual epoch number (useful on restarts)")
parser.add_argument("--classes", default=n_classes, type=int, metavar="N", help="number of total classes")
parser.add_argument("-b", "--batch-size", default=256, type=int, metavar="N",
                    help="mini-batch size (default: 256), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel")
parser.add_argument("--lr", "--learning-rate", default=0.1, type=float, metavar="LR", help="initial learning rate",
                    dest="lr")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument("--wd", "--weight-decay", default=0.0001, type=float, metavar="W",
                    help="weight decay (default: 0.)", dest="weight_decay")
parser.add_argument("-p", "--print-freq", default=5, type=int, metavar="N", help="print frequency (default: 10)")
parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
parser.add_argument("-e", "--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set")
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
parser.add_argument("--pretrained-linear", default=linear_weights_path, type=str,
                    help="path to moco pretrained checkpoint")
parser.add_argument("--patch-size", default=vit_patch_size, type=int, help="vit patch size")
parser.add_argument("--spatial-token-dim", default=384, type=int, help="spatial token dimension (default: 384)")
parser.add_argument("--seghead-type", default=seghead_type, type=str, help="alternative: transformer_block")
parser.add_argument("--inference-img-size", default=inference_image_size, type=int,
                    help="vit inference image size, 448 for pascal, 320 for other datasets")
parser.add_argument("--checkpoint-key", default=model_check_point, type=str,
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
            apply_dropout=False
        )
        output_dim = args.spatial_token_dim * 2#256
    else:
        model = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            pretrained=True,
            pretrain_weights=args.model_name,
            use_projector=False
        )
        # freeze all layers but the last classification layer of PSPNet.
        for name, param in model.named_parameters():
            param.requires_grad = False
                # print(name)
        output_dim = args.spatial_token_dim
    print(model)
    linear_classifier = nn.Conv2d(output_dim, args.classes, kernel_size=1)
    print(linear_classifier)
    linear_weights = get_linear_weights(args.pretrained_linear)
    msg_1 = linear_classifier.load_state_dict(linear_weights, strict=True)
    print(msg_1)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            linear_classifier.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        else:
            model.cuda()
            linear_classifier.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            linear_classifier = torch.nn.parallel.DistributedDataParallel(linear_classifier)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        linear_classifier = linear_classifier.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    # define loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255).cuda(args.gpu)
    cudnn.benchmark = True

    # Data loading code
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    # we scale mean value by 255 is because the values of image pixels will be from 0 - 255 after ToTensor.
    # This ToTensor is implemented by us, it won't rescale the value to 0 -1 like a standard ToTensor function.
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    if evaluate_data == "pascal":  # follow leopart protocol, 448 for both training and inference.
        val_transforms = eval_transforms.Compose([
            eval_transforms.Resize((args.inference_img_size, args.inference_img_size)),  # (448,448)
            eval_transforms.ToTensor(),
            eval_transforms.Normalize(mean=mean, std=std)])
    else:
        val_transforms = eval_transforms.Compose([
            eval_transforms.ResizeAndCenterCrop(args.inference_img_size),  # Resize smaller dimension to 320 pixels
            eval_transforms.ToTensor(),
            eval_transforms.Normalize(mean=mean, std=std)])

    if evaluate_data == "coco":
        val_dataset = EvalCoco(root_path="/workspace/coco_2017/", split="val",
                               transform=val_transforms, coarse_labels=True, data_set=coco_data_set, subset="iic_subset_val")
    elif evaluate_data == "pascal":#root: /workspace/VOCdevkit/VOC2012/
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
    eval_val = validate(val_loader, model,linear_classifier, criterion, args)

def validate(val_loader, model, linear_classifier, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    linear_semseg_meter = SemsegMeter(args.classes, val_loader.dataset.get_class_names(), ignore_index=255)

    progress = ProgressMeter(
        len(val_loader), [batch_time, losses], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()
    linear_classifier.eval()
    linear_metrics = UnsupervisedMetrics(
        "Linear_", n_classes, 0, False)
    with torch.no_grad():
        end = time.time()
        for i, (images, targets) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)
            # print(images.shape, targets.shape, targets.unique())
            # compute output
            print(images.shape)
            if eval_model_name == "hcl":
                output = model(images, image_size=args.inference_img_size ,original_size=False)
            else:
                output = model(images, original_size=False)# N x c x h x w
            # output = nn.functional.interpolate(output, size=(images.shape[2], images.shape[3]),
            #                           mode='bilinear', align_corners=False)
            linear_output = linear_classifier(output)
            linear_output = nn.functional.interpolate(linear_output, size=(images.shape[2], images.shape[3]),
                                                mode='bilinear',align_corners=False)
            loss = criterion(linear_output, targets)
            # linear_output = linear_output.argmax(1).cuda()
            #apply crf
            if using_crf:
                linear_output = torch.log_softmax(linear_output, dim=1)
                linear_output = batched_crf(images, linear_output).argmax(1).cuda()
            # measure accuracy and record loss
            losses.update(loss.item(), images.size(0))
            linear_semseg_meter.update(linear_output, targets)
            linear_metrics.update(linear_output, targets)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
    eval_results = linear_semseg_meter.return_score(verbose=True)
    eval_metrics = get_single_metric(linear_metrics)
    s=time_log()
    s += f" -------------------after crf ---------------------\n"
    for metric_k, metric_v in eval_metrics.items():
        s += f"[after crf] {metric_k} : {metric_v:.2f}\n"
    print(s)
    return eval_results

if __name__ == "__main__":
    main()
