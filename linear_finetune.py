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
from utils.eval_utils import SemsegMeter, AverageMeter, ProgressMeter, ClusterLookup, UnsupervisedMetrics, get_single_metric, time_log
from utils.utils import adjust_learning_rate
from data.data_module import EvalPascal, EvalCoco
import vit.vision_transformer as vits


finetune_config = {
    "hcl_p16_linear_finetune_pascal":{
        "finetune_model_name": "hcl",
        "vit_patch_size":16,
        "lr_schedule":[4,],
        "finetune_epochs":10,
        "finetune_data":"pascal",
        "model_check_point":"online_seg_head",
        "model_weights_path": "seghead_weights/hcl_pascal_p16.pth.tar",
    },

    "hcl_p8_linear_finetune_pascal":{
        "finetune_model_name": "hcl",
        "vit_patch_size":8,
        "lr_schedule":[4,],
        "finetune_epochs":10,
        "finetune_data":"pascal",
        "model_check_point":"online_seg_head",
        "model_weights_path":"seghead_weights/hcl_pascal_p8.pth.tar"
    },

    "hcl_p16_linear_finetune_coco":{
        "finetune_model_name": "hcl",
        "vit_patch_size":16,
        "lr_schedule":[1,4],
        "finetune_epochs":10,
        "finetune_data":"coco",
        "model_check_point":"online_seg_head",
        "model_weights_path":"seghead_weights/hcl_coco_p16.pth.tar",
    },
    "hcl_p8_linear_finetune_coco":{
        "finetune_model_name": "hcl",
        "vit_patch_size":8,
        "lr_schedule":[1,4],
        "finetune_epochs":10,
        "finetune_data":"coco",
        "model_check_point":"online_seg_head",
        "model_weights_path":"seghead_weights/hcl_coco_p8.pth.tar",
    },
}

selected_config = finetune_config["hcl_p16_linear_finetune_pascal"]
finetune_model_name = selected_config["finetune_model_name"] # "leopart", "dino", "croc"
assert finetune_model_name in ["hcl"]
lr_schedule = selected_config["lr_schedule"]# [4,14] for pascal and coco [12,16]
model_weights_path = selected_config["model_weights_path"]
vit_patch_size = selected_config["vit_patch_size"]
model_check_point = selected_config["model_check_point"]
finetune_data = selected_config["finetune_data"]  # switch dataset here
coco_data_set = "full"
seghead_type = "transformer_block"#mlp

if finetune_data == "coco":
    inference_image_size = 320
    training_image_size = 448# five crop, otherwise 448
    if coco_data_set == "full":
        n_classes = 27
    elif coco_data_set == "thing":
        n_classes = 12
    elif coco_data_set == "stuff":
        n_classes = 15
elif finetune_data == "pascal":
    n_classes = 21
    inference_image_size = 448
    training_image_size = 448

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
parser.add_argument("--schedule", default=lr_schedule, nargs="*", type=int,
                    help="learning rate schedule (when to drop lr by a ratio)")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument("--wd", "--weight-decay", default=0.0001, type=float, metavar="W",
                    help="weight decay (default: 0.)", dest="weight_decay")
parser.add_argument("-p", "--print-freq", default=10, type=int, metavar="N", help="print frequency (default: 10)")
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
parser.add_argument("--patch-size", default=vit_patch_size, type=int, help="vit patch size")
parser.add_argument("--spatial-token-dim", default=384, type=int, help="spatial token dimension (default: 384)")
parser.add_argument("--seghead-type", default=seghead_type, type=str, help="alternative: transformer_block")
parser.add_argument("--inference-img-size", default=inference_image_size, type=int,
                    help="vit inference image size, 448 for pascal, 320 for other datasets")
parser.add_argument("--training-img-size", default=training_image_size, type=int,
                    help="vit training image size, 448 for pascal, 224 for other datasets")
parser.add_argument("--checkpoint-key", default=model_check_point, type=str,
                    help='Key to use in the checkpoint (example: "online_seg_head")')
parser.add_argument('--arch', default='vit_small', type=str, choices=['vit_tiny', 'vit_small', 'vit_base'],
                    help="Name of architecture to train. For quick experiments with ViTs,we recommend using vit_tiny or vit_small.")
parser.add_argument("--model-name", default=finetune_model_name, type=str, help='eval model name')
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
        )# weights are frozen automatically.
        output_dim = args.spatial_token_dim*2
    else:
        model = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            pretrained=True,
            pretrain_weights=args.model_name,
            use_projector=False,
        )
        output_dim = args.spatial_token_dim
        # freeze all layers but the last classification layer of PSPNet.
        for name, param in model.named_parameters():
            param.requires_grad = False
            # print(name)
    print(model)
    linear_classifier = nn.Conv2d(output_dim, args.classes, kernel_size=1)
    print(linear_classifier)
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
            linear_classifier = torch.nn.parallel.DistributedDataParallel(
                linear_classifier, device_ids=[args.gpu]
            )
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


    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    print("length of model parameters required gradient: ", len(model_parameters))
    assert len(model_parameters) == 0
    linear_classifier_parameters = list(filter(lambda p: p.requires_grad, linear_classifier.parameters()))
    print("length of linear classifier parameters required gradient: ", len(linear_classifier_parameters))
    #linear_classifier_optimizer = torch.optim.Adam(linear_classifier_parameters, args.lr)
    linear_classifier_optimizer = torch.optim.SGD(
        linear_classifier_parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    cudnn.benchmark = True

    # Data loading code
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    # we scale mean value by 255 is because the values of image pixels will be from 0 - 255 after ToTensor.
    # This ToTensor is implemented by us, it won't rescale the value to 0 -1 like a standard ToTensor function.
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    if finetune_data == "pascal":  # follow leopart protocol, 448 for both training and inference.
        train_transforms = eval_transforms.Compose([
            eval_transforms.RandScale([0.75, 1.25]),
            # 0.75, 1.25 adpted from mask contrast, pspnet original ones are 0.5, 2
            eval_transforms.RandRotate([-10, 10], padding=mean, ignore_label=255),
            eval_transforms.RandomGaussianBlur(),
            eval_transforms.RandomHorizontalFlip(),
            eval_transforms.Crop([args.training_img_size, args.training_img_size], crop_type='rand', padding=mean,
                                 ignore_label=255),
            eval_transforms.ToTensor(),
            eval_transforms.Normalize(mean=mean, std=std)])
        val_transforms = eval_transforms.Compose([
            eval_transforms.Resize((args.inference_img_size, args.inference_img_size)),  # (448,448)
            eval_transforms.ToTensor(),
            eval_transforms.Normalize(mean=mean, std=std)])
    elif finetune_data == "coco":
        train_transforms = eval_transforms.Compose([
            eval_transforms.RandScale([0.75, 1.25]),
            # 0.75, 1.25 adpted from mask contrast, pspnet original ones are 0.5, 2
            eval_transforms.RandRotate([-10, 10], padding=mean, ignore_label=255),
            eval_transforms.RandomGaussianBlur(),
            eval_transforms.RandomHorizontalFlip(),
            eval_transforms.Crop([args.training_img_size, args.training_img_size], crop_type='rand', padding=mean,
                                 ignore_label=255),
            eval_transforms.ToTensor(),
            eval_transforms.Normalize(mean=mean, std=std)])
        val_transforms = eval_transforms.Compose([
            eval_transforms.ResizeAndCenterCrop(args.inference_img_size),  # (448,448)
            eval_transforms.ToTensor(),
            eval_transforms.Normalize(mean=mean, std=std)])

    if finetune_data == "coco":
        train_dataset = EvalCoco(root_path="/workspace/coco_2017/", split="train",
                                 transform=train_transforms, coarse_labels=True,
                                 data_set=coco_data_set, subset="iic_subset_train")#iic_subset_train, cocostuff10k
        val_dataset = EvalCoco(root_path="/workspace/coco_2017/", split="val",
                               transform=val_transforms, coarse_labels=True,
                               data_set=coco_data_set, subset="iic_subset_val")#, subset="iic_subset_val"
    elif finetune_data == "pascal":
        train_dataset = EvalPascal(root_path="/workspace/VOCdevkit/VOC2012/", split="train_aug",
                                   transform=train_transforms)
        val_dataset = EvalPascal(root_path="/workspace/VOCdevkit/VOC2012/", split="val", transform=val_transforms)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler
    )

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(linear_classifier_optimizer, epoch, args)

        # train for one epoch
        eval_train = train(train_loader, model, linear_classifier, criterion, linear_classifier_optimizer, epoch, args)

        # evaluate on validation set
        linear_miou = validate(val_loader, model, linear_classifier, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = linear_miou > best_iou
        best_iou = max(linear_miou, best_iou)

        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch":args.model_name,
                    "linear_classifier": linear_classifier.state_dict(),
                    "best_iou": best_iou,
                    #"optimizer": optimizer.state_dict(),
                },
                is_best,
                "linear_classifier_best"+str(epoch)+".pth.tar"
            )


def train(train_loader, model, linear_classifier, criterion, linear_classifier_optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    linear_losses = AverageMeter('Linear Loss', ':.4e')
    cluster_losses = AverageMeter('Cluster Loss', ':.4e')
    semseg_meter = SemsegMeter(args.classes, train_loader.dataset.get_class_names(), ignore_index=255)
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, linear_losses, cluster_losses],
                             prefix="Epoch: [{}]".format(epoch))
    model.train()
    linear_classifier.train()

    end = time.time()
    for i, (images, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)
        # compute output
        with torch.no_grad():
            if args.model_name == "hcl":
                output = model(images, image_size=args.training_img_size, original_size=False)
            else:
                output = model(images, original_size=False)
        linear_output = linear_classifier(output)
        linear_output = nn.functional.interpolate(linear_output, size=(images.shape[2], images.shape[3]), mode='bilinear',
                                                     align_corners=False)
        linear_loss = criterion(linear_output, targets)
        # measure accuracy and record loss
        semseg_meter.update(torch.argmax(linear_output, dim=1), targets)
        linear_losses.update(linear_loss.item(), images.size(0))

        # compute gradient and do SGD step
        linear_classifier_optimizer.zero_grad()

        linear_loss.backward()
        linear_classifier_optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    eval_results = semseg_meter.return_score(verbose=True)
    return eval_results


def validate(val_loader, model, linear_classifier, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    semseg_meter = SemsegMeter(args.classes, val_loader.dataset.get_class_names(), ignore_index=255)
    linear_metrics = UnsupervisedMetrics(
        "Linear_", n_classes, 0, False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()
    linear_classifier.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, targets) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)
            # print(images.shape, targets.shape, targets.unique())
            # compute output
            if args.model_name == "mogoseg":
                output = model(images, image_size=args.inference_img_size,original_size=False)
            else:
                output = model(images,original_size=False)
            linear_output = linear_classifier(output)
            linear_output = nn.functional.interpolate(linear_output, size=(images.shape[2], images.shape[3]),
                                                      mode='bilinear',align_corners=False)
            loss = criterion(linear_output, targets)
            # apply crf

            # measure accuracy and record loss
            losses.update(loss.item(), images.size(0))
            semseg_meter.update(torch.argmax(linear_output, dim=1), targets)
            linear_metrics.update(torch.argmax(linear_output, dim=1), targets)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
    eval_results = semseg_meter.return_score(verbose=True)
    linear_results = get_single_metric(linear_metrics)
    s = time_log()
    s += f" -------------------after crf ---------------------\n"

    for metric_k, metric_v in linear_results.items():
        s += f"[after crf] {metric_k} : {metric_v:.2f}\n"
    print(s)
    return linear_results["Linear_mIoU"]


def save_checkpoint(state, is_best, filename="lc_best.pth.tar"):
    if is_best:
        torch.save(state, filename)
        print("saved the model weights with best miou")


if __name__ == "__main__":
    main()
