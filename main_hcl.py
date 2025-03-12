import argparse
import builtins
import os
import random
import time
import warnings
import faiss
import numpy as np
import build as build
from utils.utils import perform_faiss_kmeans, broadcast_kmeans_seed, get_faiss_idx, adjust_learning_rate, concentration_estimation
from utils.eval_utils import AverageMeter, ProgressMeter
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from data.data_module import TrainDataset

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("-j", "--workers", default=32, type=int, metavar="N",
                    help="number of data loading workers (default: 32)")
parser.add_argument("--epochs", default=200, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
parser.add_argument("-b", "--batch-size", default=64, type=int, metavar="N",
                    help="mini-batch size (default: 64), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel")
parser.add_argument("--lr", "--learning-rate", default=0.0001, type=float, metavar="LR", help="initial learning rate",
                    dest="lr")
parser.add_argument("--schedule", default=[3,], nargs="*", type=int,
                    help="learning rate schedule (which epoch to drop lr by 10x)")  # coco[5,20]other[3,10]
parser.add_argument("--wd", "--weight-decay", default=0.04, type=float, metavar="W",
                    help="weight decay (default: 1e-4)", dest="weight_decay")
parser.add_argument("-p", "--print-freq", default=10, type=int, metavar="N", help="print frequency (default: 10)")
parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
parser.add_argument("--world-size", default=-1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=-1, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://224.66.41.62:23456", type=str,
                    help="url used to set up distributed training", )
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--seed", default=7, type=int, help="seed for initializing training. ")
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument("--multiprocessing-distributed", action="store_true",
                    help="Use multi-processing distributed training to launch "
                         "N processes per node, which has N GPUs. This is the "
                         "fastest way to use PyTorch for either single node or "
                         "multi node data parallel training",
                    )
# specific configs:
parser.add_argument("--image-size", default=224, type=int, help="image size for training")
parser.add_argument("--outputdim", default=256, type=int, help="feature dimension (default: 256)")
parser.add_argument("--spatial-token-dim", default=384, type=int, help="spatial token dimension (default: 384)")
parser.add_argument("--vit-patchsize", default=16, type=int, help="vit patch size")
parser.add_argument("--seghead-type", default="transformer_block", type=str, help="alternative: transformer_block")
parser.add_argument("--k-foreground", default=500, type=int, help="k clusters")
parser.add_argument("--k-background", default=500, type=int, help="k clusters")
parser.add_argument("--sample-percentage", default=0.3, type=float,#0.3
                    help="percentage of spatial tokens sampled from each image.")
parser.add_argument("--embedding-replace-percentage", default=0.5,
                    help="percentage of patch embedding in the bank to be replaced with the new ones.", )
parser.add_argument('--dataset', default='pascal', type=str)#coco_10k coco_iic_subset_train
parser.add_argument("--m", default=0.996, type=float, help=" momentum of updating momentum seg head (default: 0.999)", )
parser.add_argument("--nce-tem", default=0.07, type=float, help="nce softmax temperature (default: 0.07)")
parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")
parser.add_argument('--arch', default='vit_small', type=str, choices=['vit_tiny', 'vit_small', 'vit_base'],
                    help="Name of architecture to train. For quick experiments with ViTs,we recommend using vit_tiny or vit_small.")

# for pascal
# k:2000, sample_percentage:0.3, replace_percentage 0.5, nce tem 0.07, train for 10 epochs, schedule [3,]
# for coco
# k:2000, sample_percentage:0.3, replace_percentage 0.5, nce tem 0.07, train for 10 epochs, schedule [5,]
# for cityscapes
# k:2000, sample_percentage:0.6, replace_percentage 0.5, nce tem 0.05, train for 10 epochs, schedule [8,12]
def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
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
    # print("ngpus_per_node", ngpus_per_node)
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
    # print("=> creating model '{}'".format(args.arch))
    model = build.HCL(
        max_epoch=args.epochs,
        vit_arch=args.arch,
        patch_size=args.vit_patchsize,
        output_dim=args.outputdim,
        m=args.m,
        nce_tm=args.nce_tem,
        sp_token_dim=args.spatial_token_dim,
        sample_percentage=args.sample_percentage,
        embedding_replace_percentage=args.embedding_replace_percentage,
        image_size=args.image_size,
        seg_head_type=args.seghead_type,
        apply_dropout=False,
        use_predictor=True
    )
    print(model)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    # criterion_ce = nn.CrossEntropyLoss().cuda(args.gpu)
    mse_loss = nn.MSELoss().cuda(args.gpu)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        args.lr,
        weight_decay=args.weight_decay,
    )
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            keys_to_be_deleted = [key for key in checkpoint["state_dict"].keys() if key == "module.embedding_queue"]
            del checkpoint["state_dict"][
                keys_to_be_deleted[0]]  # remove this queue, as different dataset may have different size.
            model.load_state_dict(checkpoint["state_dict"], strict=False)

            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    # data loading module
    inv_list = ['blur', 'grey', 'brightness', 'contrast', 'saturation', 'hue']
    eqv_list = ['h_flip', 'v_flip', 'random_crop']
    if args.dataset == "pascal":
        root_path = "/workspace/VOCdevkit/VOC2012/"
        split = "train_aug"
    elif args.dataset == "coco" or args.dataset == "coco_10k" or args.dataset == "coco_iic_subset_train":
        root_path = "/workspace/coco_2017/"
        split = "train"
    elif args.dataset == "imagenet_100":
        root_path = "/workspace/imagenet_100/"
        split = "train"
    elif args.dataset == "cityscapes":
        root_path = "/workspace/cityscapes/"
        split = "train_extra"
    elif args.dataset == "coco10k&pascal":
        root_path = ["/workspace/coco_2017/", "/workspace/VOCdevkit/VOC2012/"]
        split = ["train", "train_aug"]

    train_dataset = TrainDataset(args.dataset, root_path, split, inv_list, eqv_list,
                                 res1=args.image_size, res2=args.image_size * 2, scale=(0.8, 1))# for other dataset (0.5, 1) for cityscape (0.8, 1)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        # drop_last=True,
    )
    faiss_idx_foreground = get_faiss_idx(args.world_size, args.outputdim)
    faiss_idx_background = get_faiss_idx(args.world_size, args.outputdim)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        print("build global_patch_embeddings", epoch)
        model.module.build_global_patch_embeddings(train_loader, args, epoch)
        print("start kmeans")
        # seed generated at rank 0, and then broadcast to all other ranks, so kmeans on each rank has the same seed
        seed = broadcast_kmeans_seed(100, args.gpu)
        faiss_idx_foreground.reset()
        faiss_idx_background.reset()
        kmeans_foreground = perform_faiss_kmeans(model.module.global_patch_embedding_bank_foreground, args.k_foreground, seed, faiss_idx_foreground)
        kmeans_background = perform_faiss_kmeans(model.module.global_patch_embedding_bank_background, args.k_background, seed, faiss_idx_background)
        # kmeans, kmeans_idx = perform_faiss_kmeans_on_all_ranks(global_patch_embeddings, args, k, seed)
        print("finish kmeans")
        # if epoch==args.start_epoch:
        # foreground_con = concentration_estimation(faiss_idx_foreground, model.module.global_patch_embedding_bank_foreground)
        # background_con = concentration_estimation(faiss_idx_background, model.module.global_patch_embedding_bank_background)
        # print("estimation", foreground_con, background_con, "+++++++++++++++")
        # model.module.update_nce(foreground_con, background_con)

        centroids_foreground = torch.tensor(
            faiss.vector_float_to_array(kmeans_foreground.centroids).reshape(args.k_foreground, args.outputdim)).cuda(
            args.gpu, non_blocking=True)  # convert from numpy to torch
        centroids_background = torch.tensor(
            faiss.vector_float_to_array(kmeans_background.centroids).reshape(args.k_background, args.outputdim)).cuda(
            args.gpu, non_blocking=True)  # convert from numpy to torch
        centroids_foreground = nn.functional.normalize(centroids_foreground, dim=1, p=2)
        centroids_background = nn.functional.normalize(centroids_background, dim=1, p=2)
        train(centroids_foreground,centroids_background, faiss_idx_foreground, faiss_idx_background,train_loader, model, optimizer, epoch, args, mse_loss)
        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            if epoch in [0, 4, 9, 14, 19, 24, 29]:
                online_seg_head_state_dict = model.module.online_seg_head.state_dict()
                momentum_seg_head_state_dict = model.module.momentum_seg_head.state_dict()
                #backbone_dict = model.module.vit_featurizer.state_dict()
                save_checkpoint(# only save weights from seg head.
                    {
                        "epoch": epoch + 1,
                        "arch": "vit_small",
                        #"backbone": backbone_dict,
                        "online_seg_head": online_seg_head_state_dict,
                        "momentum_seg_head": momentum_seg_head_state_dict,
                        "centroids_foreground":centroids_foreground,
                        "centroids_background":centroids_background,
                        "optimizer": optimizer.state_dict(),
                    },
                    filename="hcl_epoch" + str(epoch) + ".pth.tar"
                )



def train(centroids_foreground,centroids_background, faiss_idx_foreground, faiss_idx_background, train_loader, model, optimizer, epoch, args, mse_loss):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch),
    )
    # switch to train mode
    model.train()
    train_loader.dataset.reshuffle()
    end = time.time()
    for i, (indice, image1, image2) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # it = len(train_loader) * epoch + i  # global training iteration
        # adjust_learning_rate(optimizer, learning_rate_schedule, it)
        if args.gpu is not None:
            image1 = image1.cuda(args.gpu, non_blocking=True)
            image2 = image2.cuda(args.gpu, non_blocking=True)
            indice = indice.cuda(args.gpu, non_blocking=True)
        # compute output
        #image1 = train_loader.dataset.transform_eqv(indice, image1, mode='bilinear')
        #image2 = train_loader.dataset.transform_eqv(indice, image2, mode='bilinear')
        positive_logits_foreground, logits_foreground, positive_logits_background,  logits_background, q_predictor, k= model(image1, image2, centroids_foreground,centroids_background, faiss_idx_foreground, faiss_idx_background)
        #, q_predictor, k
        #print("loss++++++++",positive_logits_foreground.shape, logits_foreground.shape,positive_logits_background.shape, logits_background.shape)
        # formulate loss
        loss_foreground = -torch.log(torch.div(positive_logits_foreground, logits_foreground)).mean()  # N
        loss_background = -torch.log(torch.div(positive_logits_background, logits_background)).mean()  # N
        #loss_selected_foreground = -torch.log(torch.div(positive_logits_selected_foreground, logits_selected_foreground)).mean()  # N

        loss_mse = mse_loss(q_predictor, k)
        #loss_mes2 = mse_loss(q_foreground, target_foreground)
        loss = (loss_foreground+loss_background) * 0.5 + loss_mse*0.5# + loss_mes2*0.5#loss_foreground
        losses.update(loss.item(), 1)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)


if __name__ == "__main__":
    main()
