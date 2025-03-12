import torch
import torch.nn as nn
from utils.utils import trunc_normal_
import vit.pm_vit as vits
import math
import os
from utils.utils import get_random_num, broadcast_random_idx, get_binary_attn_maps, concat_all_gather_for_unequal_length

class SegHead(nn.Module):
    def __init__(self, in_channels, use_projector=True,
                 n_layers_projection_head=3, hidden_dim=2048, proj_output_dim=256,
                 output_dim=768, l2_norm=True, seg_head_type="mlp", apply_dropout=False):
        super(SegHead, self).__init__()
        self.use_projector = use_projector
        self.output_dim = output_dim
        self.seg_head_output_dim = self.output_dim
        self.norm = nn.LayerNorm(self.output_dim)
        self.norm_2 = nn.LayerNorm(self.seg_head_output_dim)
        self.multi_scale_fuser = nn.Linear(in_channels, self.output_dim)
        self.l2_norm = l2_norm
        self.apply_dropout = apply_dropout
        # Seghead
        if seg_head_type == "transformer_block":
            self.seg_head = vits.Block(
                dim=self.output_dim, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm)
            print("++++++++++++++++seghead is using transformer block++++++++++++++++")
        elif seg_head_type == "mlp":
            seg_head_layers = [nn.Linear(self.output_dim, hidden_dim)]
            seg_head_layers.append(nn.GELU())
            seg_head_layers.append(nn.Linear(hidden_dim, hidden_dim))
            seg_head_layers.append(nn.GELU())
            seg_head_layers.append(nn.Linear(hidden_dim, self.seg_head_output_dim))
            self.seg_head = nn.Sequential(*seg_head_layers)
            print("++++++++++++++++seghead is using mlp++++++++++++++++")
        # Dropout
        if self.apply_dropout:
            self.dropout = torch.nn.Dropout1d(p=.1)
        if self.use_projector:
            # Construct projection head,
            nlayers = max(n_layers_projection_head, 1)
            if nlayers == 1:
                self.projection_head = nn.Linear(self.seg_head_output_dim, proj_output_dim)
            else:
                layers = [nn.Linear(self.seg_head_output_dim, hidden_dim)]
                layers.append(nn.GELU())
                for _ in range(nlayers - 2):
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
                    layers.append(nn.GELU())
                layers.append(nn.Linear(hidden_dim, proj_output_dim))
                self.projection_head = nn.Sequential(*layers)
            print("++++++++++++++++seghead is using projector++++++++++++++++")
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_backbone(self, x):
        x = self.multi_scale_fuser(x)
        # N x num_tokens x output_dim, PM-ViT produces concatenated feature, need to reduce dimension.
        x = self.norm(x)
        x = self.seg_head(x)  # N x num_tokens x output_dim
        x = self.norm_2(x)
        return x

    def forward(self, x, image_size, original_size=True):  # x : # N x num_tokens x in_channels
        x = self.multi_scale_fuser(x)
        # N x num_tokens x output_dim, PM-ViT produces concatenated feature, need to reduce dimension.
        x = self.norm(x)
        if self.apply_dropout:
            x = self.dropout(x)
        x = self.seg_head(x)  # N x num_tokens x output_dim
        x = self.norm_2(x)
        if self.use_projector:
            x = self.projection_head(x)  # N x num_tokens x proj_output_dim
            if self.l2_norm:
                x = nn.functional.normalize(x, dim=2, p=2)  # normlize on emdim dimention.
        x = x.permute((0, 2, 1))  # N x output_dim x num_tokens
        fea_dim = int(math.sqrt(x.shape[2]))
        x = torch.reshape(x, (x.shape[0], x.shape[1], fea_dim, fea_dim))
        if original_size:
            x = nn.functional.interpolate(x, size=(image_size, image_size), mode='bilinear',
                                          align_corners=False)
        return x


class HCL(nn.Module):
    """
    Build a HCL model with: a PM-ViT as backbone, an online seg head, a momentum seg head
    """
    def __init__(self, max_epoch, vit_arch, patch_size=16, output_dim=128, m=0.999, nce_tm=0.07,
                 sp_token_dim=384, sample_percentage=0.1, embedding_replace_percentage=0.5,
                 image_size=224, seg_head_type="mlp", hidden_dim=2048, apply_dropout=False,
                 use_predictor=False):
        super(HCL, self).__init__()
        self.max_epoch = max_epoch
        self.m = m
        self.nce_tm = nce_tm
        self.patch_size = patch_size
        self.sp_token_dim = sp_token_dim
        self.in_channels = self.sp_token_dim * 4 if patch_size == 16 else self.sp_token_dim * 5
        self.num_patches = int((image_size / patch_size) ** 2)
        self.output_dim = output_dim
        self.sample_percentage = sample_percentage
        self.embedding_replace_percentage = embedding_replace_percentage
        self.use_predictor = use_predictor
        self.image_size = image_size
        self.attn_threshold=0.7
        # create the encoders
        # num_classes is the output fc dimension
        self.vit_featurizer = vits.__dict__[vit_arch](patch_size=patch_size)
        #         self.dropout = torch.nn.Dropout2d(p=.1)
        self.online_seg_head = SegHead(in_channels=self.in_channels,#sp_token_dim=self.sp_token_dim,
                                       proj_output_dim=self.output_dim, seg_head_type=seg_head_type,
                                       hidden_dim=hidden_dim, apply_dropout=apply_dropout)
        self.momentum_seg_head = SegHead(in_channels=self.in_channels,#sp_token_dim=self.sp_token_dim,
                                         proj_output_dim=self.output_dim, seg_head_type=seg_head_type,
                                         hidden_dim=hidden_dim, apply_dropout=apply_dropout)
        if self.use_predictor:
            predictor_layers = [nn.Linear(self.output_dim, hidden_dim)]
            predictor_layers.append(nn.GELU())
            predictor_layers.append(nn.Linear(hidden_dim, self.output_dim))
            self.predictor = nn.Sequential(*predictor_layers)
            print("+++++++++++++++++MgGC is using predictor+++++++++++++++++++++")
        # Freeze momentum seghead
        for param_online, param_momentum in zip(
                self.online_seg_head.parameters(), self.momentum_seg_head.parameters()
        ):
            param_momentum.data.copy_(param_online.data)  # initialize
            param_momentum.requires_grad = False  # not update by gradient
        # Freeze PM-ViT backbone
        for name, param in self.vit_featurizer.named_parameters():
            param.requires_grad = False
        # extra_pe_layers_weights = [
        # "patch_embed_2.proj.weight",
        # "patch_embed_2.proj.bias",
        # "patch_embed_3.proj.weight",
        # "patch_embed_3.proj.bias"
        # ]
        # for name, param in self.vit_featurizer.named_parameters():# freeze backbone but not two extra pe layers
        #     if name not in extra_pe_layers_weights:
        #         param.requires_grad = False
        self.register_buffer("global_patch_embedding_bank_foreground",
                             torch.randn(10560 * (int(self.num_patches * self.sample_percentage)), self.output_dim))
        self.register_buffer("global_patch_embedding_bank_background",
                             torch.randn(10560 * (int(self.num_patches * self.sample_percentage)), self.output_dim))
    @torch.no_grad()
    def _momentum_update(self):
        """
        Momentum update of the momentum seghead
        """
        current_m = self.m
        for param_online, param_momentum in zip(
                self.online_seg_head.parameters(), self.momentum_seg_head.parameters()
        ):
            param_momentum.data = param_momentum.data * current_m + param_online.data * (1.0 - current_m)


    @torch.no_grad()
    def build_global_patch_embeddings(self, dataloader, args, epoch):
        global_patch_embeddings_foreground = []
        global_patch_embeddings_background = []
        print("epoch", epoch, "start to sample,", self.sample_percentage, "% of tokens")
        for i, (indice, image1, image2) in enumerate(dataloader):
            if i % 500 == 0:
                print(i, image2.shape)
            if args.gpu is not None:
                image2 = image2.cuda(args.gpu, non_blocking=True)
            batch_size = image2.shape[0]
            # Use momentum seg head to generate patch embedding
            raw_features, attns = self.vit_featurizer(image2, return_attn=True)  # attns: N x 6 x 197 x 197
            attns = get_binary_attn_maps(attns, threshold=self.attn_threshold)  # N x 1 x 14 x 14
            attns = torch.reshape(attns, (attns.shape[0], -1))  # N x 196
            spatial_tokens = self.momentum_seg_head(raw_features, image_size=self.image_size,
                                                    original_size=False)  # batchsize x dim x 14 x 14
            spatial_tokens = spatial_tokens.permute((0, 2, 3, 1))  # batchsize x 14 x 14 x dim
            spatial_tokens = torch.reshape(spatial_tokens, (
                spatial_tokens.shape[0], spatial_tokens.shape[1] * spatial_tokens.shape[2],
                spatial_tokens.shape[3]))  # batchsize x 196 x dim
            num_of_patch = spatial_tokens.shape[1]
            foreground_random_idx_list = get_random_num(num_of_patch, batch_size, self.sample_percentage, "foreground", attns)
            background_random_idx_list = get_random_num(num_of_patch, batch_size, self.sample_percentage, "background",attns)
            for j in range(batch_size):
                global_patch_embeddings_foreground.append(spatial_tokens[j][foreground_random_idx_list[j]])
                global_patch_embeddings_background.append(spatial_tokens[j][background_random_idx_list[j]])
            del image2
        global_patch_embeddings_foreground = torch.cat(global_patch_embeddings_foreground, dim=0)
        global_patch_embeddings_background = torch.cat(global_patch_embeddings_background, dim=0)
        global_patch_embeddings_foreground = concat_all_gather_for_unequal_length(global_patch_embeddings_foreground)
        global_patch_embeddings_background = concat_all_gather_for_unequal_length(global_patch_embeddings_background)

        print("global_patch_embedding_bank_foreground shape", self.global_patch_embedding_bank_foreground.shape)
        print("global_patch_embedding_bank_background shape", self.global_patch_embedding_bank_background.shape)
        print(self.embedding_replace_percentage)
        if epoch == 0:
            self.global_patch_embedding_bank_foreground = global_patch_embeddings_foreground
            self.global_patch_embedding_bank_background = global_patch_embeddings_background
            print("+++++++++epoch0+++++++++++")
        elif self.embedding_replace_percentage == 1:
            self.global_patch_embedding_bank_foreground = global_patch_embeddings_foreground
            self.global_patch_embedding_bank_background = global_patch_embeddings_background
            print("embedding_replace_percentage")
        else:
            len_of_bank_foreground = self.global_patch_embedding_bank_foreground.shape[0]
            len_of_bank_background = self.global_patch_embedding_bank_background.shape[0]
            len_of_embeddings_foreground = global_patch_embeddings_foreground.shape[0]
            len_of_embeddings_background = global_patch_embeddings_background.shape[0]
            if len_of_bank_foreground>=len_of_embeddings_foreground:
                random_idx_foreground = broadcast_random_idx(len_of_embeddings_foreground, self.embedding_replace_percentage, args.gpu)
            else:
                random_idx_foreground = broadcast_random_idx(len_of_bank_foreground, self.embedding_replace_percentage, args.gpu)
            if len_of_bank_background>=len_of_embeddings_background:
                random_idx_background = broadcast_random_idx(len_of_embeddings_background, self.embedding_replace_percentage, args.gpu)
            else:
                random_idx_background = broadcast_random_idx(len_of_bank_background, self.embedding_replace_percentage, args.gpu)
            # randomly update n% of embedding bank
            self.global_patch_embedding_bank_foreground[random_idx_foreground] = global_patch_embeddings_foreground[random_idx_foreground]
            self.global_patch_embedding_bank_background[random_idx_background] = global_patch_embeddings_background[random_idx_background]
            print("Updated ", self.embedding_replace_percentage * 100, "% of embeddings")
        print("global_patch_embedding_bank_foreground shape", self.global_patch_embedding_bank_foreground.shape)
        print("global_patch_embedding_bank_background shape", self.global_patch_embedding_bank_background.shape)

    def forward(self, img_q, img_k, centroids_foreground,centroids_background, faiss_idx_foreground, faiss_idx_background):
        batch_size = img_q.shape[0]
        q = self.online_seg_head(self.vit_featurizer(img_q), image_size=self.image_size,
                                 original_size=False)  # queries: N x dim x 14 x 14
        q = q.permute((0, 2, 3, 1))  # N x 14 x 14 x dim
        q = torch.reshape(q, (batch_size * q.shape[1] * q.shape[2], q.shape[3]))  # 196N x dim
        if self.use_predictor:
            q_predictor = self.predictor(q)
            q_predictor = nn.functional.normalize(q_predictor, dim=1, p=2)
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update()  # update the momentum seg head
            k_raw_features, k_attns = self.vit_featurizer(img_k, return_attn=True)
            k_attns = get_binary_attn_maps(k_attns, threshold=self.attn_threshold)  # N x 1 x 14 x 14
            k_attns = torch.reshape(k_attns, (k_attns.shape[0] * k_attns.shape[2] * k_attns.shape[3],))  # 196N
            foreground_idx = torch.argwhere(k_attns == 1).squeeze().tolist()
            background_idx = torch.argwhere(k_attns == 0).squeeze().tolist()
            k = self.momentum_seg_head(k_raw_features, image_size=self.image_size,
                                       original_size=False)  # keys: N x dim x 14 x 14
            k = k.permute((0, 2, 3, 1))  # N x 14 x 14 x dim
            k = torch.reshape(k, (batch_size * k.shape[1] * k.shape[2], k.shape[3]))  # 196N x dim
            k_foreground = k[foreground_idx]
            k_background = k[background_idx]
            cluster_ids_foreground = faiss_idx_foreground.search(k_foreground.cpu().numpy().astype('float32'), 1)[1].squeeze()
            cluster_ids_background = faiss_idx_background.search(k_background.cpu().numpy().astype('float32'), 1)[1].squeeze()
            k_centroid_foreground = centroids_foreground[cluster_ids_foreground]
            k_centroid_background = centroids_background[cluster_ids_background]

        positive_logits_foreground = torch.exp(torch.einsum("nc,nc->n", [k_centroid_foreground, q[foreground_idx]]) / self.nce_tm)  # a
        positive_logits_background = torch.exp(torch.einsum("nc,nc->n", [k_centroid_background, q[background_idx]]) / self.nce_tm)  # 196N-a
        # centroids = torch.cat((centroids_foreground, centroids_background), dim=0)
        # print("centroids++++++++++++", centroids.shape)
        # print(k_centroid_foreground.shape, k_centroid_background.shape)
        logits_foreground = torch.exp(torch.einsum("nc,mc->nm", [q[foreground_idx],
                                                      centroids_foreground]) / self.nce_tm)  # 196N x dim, k x dim ->  196N x k, k: number of clusters
        logits_background = torch.exp(torch.einsum("nc,mc->nm", [q[background_idx],
                                                                 centroids_background]) / self.nce_tm)
        logits_foreground = torch.sum(logits_foreground, dim=1)  # 196N
        logits_background = torch.sum(logits_background, dim=1)  # 196N
        if self.use_predictor:
            return positive_logits_foreground, logits_foreground, positive_logits_background,  logits_background, q_predictor, k
        return positive_logits_foreground, logits_foreground, positive_logits_background, logits_background,

class HCLEval(nn.Module):
    """
    Build a MgGC model with: a PM-ViT as backbone, and a seg head
    """

    def __init__(self, vit_arch, pretrained_path, checkpoint_key,
                 patch_size=16, embed_dim=384, seg_head_type="mlp",
                 apply_dropout=False, use_projector=False):
        super(HCLEval, self).__init__()
        self.patch_size = patch_size
        self.in_channels = embed_dim * 4 if patch_size == 16 else embed_dim * 5
        self.checkpoint_key = checkpoint_key
        self.vit_featurizer = vits.__dict__[vit_arch](
            patch_size=patch_size)  # Dino weights will be loaded automatically.
        self.seg_head = SegHead(in_channels=self.in_channels,#sp_token_dim=embed_dim,
                                use_projector=use_projector, seg_head_type=seg_head_type,
                                apply_dropout=apply_dropout)
        for name, param in self.seg_head.named_parameters():
            param.requires_grad = False

        for name, param in self.vit_featurizer.named_parameters():  # Freeze backbone weights
            param.requires_grad = False

        # load HCL seg head pretrained weights during finetuning
        if os.path.isfile(pretrained_path):
            print("loading weights from, ", pretrained_path)
            assert self.checkpoint_key in ["online_seg_head", "momentum_seg_head"]
            state_dict = torch.load(pretrained_path, map_location="cpu")[self.checkpoint_key]
            msg = self.seg_head.load_state_dict(state_dict, strict=False)
            print(msg)
        else:
            print("=> no checkpoint found at '{}'".format(pretrained_path))

    def forward_backbone(self, x, last_self_attention=False):
        with torch.no_grad():
            x = self.vit_featurizer(x, return_attn=last_self_attention)  # N x embed_dim x 28 x 28
            if last_self_attention:
                x, attn = x
                return self.seg_head.forward_backbone(x), attn[:, :, 0, 1:]
            return self.seg_head.forward_backbone(x)


    def forward(self, x, image_size, original_size=True):
        with torch.no_grad():
            x = self.vit_featurizer(x)  # N x embed_dim x 28 x 28
            return self.seg_head(x, image_size=image_size, original_size=original_size)

