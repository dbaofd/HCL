"""
Adapted from
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial
import torch
import torch.nn as nn
from utils.utils import trunc_normal_


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """

    def __init__(self, img_size=[224], patch_size=16, in_chans=3, embed_dim=768,
                 output_dim=128, hidden_dim=2048, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, use_projector=False, n_layers_projection_head=3, l2_norm=True, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.out_fea_dim = int(
            img_size[0] / patch_size)  # out_fea_dim * out_fea_dim is the number of patches,i.e.,num_spa_tokens.
        self.image_size = img_size[0]
        self.use_projector = use_projector
        self.l2_norm = l2_norm
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        if self.use_projector:
            # Construct projection head, adopted from leopart
            nlayers = max(n_layers_projection_head, 1)
            if nlayers == 1:
                self.projection_head = nn.Linear(embed_dim,
                                                 output_dim)  # after mergeing multi-level patch feature, the dim became 2 times of original.
            else:
                layers = [nn.Linear(embed_dim, hidden_dim)]
                layers.append(nn.GELU())
                for _ in range(nlayers - 2):
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
                    layers.append(nn.GELU())
                layers.append(nn.Linear(hidden_dim, output_dim))
                self.projection_head = nn.Sequential(*layers)
            print("++++++++++++++++vit is using projector++++++++++++++++")

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x, original_size=True, return_cls=False):
        if self.image_size != x.shape[2]:  # during inference, the input size could be different from the training input
            self.out_fea_dim = x.shape[2] // self.patch_embed.patch_size  # update from 28 -> 56
            self.image_size = x.shape[2]  # 224->448
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # N x (num_spa_tokens+1) x emdim
        x_cls = x[:, 0]  # N x emdim
        x = x[:, 1:]
        if self.use_projector:
            x = self.projection_head(x)
            x_cls = self.projection_head(x_cls)
            if self.l2_norm:
                x = nn.functional.normalize(x, dim=2, p=2)  # normlize on emdim dimention.
                x_cls = nn.functional.normalize(x_cls, dim=1, p=2)  # normlize on emdim dimention.
        x = x.permute((0, 2, 1))  # N x emdim x num_spa_tokens
        x = torch.reshape(x,
                          (x.shape[0], x.shape[1], self.out_fea_dim, self.out_fea_dim))  # N x emdim x out_sim x out_dim
        if original_size:
            x = nn.functional.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear',
                                          align_corners=False)
        if return_cls:
            return x, x_cls
        return x

    def forward_backbone(self, x, last_self_attention=False):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                x = blk(x, return_attention=last_self_attention)
        if last_self_attention:
            x, attn = x
        x = self.norm(x)
        x = x[:, 1:]
        if last_self_attention:
            return x, attn[:, :, 0, 1:]
        return x

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, pretrained=True, pretrain_weights="dino", **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if pretrained:
        valid_weights = ["dino", "leopart", "croc-coco", "croc-imagenet"]
        assert (pretrain_weights in valid_weights)
        if patch_size == 16 and pretrain_weights == "dino":
            pretrained_weights_path = "weights/pretrain/dino_deitsmall16_pretrain.pth"
            print("loading pretrained weights ", pretrained_weights_path)
            state_dict = torch.load(pretrained_weights_path, map_location="cpu")
            msg = model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights_path, msg))
        elif patch_size == 16 and pretrain_weights == "leopart":
            pretrained_weights_path = "weights/pretrain/leopart_vits16.ckpt"
            print("loading pretrained weights ", pretrained_weights_path)
            state_dict = torch.load(pretrained_weights_path, map_location="cpu")
            for k in list(state_dict.keys()):
                state_dict[k[len("model."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            msg = model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights_path, msg))
        elif patch_size == 16 and (pretrain_weights == "croc-coco" or pretrain_weights == "croc-imagenet"):
            if pretrain_weights == "croc-coco":
                pretrained_weights_path = "weights/pretrain/croc_coco+.pth"  # croc_imagenet1k.pth"
            elif pretrain_weights =="croc-imagenet":
                pretrained_weights_path = "weights/pretrain/croc_imagenet1k.pth"
            check_point = "student"
            print("loading pretrained weights ", pretrained_weights_path)
            state_dict = torch.load(pretrained_weights_path, map_location="cpu")[check_point]
            for k in list(state_dict.keys()):
                state_dict[k[len("module.backbone."):]] = state_dict[
                    k]  # len("bakcbone") for teacher len("module.backbone.") for student
                # delete renamed or unused k
                del state_dict[k]
            msg = model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights_path, msg))
        elif patch_size == 8:
            pretrained_weights_path = "weights/pretrain/dino_deitsmall8_pretrain.pth"
            print("loading pretrained weights ", pretrained_weights_path)
            state_dict = torch.load(pretrained_weights_path, map_location="cpu")
            msg = model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights_path, msg))
    return model


def vit_base(patch_size=16, pretrained=True, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    if pretrained:
        pretrained_weights = "weights/pretrain/dino_vitbase8_pretrain.pth"
        #         checkpoint_key= "teacher"
        print("loading pretrained weights ", pretrained_weights)
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    return model
