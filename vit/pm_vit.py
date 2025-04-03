import math
from functools import partial
import torch
import torch.nn as nn
from utils.utils import trunc_normal_
from vit.vision_transformer import Block, PatchEmbed
from utils.utils import pi_resize_patch_embed


class PMViT16(nn.Module):
    """ PM-ViT with patch size 16 """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, l2_norm=True, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.out_fea_dim = int(img_size[0] / patch_size)
        # out_fea_dim * out_fea_dim is the number of patches,i.e.,num_spa_tokens.
        self.out_fea_dim_2 = int(img_size[0] / (2 * patch_size))  # 32
        self.out_fea_dim_3 = int(img_size[0] / (7 * (patch_size / 2)))  # 56
        self.fusion_dim = embed_dim * 2
        self.image_size = img_size[0]
        self.l2_norm = l2_norm
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed_2 = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size * 2, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed_3 = PatchEmbed(
            img_size=img_size[0], patch_size=int(7 * (patch_size / 2)), in_chans=in_chans, embed_dim=embed_dim)
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

    def interpolate_pos_encoding(self, x, w, h, patch_embed):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // patch_embed.patch_size
        h0 = h // patch_embed.patch_size
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
        x_1 = self.patch_embed(x)  # patch linear embedding
        x_2 = self.patch_embed_2(x)
        x_3 = self.patch_embed_3(x)
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_1 = torch.cat((cls_tokens, x_1), dim=1)
        x_2 = torch.cat((cls_tokens, x_2), dim=1)
        x_3 = torch.cat((cls_tokens, x_3), dim=1)

        # add positional encoding to each token
        x_1 = x_1 + self.interpolate_pos_encoding(x_1, w, h, self.patch_embed)
        x_2 = x_2 + self.interpolate_pos_encoding(x_2, w, h, self.patch_embed_2)
        x_3 = x_3 + self.interpolate_pos_encoding(x_3, w, h, self.patch_embed_3)

        return self.pos_drop(x_1), self.pos_drop(x_2), self.pos_drop(x_3)

    def forward(self, x, return_attn=False, return_cls=False):
        if self.image_size != x.shape[2]:  # during inference, the input size could be different from the training input
            self.out_fea_dim = x.shape[2] // self.patch_embed.patch_size
            self.out_fea_dim_2 = x.shape[2] // self.patch_embed_2.patch_size
            self.out_fea_dim_3 = x.shape[2] // self.patch_embed_3.patch_size
            self.image_size = x.shape[2]  # 224->448
        x_1, x_2, x_3 = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x_1 = blk(x_1)
                x_2 = blk(x_2)
                x_3 = blk(x_3)
            else:
                x_1, x_1_attn = blk(x_1, return_attention=True)
                x_2 = blk(x_2)
                x_3 = blk(x_3)
        x_1 = self.norm(x_1)  # N x (num_spa_tokens+1) x emdim
        x_2 = self.norm(x_2)  # N x (num_spa_tokens+1) x emdim
        x_3 = self.norm(x_3)  # N x (num_spa_tokens+1) x emdim

        x_1_cls = x_1[:, 0]  # N x emdim
        x_1_cls_token = x_1[:, 0]  # N x emdim
        x_1 = x_1[:, 1:]#N x num_spa_tokens x emdim
        x_2 = x_2[:, 1:]
        x_3 = x_3[:, 1:]
        x_1 = x_1.permute((0, 2, 1))  # N x emdim x num_spa_tokens
        x_2 = x_2.permute((0, 2, 1))  # N x emdim x num_spa_tokens
        x_3 = x_3.permute((0, 2, 1))  # N x emdim x num_spa_tokens

        x_1 = torch.reshape(x_1, (
            x_1.shape[0], x_1.shape[1], self.out_fea_dim, self.out_fea_dim))  # N x emdim x out_sim x out_dim
        x_2 = torch.reshape(x_2, (
            x_2.shape[0], x_2.shape[1], self.out_fea_dim_2, self.out_fea_dim_2))  # N x emdim x out_sim x out_dim
        x_3 = torch.reshape(x_3, (
            x_3.shape[0], x_3.shape[1], self.out_fea_dim_3, self.out_fea_dim_3))  # N x emdim x out_sim x out_dim
        # upsample to x_1 spatial dimension
        x_2 = nn.functional.interpolate(x_2, size=(self.out_fea_dim, self.out_fea_dim), mode='bilinear',
                                        align_corners=False)
        x_3 = nn.functional.interpolate(x_3, size=(self.out_fea_dim, self.out_fea_dim), mode='bilinear',
                                        align_corners=False)
        x_1_cls = torch.reshape(x_1_cls, (x_1_cls.shape[0], x_1_cls.shape[1], 1, 1))
        x_1_cls = nn.functional.interpolate(x_1_cls, size=(self.out_fea_dim, self.out_fea_dim), mode='nearest')

        x = torch.cat((x_1, x_2, x_3,x_1_cls ), dim=1) #, x_1_cls,

        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        x = x.permute((0, 2, 1))  # N x num_tokens x emdim
        if return_attn and not return_cls:
            return x, x_1_attn  # N x 6 x 197 x 197
        elif return_attn and return_cls:
            return x, x_1_attn, x_1_cls_token
        return x

    def get_last_selfattention(self, x):
        x_1, x_2, x_3 = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x_1 = blk(x_1)
            else:
                x_1_attn = blk(x_1, return_attention=True)

                # return attention of the last block
                return x_1_attn

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


class PMViT8(nn.Module):
    """ PM-ViT with patch size 8 """

    def __init__(self, img_size=[224], patch_size=8, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, l2_norm=True, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.image_size = img_size[0]
        self.l2_norm = l2_norm
        self.out_fea_dim = int(
            img_size[0] / patch_size)  # 28 out_fea_dim * out_fea_dim is the number of patches,i.e.,num_spa_tokens.
        self.out_fea_dim_2 = int(img_size[0] / (2 * patch_size))  # 14
        self.out_fea_dim_3 = int(img_size[0] / (4 * patch_size))  # 7sys
        self.out_fea_dim_4 = int(img_size[0] / (7 * patch_size))  # 4
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)  # 8 x 8
        self.patch_embed_2 = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size * 2, in_chans=in_chans, embed_dim=embed_dim)  # 16 x 16
        self.patch_embed_3 = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size * 4, in_chans=in_chans, embed_dim=embed_dim)  # 32 x 32
        self.patch_embed_4 = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size * 7, in_chans=in_chans, embed_dim=embed_dim)  # 56 x 56

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

    def interpolate_pos_encoding(self, x, w, h, patch_embed):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // patch_embed.patch_size
        h0 = h // patch_embed.patch_size
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
        x_1 = self.patch_embed(x)  # patch linear embedding
        x_2 = self.patch_embed_2(x)
        x_3 = self.patch_embed_3(x)
        x_4 = self.patch_embed_4(x)

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_1 = torch.cat((cls_tokens, x_1), dim=1)
        x_2 = torch.cat((cls_tokens, x_2), dim=1)
        x_3 = torch.cat((cls_tokens, x_3), dim=1)
        x_4 = torch.cat((cls_tokens, x_4), dim=1)

        # add positional encoding to each token
        x_1 = x_1 + self.interpolate_pos_encoding(x_1, w, h, self.patch_embed)
        x_2 = x_2 + self.interpolate_pos_encoding(x_2, w, h, self.patch_embed_2)
        x_3 = x_3 + self.interpolate_pos_encoding(x_3, w, h, self.patch_embed_3)
        x_4 = x_4 + self.interpolate_pos_encoding(x_4, w, h, self.patch_embed_4)

        return self.pos_drop(x_1), self.pos_drop(x_2), self.pos_drop(x_3), self.pos_drop(x_4)

    def forward(self, x, return_attn=False, return_cls=False):
        if self.image_size != x.shape[2]:  # during inference, the input size could be different from the training input
            self.out_fea_dim = x.shape[2] // self.patch_embed.patch_size
            self.out_fea_dim_2 = x.shape[2] // self.patch_embed_2.patch_size
            self.out_fea_dim_3 = x.shape[2] // self.patch_embed_3.patch_size
            self.out_fea_dim_4 = x.shape[2] // self.patch_embed_4.patch_size
            self.image_size = x.shape[2]  # 224->448
        x_1, x_2, x_3, x_4 = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x_1 = blk(x_1)
                x_2 = blk(x_2)
                x_3 = blk(x_3)
                x_4 = blk(x_4)
            else:
                x_1, x_1_attn = blk(x_1, return_attention=True)
                x_2 = blk(x_2)
                x_3 = blk(x_3)
                x_4 = blk(x_4)
        x_1 = self.norm(x_1)  # N x (num_spa_tokens+1) x emdim
        x_2 = self.norm(x_2)  # N x (num_spa_tokens+1) x emdim
        x_3 = self.norm(x_3)  # N x (num_spa_tokens+1) x emdim
        x_4 = self.norm(x_4)  # N x (num_spa_tokens+1) x emdim

        x_1_cls = x_1[:, 0]  # N x emdim
        x_1_cls_token = x_1[:, 0]  # N x emdim
        x_1 = x_1[:, 1:]
        x_2 = x_2[:, 1:]
        x_3 = x_3[:, 1:]
        x_4 = x_4[:, 1:]
        # only the following part changed
        # if return_attn:
        #     return (x_1, x_2, x_3, x_4, x_1_cls), x_1_attn
        # return (x_1, x_2, x_3, x_4, x_1_cls)
        x_1 = x_1.permute((0, 2, 1))  # N x emdim x num_spa_tokens
        x_2 = x_2.permute((0, 2, 1))  # N x emdim x num_spa_tokens
        x_3 = x_3.permute((0, 2, 1))  # N x emdim x num_spa_tokens
        x_4 = x_4.permute((0, 2, 1))  # N x emdim x num_spa_tokens

        x_1 = torch.reshape(x_1, (
            x_1.shape[0], x_1.shape[1], self.out_fea_dim, self.out_fea_dim))  # N x emdim x out_sim x out_dim
        x_2 = torch.reshape(x_2, (
            x_2.shape[0], x_2.shape[1], self.out_fea_dim_2, self.out_fea_dim_2))  # N x emdim x out_sim x out_dim
        x_3 = torch.reshape(x_3, (
            x_3.shape[0], x_3.shape[1], self.out_fea_dim_3, self.out_fea_dim_3))  # N x emdim x out_sim x out_dim
        x_4 = torch.reshape(x_4, (
            x_4.shape[0], x_4.shape[1], self.out_fea_dim_4, self.out_fea_dim_4))  # N x emdim x out_sim x out_dim

        # upsample to x_1 spatial dimension
        x_2 = nn.functional.interpolate(x_2, size=(self.out_fea_dim, self.out_fea_dim), mode='bilinear',
                                        align_corners=False)
        x_3 = nn.functional.interpolate(x_3, size=(self.out_fea_dim, self.out_fea_dim), mode='bilinear',
                                        align_corners=False)
        x_4 = nn.functional.interpolate(x_4, size=(self.out_fea_dim, self.out_fea_dim), mode='bilinear',
                                        align_corners=False)

        x_1_cls = torch.reshape(x_1_cls, (x_1_cls.shape[0], x_1_cls.shape[1], 1, 1))
        x_1_cls = nn.functional.interpolate(x_1_cls, size=(self.out_fea_dim, self.out_fea_dim), mode='nearest')

        x = torch.cat((x_1, x_2, x_3, x_4, x_1_cls), dim=1)

        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        x = x.permute((0, 2, 1))  # N x num_tokens x emdim
        if return_attn and not return_cls:
            return x, x_1_attn  # N x 6 x 197 x 197
        elif return_attn and return_cls:
            return x, x_1_attn, x_1_cls_token
        return x

    def get_last_selfattention(self, x):
        x_1, x_2, x_3 = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x_1 = blk(x_1)
            else:
                x_1_attn = blk(x_1, return_attention=True)

                # return attention of the last block
                return x_1_attn

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_small(patch_size=16, pretrained=True, **kwargs):
    valid_patch_size = [8, 16]
    assert patch_size in valid_patch_size
    if patch_size == 16:
        model = PMViT16(
            patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        if pretrained:
            # pretrained_weights_path = "mogoseg_vit_epoch9.pth.tar"
            # state_dict = torch.load(pretrained_weights_path, map_location="cpu")["backbone"]
            # print("loading pretrained weights ", pretrained_weights_path)
            # msg = model.load_state_dict(state_dict, strict=False)
            # print(msg, "okokokokokokok") # when finetune extra pe layers, we need to save the whole backbone weights and load it here.
            pretrained_weights_path = "weights/pretrain/dino_deitsmall16_pretrain.pth"
            print("loading pretrained weights ", pretrained_weights_path)
            state_dict = torch.load(pretrained_weights_path, map_location="cpu")
            # use pi_resize proposed in FlexiViT to resize the original patch embedding weights to new scale.
            state_dict["patch_embed_2.proj.weight"] = pi_resize_patch_embed(
                patch_embed=state_dict["patch_embed.proj.weight"], new_patch_size=(32, 32)
            )
            # state_dict["patch_embed_2.proj.weight"] = nn.functional.interpolate(
            #     state_dict["patch_embed.proj.weight"],
            #     size=(32,32), mode='bilinear')#bicubic bilinear

            state_dict["patch_embed_2.proj.bias"] = state_dict["patch_embed.proj.bias"]
            state_dict["patch_embed_3.proj.weight"] = pi_resize_patch_embed(
                patch_embed=state_dict["patch_embed.proj.weight"], new_patch_size=(56, 56)
            )
            # state_dict["patch_embed_3.proj.weight"] = nn.functional.interpolate(
            #     state_dict["patch_embed.proj.weight"],
            #     size=(56, 56), mode='bilinear')
            state_dict["patch_embed_3.proj.bias"] = state_dict["patch_embed.proj.bias"]

            msg = model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights_path, msg))
    elif patch_size == 8:
        model = PMViT8(
            patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

        if pretrained:
            pretrained_weights_path = "weights/pretrain/dino_deitsmall8_pretrain.pth"
            print("loading pretrained weights ", pretrained_weights_path)
            state_dict = torch.load(pretrained_weights_path, map_location="cpu")
            # use pi_resize proposed in FlexiViT to resize the original patch embedding weights to new scale.
            state_dict["patch_embed_2.proj.weight"] = pi_resize_patch_embed(
                patch_embed=state_dict["patch_embed.proj.weight"], new_patch_size=(patch_size * 2, patch_size * 2)
            )
            state_dict["patch_embed_2.proj.bias"] = state_dict["patch_embed.proj.bias"]
            state_dict["patch_embed_3.proj.weight"] = pi_resize_patch_embed(
                patch_embed=state_dict["patch_embed.proj.weight"], new_patch_size=(patch_size * 4, patch_size * 4)
            )
            state_dict["patch_embed_3.proj.bias"] = state_dict["patch_embed.proj.bias"]

            state_dict["patch_embed_4.proj.weight"] = pi_resize_patch_embed(
                patch_embed=state_dict["patch_embed.proj.weight"], new_patch_size=(patch_size * 7, patch_size * 7)
            )
            state_dict["patch_embed_4.proj.bias"] = state_dict["patch_embed.proj.bias"]

            msg = model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights_path, msg))
    return model
