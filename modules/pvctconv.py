# ------------------------------------------
# CSWin Transformer
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Xiaoyi Dong
# ------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.nn import BatchNorm1d,BatchNorm2d,Conv1d,Conv2d
from torch.nn import LayerNorm,Conv3d
# from modules.functional import MABN1d as BatchNorm1d,MABN2d as BatchNorm2d,CenConv1d as Conv1d,CenConv2d as Conv2d,MABN_Layer as LayerNorm,CenConv3d as Conv3d
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops.layers.torch import Rearrange
import torch.utils.checkpoint as checkpoint
import numpy as np
import time
import modules.functional as F
from modules.voxelization import Voxelization
from modules.shared_transformer import SharedTransformer
from modules.shared_mlp import SharedMLP
from modules.se import SE3d
# from utils import FRN1d,FRN2d,TLU1d,TLU2d
__all__ = ['PVCTConv']
# def _cfg(url='', **kwargs):
#     return {
#         'url': url,
#         'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
#         'crop_pct': .9, 'interpolation': 'bicubic',
#         'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
#         'first_conv': 'patch_embed.proj', 'classifier': 'head',
#         **kwargs
#     }


# default_cfgs = {
#     'cswin_224': _cfg(),
#     'cswin_384': _cfg(
#         crop_pct=1.0
#     ),
#
# }


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

class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0., qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        # idx:branch编号 0/1 最后一层：idx=-1
        # resolution=H/4 8 16 32   split_size=1 2 7 7
        # if idx == -1:
        #     D_sp, H_sp, W_sp = self.resolution, self.resolution, self.resolution
        if idx == 0:
            D_sp, H_sp, W_sp = self.resolution, self.resolution, self.split_size
        elif idx == 1:
            D_sp, W_sp, H_sp = self.resolution, self.split_size, self.resolution
        elif idx == 2:
            D_sp, W_sp, H_sp = self.split_size, self.resolution, self.resolution
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.D_sp = D_sp
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.get_v = Conv3d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        # B*64 D/4*H/4*W/4 C/2
        # B*27 D/3*H/3*W/3 C/2
        B, N, C = x.shape
        D = H = W = int(np.ceil((np.power(N,1.0/3))))
        # B*64 C D/4 H/4 W/4
        x = x.transpose(-2,-1).contiguous().view(B, C, D, H, W)
        # H_sp=H/4 8 16 32
        # W_sp=1   2  7  7
        # B*64*1*W/4 D/4*H/4 1 C
        x = img2windows(x, self.D_sp, self.H_sp, self.W_sp)
        # B*64*1*W/4 nh H/4*1 C/nh
        x = x.reshape(-1, self.D_sp*self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        D = H = W = int(np.ceil((np.power(N,1.0/3))))
        x = x.transpose(-2,-1).contiguous().view(B, C, D, H, W)

        D_sp, H_sp, W_sp = self.D_sp, self.H_sp, self.W_sp
        x = x.view(B, C, D // D_sp, D_sp, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().reshape(-1, C, D_sp, H_sp, W_sp) ### B', C, H', W'

        lepe = func(x) ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, D_sp * H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.D_sp * self.H_sp* self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        # 3 B*64 D/4*H/4*W/4 C/2
        q,k,v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        # H/4 8 16 32
        D = H = W = self.resolution
        # B*64 D/4*H/4*W/4 C/2
        B, L, C = q.shape
        assert L == D * H * W, "flatten img_tokens has wrong size"
        # B*64*1*W/4 nh H/4*1 C/nh
        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.D_sp * self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x,self.D_sp, self.H_sp, self.W_sp, D, H, W).view(B, -1, C)  # B H' W' C

        return x


class CSWinBlock(nn.Module):

    def __init__(self, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)
        # 最后一层patches_resolution == split_size=7
        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 3
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        
        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx = -1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim//3, resolution=self.patches_resolution, idx = i,
                    split_size=split_size, num_heads=num_heads//3, dim_out=dim//3,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        # self.norm2 = norm_layer(dim)
        self.norm2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # print(x.shape)
        D = H = W = self.patches_resolution
        # B*64 D/4*H/4*W/4 C
        B, L, C = x.shape
        assert L == D * H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        # 3 B*64 D/4*H/4*W/4 C
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        
        if self.branch_num == 3:
            # 3 B*64 D/4*H/4*W/4 C/2
            x1 = self.attns[0](qkv[:,:,:,:C//3])
            x2 = self.attns[1](qkv[:,:,:,C//3:2*(C//3)])
            x3 = self.attns[2](qkv[:,:,:,2*(C//3):])
            attened_x = torch.cat([x1,x2,x3], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x.transpose(1,2)).transpose(1,2)))
        # print(x.shape)
        return x

def img2windows(img, D_sp, H_sp, W_sp):
    """
    img: B C D H W
    Returns: B*num_windows window_size*window_size*window_size C
    """
    B, C, D, H, W = img.shape
    img_reshape = img.view(B, C, D // D_sp, D_sp, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous().reshape(-1, D_sp* H_sp* W_sp, C)
    return img_perm

def windows2img(img_splits_hw, D_sp, H_sp, W_sp, D, H, W):
    """
    img_splits_hw: B' D H W C
    Returns: B D H W C
    """
    B = int(img_splits_hw.shape[0] / (D * H * W / D_sp / H_sp / W_sp))

    img = img_splits_hw.view(B, D // D_sp, H // H_sp, W // W_sp, D_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return img


class CSWinTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=32, in_chans=6, embed_dim=96, depth=1, split_size = 1,
                 num_heads=6, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=LayerNorm, use_chk=False):
        super().__init__()
        self.use_chk = use_chk
        self.img_size = img_size
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        heads=num_heads
        # done
        self.projection = nn.Sequential(
            Conv3d(in_chans, embed_dim, 1, 1, 0),
            Rearrange('b c d h w -> b (d h w) c', d=img_size, h=img_size, w=img_size),
            LayerNorm(embed_dim)
        )

        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule
        self.stage1 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads, reso=img_size, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(curr_dim)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            # nn.init.kaiming_normal_(m.weight,1.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1.0)
            # trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B = x.shape[0]
        # B*64 D/4*H/4*W/4 C
        x = self.projection(x)

        for blk in self.stage1:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.norm(x)
        return x.transpose(1,2).reshape(-1,self.embed_dim,self.img_size,self.img_size,self.img_size)

    def forward(self, x):
        x = self.forward_features(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

class Pct(nn.Module):
    def __init__(self, in_channels, output_channels):
        super(Pct, self).__init__()
        self.conv1 = Conv1d(in_channels, output_channels, kernel_size=1, bias=False)
        self.conv2 = Conv1d(output_channels, output_channels, kernel_size=1, bias=False)
        self.bn1 = BatchNorm1d(output_channels)
        self.bn2 = BatchNorm1d(output_channels)
        # self.bn1 = nn.GroupNorm(int(output_channels//16),output_channels)
        # self.bn2 = nn.GroupNorm(int(output_channels//16),output_channels)
        # self.bn1 = FRN1d(output_channels)
        # self.bn2 = FRN1d(output_channels)
        # self.act1 = TLU1d(output_channels)
        # self.act2 = TLU1d(output_channels)
        self.sa = SA_Layer(output_channels)


    def forward(self, x):
        # print('PCTinput:{}'.format(x.size()))
        x = x.permute(0, 2, 1)
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))
        # # B, D, N
        # x = self.act1(self.bn1(self.conv1(x)))
        # # B, D, N
        # x = self.act2(self.bn2(self.conv2(x)))
        x = self.sa(x)
        # print('PCToutput:{}'.format(x.shape))
        return x

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = Conv1d(channels, channels, 1)
        self.after_norm = BatchNorm1d(channels)
        # self.after_norm = nn.GroupNorm(int(channels//16),channels)
        self.act = nn.ReLU()
        # self.after_norm = FRN1d(channels)
        # self.act = TLU1d(channels)
        self.softmax = nn.Softmax(dim=-1)
        # self.apply(self._init_weights)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv1d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
class PVCTConv(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, depth, split_size ,
                 num_heads, mlp_ratio=4., normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = img_size
        self.boxsize = 3
        self.mlp_dims = out_channels
        self.drop_path1 = 0.1
        self.drop_path2 = 0.2
        self.voxelization = Voxelization(img_size, normalize=normalize, eps=eps)
        self.voxel_encoder = CSWinTransformer(img_size,in_channels, out_channels,  depth, split_size ,
                 num_heads, mlp_ratio=4.)
        self.SE = SE3d(out_channels)
        # self.point_features = SharedTransformer(in_channels, out_channels)
        self.point_features = SharedMLP(in_channels, out_channels)

    def forward(self, inputs):
        features, coords = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_encoder(voxel_features)
        voxel_features = self.SE(voxel_features)
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        fused_features = voxel_features + self.point_features(features)
        return fused_features, coords
if __name__ == '__main__':
    x = torch.randn(2,6,32,32,32)
    model = CSWinTransformer()
    out = model(x)
    print(out.shape)