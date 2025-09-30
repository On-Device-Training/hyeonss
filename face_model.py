#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:09:25 2019
Pytorch mobilefacenet & arcface architecture

@author: AIRocker
"""

from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
from torch import nn
import torch
import math
from functools import partial
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F


from torch.quantization import QuantStub, DeQuantStub
from torch.ao.quantization import default_qconfig

############################# Edge Face Net #######################################

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor



class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features,
            act_layer=nn.GELU,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = in_features
        hidden_features = hidden_features or in_features
        bias = bias
        drop_probs = drop

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """

    def __init__(
            self,
            num_channels: int,
            eps: float = 1e-6,
            affine: bool = True,
            **kwargs,
    ):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class SimpleClassifierHead(nn.Module):
    def __init__(
            self,
            in_features,
            num_classes,
            drop_rate = 0.,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_features = in_features

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.norm = LayerNorm2d(in_features, eps=1e-6)
        self.flatten = nn.Flatten(1)
        self.pre_logits = nn.Identity()
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(self.num_features, num_classes) 

    def forward(self, x):
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.pre_logits(x)
        x = self.drop(x)
        x = self.fc(x)
        return x


class PositionalEncodingFourier(nn.Module):
    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        try:
            self.token_projection.qconfig = default_qconfig
        except Exception:
            pass
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, shape: Tuple[int, int, int]):
        device = self.token_projection.weight.device
        dtype = self.token_projection.weight.dtype
        inv_mask = ~torch.zeros(shape).to(device=device, dtype=torch.bool)
        y_embed = inv_mask.cumsum(1, dtype=torch.float32)
        x_embed = inv_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.int64, device=device).to(torch.float32)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(),
             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(),
             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos.to(dtype))

        return pos


class ConvBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            kernel_size=7,
            conv_bias=True,
            expand_ratio=4,
            ls_init_value=1e-6,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU, drop_path=0.,
    ):
        super().__init__()
        dim_out = dim_out or dim

        self.conv_dw = nn.Conv2d(dim, dim_out, kernel_size=kernel_size, stride=1, padding=(kernel_size // 2), groups=dim, bias=conv_bias)
        try:
            self.conv_dw.qconfig = default_qconfig
        except Exception:
            pass
        self.norm = norm_layer(dim_out)
        self.mlp = Mlp(dim_out, int(expand_ratio * dim_out), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim_out)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)

        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = shortcut + self.drop_path(x)
        return x


class CrossCovarianceAttn(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        try:
            self.qkv.qconfig = default_qconfig
            self.proj.qconfig = default_qconfig
        except Exception:
            pass

        # self.quant = QuantStub()
        # self.dequant = DeQuantStub()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 4, 1)
        q, k, v = qkv.unbind(0)

        # NOTE, this is NOT spatial attn, q, k, v are B, num_heads, C, L -->  C x C attn map
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v)

        x = x.permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SplitTransposeBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_scales=1,
            num_heads=8,
            expand_ratio=4,
            use_pos_emb=True,
            conv_bias=True,
            qkv_bias=True,
            ls_init_value=1e-6,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            drop_path=0.,
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__()
        width = max(int(math.ceil(dim / num_scales)), int(math.floor(dim // num_scales)))
        self.width = width
        self.num_scales = max(1, num_scales - 1)

        convs = []
        for i in range(self.num_scales):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=1, padding=(3 // 2), groups=width, bias=conv_bias))
        self.convs = nn.ModuleList(convs)

        self.pos_embd = None
        if use_pos_emb:
            self.pos_embd = PositionalEncodingFourier(dim=dim)
        self.norm_xca = norm_layer(dim)
        self.gamma_xca = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value > 0 else None
        self.xca = CrossCovarianceAttn(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)

        self.norm = norm_layer(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(expand_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x

        # scales code re-written for torchscript as per my res2net fixes -rw
        # NOTE torch.split(x, self.width, 1) causing issues with ONNX export
        spx = x.chunk(len(self.convs) + 1, dim=1)
        spo = []
        sp = spx[0]
        for i, conv in enumerate(self.convs):
            if i > 0:
                sp = sp + spx[i]
            sp = conv(sp)
            spo.append(sp)
        spo.append(spx[-1])
        x = torch.cat(spo, 1)

        # XCA
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        if self.pos_embd is not None:
            pos_encoding = self.pos_embd((B, H, W)).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding
        x = x + self.drop_path(self.gamma_xca * self.xca(self.norm_xca(x)))
        x = x.reshape(B, H, W, C)

        # Inverted Bottleneck
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = shortcut + self.drop_path(x)
        return x


class EdgeNeXtStage(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            stride=2,
            depth=2,
            num_global_blocks=1,
            num_heads=4,
            scales=2,
            kernel_size=7,
            expand_ratio=4,
            use_pos_emb=False,
            conv_bias=True,
            ls_init_value=1.0,
            drop_path_rates=None,
            norm_layer=LayerNorm2d,
            norm_layer_cl=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU
    ):
        super().__init__()
        # 0. downsample layer
        if stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                nn.Conv2d(in_chs, out_chs, kernel_size=2, stride=2, bias=conv_bias)
            )
            in_chs = out_chs

        # 1. stage blocks
        stage_blocks = []
        for i in range(depth): # add blocks to the stage
            if i < depth - num_global_blocks: # 
                stage_blocks.append(
                    ConvBlock(
                        dim=in_chs,
                        dim_out=out_chs,
                        conv_bias=conv_bias,
                        kernel_size=kernel_size,
                        expand_ratio=expand_ratio,
                        ls_init_value=ls_init_value,
                        drop_path=drop_path_rates[i],
                        norm_layer=norm_layer_cl,
                        act_layer=act_layer,
                    )
                )
            else:
                stage_blocks.append(
                    SplitTransposeBlock(
                        dim=in_chs,
                        num_scales=scales,
                        num_heads=num_heads,
                        expand_ratio=expand_ratio,
                        use_pos_emb=use_pos_emb,
                        conv_bias=conv_bias,
                        ls_init_value=ls_init_value,
                        drop_path=drop_path_rates[i],
                        norm_layer=norm_layer_cl,
                        act_layer=act_layer,
                    )
                )
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class EdgeNeXt(nn.Module):
    def __init__(
            self,
            in_chans=3,
            num_classes=512,
            global_pool='avg',
            dims=(24, 48, 88, 168),
            depths=(3, 3, 9, 3),
            global_block_counts=(0, 1, 1, 1),
            kernel_sizes=(3, 5, 7, 9),
            heads=(8, 8, 8, 8),
            d2_scales=(2, 2, 3, 4),
            use_pos_emb=(False, True, False, False),
            ls_init_value=1e-6,
            expand_ratio=4,
            conv_bias=True,
            act_layer=nn.GELU,
            drop_path_rate=0.,
            drop_rate=0.,
            prune_ratio: Union[List[float], float]=None,  # new argument for pruning ratio
    ):
        super().__init__()
        # initialization
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.drop_rate = drop_rate
        norm_layer = partial(LayerNorm2d, eps=1e-6)
        norm_layer_cl = partial(nn.LayerNorm, eps=1e-6)
        self.feature_info = []

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # 0. stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4, bias=conv_bias),
            norm_layer(dims[0]),
        )

        # 1. stages
        curr_stride = 4 # stride -> downsample layer
        stages = []
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        in_chs = dims[0]
        for i in range(4):
            stride = 2 if curr_stride == 2 or i > 0 else 1
            # FIXME support dilation / output_stride
            curr_stride *= stride
            stages.append(EdgeNeXtStage(
                in_chs=in_chs,
                out_chs=dims[i],
                stride=stride,
                depth=depths[i],
                num_global_blocks=global_block_counts[i],
                num_heads=heads[i],
                drop_path_rates=dp_rates[i],
                scales=d2_scales[i],
                expand_ratio=expand_ratio,
                kernel_size=kernel_sizes[i],
                use_pos_emb=use_pos_emb[i],
                ls_init_value=ls_init_value,
                conv_bias=conv_bias,
                norm_layer=norm_layer,
                norm_layer_cl=norm_layer_cl,
                act_layer=act_layer,
            ))
            # NOTE feature_info use currently assumes stage 0 == stride 1, rest are stride 2
            in_chs = dims[i]
            self.feature_info += [dict(num_chs=in_chs, reduction=curr_stride, module=f'stages.{i}')]

        self.stages = nn.Sequential(*stages)

        # 2. classifier head
        self.num_features = dims[-1]

        self.norm_pre = nn.Identity()

        self.head = SimpleClassifierHead(self.num_features, num_classes)

        # pruning MLP channels if specified
        if prune_ratio is not None:
            self.mlp_channel_prune(prune_ratio)

    def forward_features(self, x):
        x = self.quant(x)
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm_pre(x)
        x = self.dequant(x)

        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
    
    @torch.no_grad()
    def mlp_channel_prune(self,
                    prune_ratio: Union[List[float], float]):
        """
        Prune MLPs (two Linear layers: fc1, fc2).
        Only hidden dimension is pruned.
        """
        # get all Mlp modules
        all_mlps = [module for stage in self.stages for block in stage.blocks for module in block.modules() if isinstance(module, Mlp)]

        if isinstance(prune_ratio, float):
            prune_ratio = [prune_ratio] * len(all_mlps)

        for i, p_ratio in enumerate(prune_ratio):
            mlp = all_mlps[i]
            fc1, fc2 = mlp.fc1, mlp.fc2

            # get number of hidden channels to keep
            hidden_dim = fc1.out_features
            n_keep = int(round(hidden_dim * (1 - p_ratio)))

            # fc1: prune output channels
            mlp.fc1 = nn.Linear(fc1.weight.shape[1], n_keep, bias=(fc1.bias is not None), device=fc1.weight.device, dtype=fc1.weight.dtype)

            # fc2: prune input channels
            mlp.fc2 = nn.Linear(n_keep, fc2.weight.shape[0], bias=(fc2.bias is not None), device=fc2.weight.device, dtype=fc2.weight.dtype)

            # # prune channels for state_dict size matching
            # # fc1: prune output channels
            # fc1 = nn.Linear(fc1.weight.shape[1], n_keep, bias=(fc1.bias is not None), device=fc1.weight.device, dtype=fc1.weight.dtype)
            # fc1.weight.set_(fc1.weight.detach()[:n_keep, :])
            # if fc1.bias is not None:
            #     fc1.bias.set_(fc1.bias.detach()[:n_keep])

            # # fc2: prune input channels (no pruning for bias)
            # fc2.weight.set_(fc2.weight.detach()[:, :n_keep])

#################################################################

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

##################################  MobileFaceNet #############################################################
    
class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Depth_Wise(Module):
     def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
     def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)
    def forward(self, x):
        return self.model(x)

class MobileFaceNet(Module):
    def __init__(self, embedding_size):
        super(MobileFaceNet, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7,7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2_dw(out)

        out = self.conv_23(out)

        out = self.conv_3(out)
        
        out = self.conv_34(out)

        out = self.conv_4(out)

        out = self.conv_45(out)

        out = self.conv_5(out)

        out = self.conv_6_sep(out)

        out = self.conv_6_dw(out)

        out = self.conv_6_flatten(out)

        out = self.linear(out)

        out = self.bn(out)
        return l2_norm(out)

##################################  Arcface head #############################################################

class Arcface(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self, embedding_size=512, classnum=51332,  s=64., m=0.5):
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        nn.init.xavier_uniform_(self.kernel)
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)
    def forward(self, embbedings, label):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel,axis=0) # normalize for each column
        # cos(theta+m)
        cos_theta = torch.mm(embbedings,kernel_norm)
#         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s # scale up in order to make softmax work, first introduced in normface
        return output

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input = torch.Tensor(2, 3, 112, 112).to(device)
    net = MobileFaceNet(512).to(device)
    x = net(input)
    print(x.shape)

