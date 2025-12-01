# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional

from timm.models.layers import DropPath, to_2tuple

# from mamba_ssm.modules.mamba_simple import Mamba
from mamba_dependency import Mamba


try:
    from mamba_dependency import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# 对图像进行2D卷积，然后展开成一维向量，B C H W -> B N C
# class PatchEmbed(nn.Module):
#     """ 2D Image to Patch Embedding
#     """
#     def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
#         self.num_patches = self.grid_size[0] * self.grid_size[1]
#         self.flatten = flatten

#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

#     def forward(self, x):
#         B, C, H, W = x.shape
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#         x = self.proj(x)
#         if self.flatten:
#             x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
#         x = self.norm(x)
#         return x
    

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn  # 使用了RMSNorm
            if residual is None:
                # hidden_states = hidden_states.to("cuda")   # 修改后的代码
                # norm_weight = self.norm.weight
                # 获取weight参数中的张量
                # weight_tensor = norm_weight.data

                # 将weight数据放置在GPU上
                # weight_tensor_gpu = weight_tensor.to("cuda")

                # 创建新的torch.nn.Parameter对象，将GPU上的数据作为参数
                # new_weight = torch.nn.Parameter(weight_tensor_gpu)
                # self.norm.weight = new_weight

                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )    
        hidden_states = self.mixer(hidden_states, inference_params=inference_params) 
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,
    bimamba_type="none",
    if_devide_out=False,
    init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba_type=bimamba_type, if_devide_out=if_devide_out, init_layer_scale=init_layer_scale, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,                                         
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,# True
        residual_in_fp32=residual_in_fp32, # True
    )
    block.layer_idx = layer_idx  # 0
    return block


# def demo():
#     kwargs = {}
#     model = VisionMamba(
#         patch_size=16, stride=8, embed_dim=384, depth=1, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type=None, if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, **kwargs)
#     model = model.to("cuda")

#     input0 = torch.randn(1, 3, 224, 224, dtype=torch.float32).requires_grad_() # 将输入张量的数据类型设置为 float32
#     input0 = input0.to("cuda")
#     # 卷积操作后把结果放到gpu上

#     torch.manual_seed(0); torch.cuda.manual_seed(0)
#     y = model.forward(input0)
#     y.sum().backward()
    
#     # 输出处理结果
#     print("得到Vision Mamba的输出:")
#     print("shape:", y.shape)
#     print("type:", type(y))

def demo2():
    # 决定输入数据的参数
    # batch_size = 1  # 批大小，自己决定
    # seq_len = 120  # 输入序列长度，自己决定
    # embed_dim = 192 # 序列中每一个token的维度，根据自己的情况决定

    # ssm_cfg = None
    # norm_epsilon = 1e-05 # 与Layer Norm有关的参数
    # rms_norm = True # 使用RMSNorm
    # residual_in_fp32 = True
    # fused_add_norm = True 
    # layer_idx = 0
    # if_bimamba = False
    # bimamba_type = None # 选择Mamba selective scan的机制
    # drop_path_rate = 0.0 # dropout 概率
    # if_devide_out = True
    # init_layer_scale = None
    # factory_kwargs = {'device': None, 'dtype': None}

    vim_block = create_block(
                    d_model=384,
                    ssm_cfg=None,
                    norm_epsilon=1e-05,
                    rms_norm=True,
                    residual_in_fp32=True,
                    fused_add_norm=True,
                    layer_idx=0,
                    if_bimamba=False,
                    bimamba_type=None,
                    drop_path=0.0,
                    if_devide_out=True,
                    init_layer_scale=None,
                    # **factory_kwargs,
                ).cuda()
    
    input0 = torch.randn(196, 128, 384, dtype=torch.float32).requires_grad_() # 将输入张量的数据类型设置为 float32
    input0 = input0.cuda() # 放到gpu上
    residual = None
    inference_params = None

    torch.manual_seed(0); torch.cuda.manual_seed(0)
    y,_ = vim_block.forward(input0, residual, inference_params)    
    y.sum().backward()
    
    # 输出处理结果
    print("得到Mamba Block的输出:")
    print("type:", type(y))
    print("shape:", y.shape)

# if __name__ == '__main__':
    # # demo()
    # demo2()