"""
Implementation of CLIP model
Adapted from: https://github.com/openai/CLIP/blob/main/clip/clip.py
"""

import os
import urllib
import hashlib
import warnings
import math
import torch
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from collections import OrderedDict
from typing import Tuple, Union
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class PromptResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask=None, block_id=1, args=None):
        """
        Args:
            block_id: the id the the block in the whole model, start from 1
        """
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        attn_mask_ = self.attn_mask
        if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
            attn_mask_ = self.attn_mask(x.size(0))   # LND

        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self,x:torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class PromptTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask = None, args=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[PromptResidualAttentionBlock(width, heads, attn_mask, i + 1, args)
                                            for i in range(layers)])
    def forward(self, x: torch.Tensor):
        for i in range(self.layers):
            x = self.resblocks[i](x)
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask=None, block_id=1, args=None):
        """
        Args:
            block_id: the id the the block in the whole model, start from 1
        """
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        if args is not None:
            self.visual_prompt_length = args.global_visual_prompt_length


    def attention(self, q: torch.Tensor,k: torch.Tensor, v: torch.Tensor):
        attn_mask_ = self.attn_mask
        if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
            attn_mask_ = self.attn_mask(q.size(0))   # LND

        attn_mask_ = attn_mask_.to(dtype=q.dtype, device=q.device) if attn_mask_ is not None else None
        output = self.attn(q, k, v, need_weights=False, attn_mask=attn_mask_)[0]
        return output

    def forward(self, x_tuple:tuple):

        x, video_frame,visual = x_tuple

        if  visual:
            B = x.size(1)
            BT = B*video_frame
            T = video_frame
            dim = x.size(-1)
            visual_prompt,frame_token= x[:self.visual_prompt_length,:,:],x[self.visual_prompt_length:,:,:].reshape(-1,BT,dim)
            frame_token = self.ln_1(frame_token)
            visual_prompt = self.ln_1(visual_prompt)
            #attention1 attn_output_frames
            
            query1 = frame_token #  Frame tokens: [4+50, batch_size*num_frames, dim]
            
            key1 = torch.zeros(self.visual_prompt_length+query1.size(0),BT,dim).to(x.device)  #[4+49, batch_size*num_frames,dim]
            for i in range(0,BT,B):
                key1[:,i:i+B, :] = torch.cat((
                            visual_prompt,
                            query1[:, i:i+B, :]), dim=0)

            attention_output_frames = self.attention(query1,key1,key1).reshape(-1,B,dim) # [54*num_frames,batch_size, dim]

            #attention2 attn_output_global_prompt
            
            query2 = visual_prompt  # [4, batch_size, dim]
            key2 = torch.cat((visual_prompt,frame_token.reshape(-1,B,dim)),dim=0).to(x.device)   # [4+50*num_frames,batch_size,dim]

            attention_output_prompt = self.attention(query2,key2,key2)
            x = x + torch.cat((attention_output_prompt,attention_output_frames),dim=0) #  cancatenate: torch.cat([attn_output_global, attn_output_frames]
            #x = x + attention_output_frames

        else:
            x_ln = self.ln_1(x)
            x = x + self.attention(x_ln,x_ln,x_ln)
        # place 2, after self-attention
        x = x + self.mlp(self.ln_2(x))
        return (x, video_frame,visual)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask = None, args=None):
        super().__init__()
        self.width = width
        self.layers = layers

        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, i + 1,args)
                                            for i in range(layers)])

    def forward(self, x: torch.Tensor, video_frame=-1, visual=False):
        if not visual:
            return self.resblocks((x,video_frame,False))[0]
        else:
            return self.resblocks((x,video_frame,True))[0]

class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                     linear_patch: str = '2d',
                    video_frames=None, args=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.width = width
        assert linear_patch in ['2d', '3d']

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2+1 , width))
        
        if args.time_embedding != 0: 
            self.frame_embedding = nn.Parameter(scale * torch.randn(video_frames,width).unsqueeze(1))
        else:
            self.frame_embedding = None
            
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, args=args)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        
        ############################################ NEW ADDED CODE ############################################
        self.linear_patch = linear_patch
        self.video_frames = video_frames
        # For 3D patch
        if self.linear_patch == '3d':
            self.conv2 = nn.Conv3d(in_channels=3, out_channels=width, kernel_size=(3, patch_size, patch_size),
                                   stride=(1, patch_size, patch_size), padding=(1, 0, 0), bias=False)
        # position ids (1, len_position_emb)
        self.register_buffer("position_ids", torch.arange(self.positional_embedding.shape[0]).expand(1, -1))
        self.num_tokens = args.global_visual_prompt_length
        self.shared_latent_space = args.shared_latent_space


        #global prompt
        self.prompt_dropout =  nn.Dropout(0.0)
        self.prompt_proj = nn.Identity()
        prompt_dim = 768
        self.prompt_embeddings = nn.Parameter(torch.zeros(
            1, self.num_tokens, prompt_dim))
        # xavier_uniform initialization
        patch_size = _pair(patch_size)
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

 
        
    def incorporate_prompt(self, x, unified_visual_prompt):
        # combine prompt embeddings with image-patch embeddings

        BT = x.shape[0]
        B = BT//self.video_frames
        # after CLS token, all before image patches
        #x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)

        ## divide prompt
        if self.shared_latent_space == "transformer":
            unified_visual_frame_prompt = unified_visual_prompt.reshape(B,self.video_frames,self.num_tokens,x.size(-1))
        elif  self.shared_latent_space == "linear":
            unified_visual_frame_prompt = unified_visual_prompt.view(B,self.num_tokens,x.size(-1)).unsqueeze(1).expand(-1,self.video_frames,-1,-1)
        else:
            raise NotImplementedError('Do not find implementation of {}'.format(self.shared_latent_space))

        x = x.view(B,self.video_frames,x.size(-2),x.size(-1))
        

        unified_visual_global_prompt = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1))

        x_local_prompt = torch.cat((x[:,:,0:1,:],
                          unified_visual_frame_prompt,
                          x[:,:,1:,:],),dim=2).permute(0,2,1,3).reshape(B,-1,x.size(-1))

        x_prompt = torch.cat((unified_visual_global_prompt,x_local_prompt),dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        return x_prompt

    def forward_deep_prompt(self, x,unified_visual_prompt):

        ## x.shape L,N,D (N=BxT)
        attn_weights = []
        hidden_states = None
        weights = None
        B = x.shape[1]

        num_layers = self.transformer.layers

        for i in range(num_layers):
            if i == 0:
                ##(cls_token + n_prompt + n_patches,batch_size, hidden_dim) (55,768,768)
                hidden_states = self.transformer.resblocks[i]((x,self.video_frames,True))[0]
            else:
                if i <= len(unified_visual_prompt):
                    if self.shared_latent_space == "transformer":
                        unified_visual_frame_prompt = unified_visual_prompt[i].reshape(B,self.video_frames,self.num_tokens,x.size(-1)).permute(2,1,0,3)
                    elif  self.shared_latent_space == "linear":
                        unified_visual_frame_prompt = unified_visual_prompt[i].view(B,self.num_tokens,x.size(-1)).unsqueeze(1).expand(-1,self.video_frames,-1,-1).permute(2,1,0,3)  
                    else:
                        raise NotImplementedError('Do not find implementation of {}'.format(self.shared_latent_space))

                    hidden_states_global = hidden_states[:self.num_tokens, :, :]

                    hidden_states = hidden_states[self.num_tokens:, :, :].reshape(-1,self.video_frames,B,x.size(-1))
                    #hidden_states = hidden_states.reshape(-1,self.video_frames,B,x.size(-1))
                    hidden_states_local = torch.cat((
                        hidden_states[:1,:,:,:],
                        unified_visual_frame_prompt,
                        hidden_states[1+self.num_tokens:,:,:,:],
                    ), dim=0).reshape(-1,B,x.size(-1))

                    hidden_states = torch.cat((hidden_states_global,hidden_states_local),dim=0)

                hidden_states = self.transformer.resblocks[i]((hidden_states,self.video_frames,True))[0]

        #    if self.transformer.vis:
        #        attn_weights.append(weights)
        return hidden_states

    def forward(self, x: torch.Tensor,unified_visual_prompt, video_frame=-1):
        if x.ndim == 5: B, T, C, H, W = x.shape
        if x.ndim == 4:
            BT, C, H, W = x.shape
            B = BT // video_frame

        if self.linear_patch == '3d':
            assert video_frame != -1
            # [B, T, C, H, W]
            x_3d = x.reshape(-1, video_frame, x.shape[-3], x.shape[-2], x.shape[-1])
            # [B, C, T, H, W]
            x_3d = x_3d.permute(0, 2, 1, 3, 4)
            # [B, width, T, grid, grid], grid = H // patch_size
            x_3d = self.conv2(x_3d)		
            # [B, T, width, grid, grid]
            x_3d = x_3d.permute(0, 2, 1, 3, 4)
            # shape = [B x T, width, grid, grid]
            x = x_3d.reshape(-1, x_3d.shape[-3], x_3d.shape[-2], x_3d.shape[-1]).contiguous() 
        else:
            # [B x T, width, grid, grid]
            x = self.conv1(x)
        # [B x T, width, grid x grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # [B x T, grid x grid, width]
        x = x.permute(0, 2, 1)

        # shape = [B x T, grid x grid + 1, width]
        '''
        if self.frame_embedding is not None:
            frame_embedding = self.frame_embedding.repeat(B,1,1).reshape(B,video_frame,1,self.width)
            #print('frame_embedding',frame_embedding.reshape(BT // video_frame, -1, self.width).shape)
            x = (x.reshape(B, video_frame, -1, self.width) + frame_embedding.to(x.dtype)).reshape(BT, -1, self.width)
        '''
        x = torch.cat([self.class_embedding.to(x.dtype) + \
                        torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  
        
        #x = x + self.positional_embedding[1:,:].to(x.dtype)
        x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)
        
        x = self.incorporate_prompt(x,unified_visual_prompt[0])

        x = x.permute(1, 0, 2)  					# NLD -> LND
        # org forward 
        #x = self.transformer(x, video_frame=video_frame, visual=True)
        x= self.forward_deep_prompt(x,unified_visual_prompt)

        x = x.permute(1, 0, 2)  					# LND -> NLD

        return x

class Prompt_class(nn.Module):
    def __init__(self,
                    # vision
                    vision_width: int=384,
                    transformer_width: int=384,
                    transformer_heads: int=4,
                    unified_text_prompt_length: int=8,
                    unified_prompt_length: int=16,
                    ):

        super().__init__()
        self.shared_latent_space = "transformer"
        self.unified_text_prompt_length = unified_text_prompt_length
        self.unified_prompt_width = 384
        self.unified_prompt_length = unified_text_prompt_length
        self.unified_prompt_layers = 1
        if self.shared_latent_space == "transformer":
            self.unified_prompt_mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(vision_width, vision_width * 2)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(vision_width * 2, transformer_width))
            ]))
            PromptTransformer_heads = self.unified_prompt_width//64
            self.PromptTransformer = PromptTransformer(
                width= self.unified_prompt_width,
                layers= 1,
                heads=PromptTransformer_heads)
            
            self.unified_prompt_tokens = torch.arange(self.unified_prompt_length).long()
            self.unified_prompt_embedding = nn.Embedding(self.unified_prompt_length, self.unified_prompt_width*self.unified_prompt_layers)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype
    
    def encode_prompt(self,batch_size,device, visual=False):
        if self.shared_latent_space == "transformer":
            unified_prompt_tokens = self.unified_prompt_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
            unified_prompt_embedding = self.unified_prompt_embedding(unified_prompt_tokens)
            unified_prompt_embedding = unified_prompt_embedding.view(batch_size,self.unified_prompt_length,self.unified_prompt_layers,self.unified_prompt_width)

            unified_prompt_embedding =  unified_prompt_embedding.permute(2,0,1,3)  ##layers,bz,length,width
            unified_prompt_embedding = unified_prompt_embedding.reshape(self.unified_prompt_layers*batch_size,self.unified_prompt_length,self.unified_prompt_width).permute(1,0,2)
        
            unified_prompt_output = self.PromptTransformer(unified_prompt_embedding)
            unified_prompt_output = unified_prompt_output.permute(1,0,2).view(self.unified_prompt_layers,batch_size,self.unified_prompt_length,self.unified_prompt_width)
            unified_text_prompt = self.unified_prompt_mlp(unified_prompt_output[:,:,:self.unified_text_prompt_length,:])
            if visual:
                unified_visual_prompt = unified_prompt_output[:,:,self.unified_text_prompt_length:,:]
                return unified_text_prompt.squeeze(), unified_visual_prompt
            else:
                return unified_text_prompt.squeeze()

if __name__=='__main__':
    prompt_class = Prompt_class().cuda()
    batch_size = 128
    device = 'cuda'
    unified_text_prompt = prompt_class.encode_prompt(batch_size, device=device)
    print(unified_text_prompt.shape)

