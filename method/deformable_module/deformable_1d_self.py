import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import math
# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

# tensor helpers

def grid_sample_1d(feats, grid, *args, **kwargs):
    # does 1d grid sample by reshaping it to 2d
    grid = rearrange(grid, '... -> ... 1 1')
    grid = F.pad(grid, (1, 0), value = 0.) #网格 BS H——out W_out 2 
    feats = rearrange(feats, '... -> ... 1') #BS dim H W    根据grid的2确定这边的坐标选择  最终会得到BS h_out w_out dim
    out = F.grid_sample(feats, grid, **kwargs)
    return rearrange(out, '... 1 -> ...')



def grid_sample_1d_direct(feats, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    """
    直接执行1D网格采样，避免转换为2D的额外开销
    
    参数:
        feats: 输入特征，形状为 [batch_size, channels, length]
        grid: 采样网格，形状为 [batch_size, length_out]，值范围为 [-1, 1]
        mode: 插值模式，'bilinear' 或 'nearest'
        padding_mode: 填充模式，'zeros'、'border' 或 'reflection'
        align_corners: 是否对齐角点
        
    返回:
        采样结果，形状为 [batch_size, channels, length_out]
    """
    # 确保输入维度正确
    assert feats.dim() == 3, "feats 必须是 [batch_size, channels, length] 形状"
    assert grid.dim() == 2, "grid 必须是 [batch_size, length_out] 形状"
    
    batch_size, channels, in_length = feats.shape
    _, out_length = grid.shape
    
    # 将grid从[-1,1]范围映射到[0, in_length-1]
    if align_corners:
        grid = (grid + 1) * (in_length - 1) / 2
    else:
        grid = (grid + 1) * in_length / 2 - 0.5
    
    # 计算两个最近的整数坐标点（1D中只有左右两个点）
    grid_floor = torch.floor(grid).clamp(0, in_length - 1).long()  # 左侧点
    grid_ceil = torch.ceil(grid).clamp(0, in_length - 1).long()    # 右侧点
    
    # 计算权重（距离比例）
    weight_ceil = grid - grid_floor.float()  # x与左侧点的距离比例
    weight_floor = 1 - weight_ceil           # x与右侧点的距离比例
    
    # 提取两个最近点的值
    feats_floor = feats.gather(2, grid_floor.unsqueeze(1).expand(-1, channels, -1))
    feats_ceil = feats.gather(2, grid_ceil.unsqueeze(1).expand(-1, channels, -1))
    
    # 应用线性插值（1D中的双线性插值实际上就是线性插值）
    if mode == 'bilinear':
        output = weight_floor.unsqueeze(1) * feats_floor + weight_ceil.unsqueeze(1) * feats_ceil
    else:  # 'nearest'
        mask = weight_ceil >= 0.5
        output = torch.where(mask.unsqueeze(1), feats_ceil, feats_floor)
    
    # 处理填充模式
    if padding_mode == 'zeros':
        # 找出超出边界的位置
        outside = (grid < 0) | (grid >= in_length)
        # 将超出边界的位置设为0
        output = output.masked_fill(outside.unsqueeze(1), 0)
    elif padding_mode == 'border':
        # 边界处理已经在clamp操作中完成
        pass
    elif padding_mode == 'reflection':
        # 反射填充实现略复杂，这里省略
        pass
    
    return output

def normalize_grid(arange, dim = 1, out_dim = -1):
    # 实现与torch.nn.functional.grid_sample一致的归一化逻辑
    n = arange.shape[-1]
    # 将坐标从[0, n-1]映射到[-1, 1]
    # 注意：这与之前的实现略有不同，因为grid_sample期望的坐标范围是[-1, 1]
    normalized = 2.0 * arange / (n - 1) - 1.0
    return torch.clamp(normalized, -1.0, 1.0)  # 确保结果在[-1, 1]范围内

class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

# continuous positional bias from SwinV2

class CPB(nn.Module):
    """ https://arxiv.org/abs/2111.09883v1 """

    def __init__(self, dim, *, heads, offset_groups, depth, log_distance = True):
        super().__init__()
        self.heads = heads
        self.offset_groups = offset_groups
        self.log_distance = log_distance

        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(
            nn.Linear(1, dim),
            nn.ReLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU()
            ))

        self.mlp.append(nn.Linear(dim, heads // offset_groups))

    def forward(self, grid_q, grid_kv):
        device, dtype = grid_q.device, grid_kv.dtype

        grid_q = rearrange(grid_q, 'n -> 1 n')
        grid_kv = rearrange(grid_kv, 'b n -> b n')

        pos = rearrange(grid_q, 'b i -> b i 1 1') - rearrange(grid_kv, 'b j -> b 1 j 1')

        if self.log_distance:
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        bias = pos

        for layer in self.mlp:
            bias = layer(bias)

        bias = rearrange(bias, '(b g) i j o -> b (g o) i j', g = self.offset_groups)

        return bias

# main class

class LowRankConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=0.5):
        super().__init__()
        mid_channels = int(in_channels * reduction_ratio)
        self.conv1 = nn.Conv1d(in_channels, mid_channels, 1, bias=False)
        self.conv2 = nn.Conv1d(mid_channels, out_channels, 1, bias=False)
        
    def forward(self, x):
        return self.conv2(self.conv1(x))
    
class DeformableAttention1D(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 4,
        dropout = 0.05,
        downsample_factor = 4,
        offset_scale = None,
        offset_groups = 2,
        offset_kernel_size = 6,
        cpb_log_distance = True,
        group_queries = True,
        group_key_values = True,
        offset_num=4,
        dim_ff=2048,
        position = False
    ):
        super().__init__()
        offset_scale = default(offset_scale, downsample_factor)
        assert offset_kernel_size >= downsample_factor, 'offset kernel size must be greater than or equal to the downsample factor'
        assert divisible_by(offset_kernel_size - downsample_factor, 2)

        offset_groups = default(offset_groups, heads)
        assert divisible_by(heads, offset_groups)

        assert divisible_by(dim, heads)
        dim_head = dim // heads

        inner_dim = dim_head * heads #heads =4  dim_head-> 256
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.offset_groups = offset_groups

        offset_dims = inner_dim // offset_groups
        # print(f'offset_dims :{offset_dims}')
        self.offset_num = offset_num
        self.downsample_factor = downsample_factor
        
        self.to_offsets = nn.Sequential(
            nn.Conv1d(offset_dims, offset_dims, offset_kernel_size, groups = offset_dims, stride = downsample_factor, padding = (offset_kernel_size - downsample_factor) // 2),
            nn.GELU(),
            nn.Conv1d(offset_dims, self.offset_num, 1, bias = False),
            # Rearrange('b 1 n -> b n'),
            nn.Tanh(),
            Scale(offset_scale)
        )
        
        self.dropout = nn.Dropout(dropout)
        ffn_dropout = 0.05
        # self.mlp = 'linear'
        self.mlp = 'linear'
        if self.mlp == 'linear':
            self.to_q = nn.Sequential(
                nn.Linear(dim, inner_dim),
                nn.ReLU(),
                nn.Dropout(ffn_dropout),
                nn.Linear(inner_dim, dim)
            )
            self.to_v = nn.Sequential(
                nn.Linear(dim, inner_dim),
                nn.ReLU(),
                nn.Dropout(ffn_dropout),
                nn.Linear(inner_dim, dim)
            )
            self.to_k = nn.Sequential(
                nn.Linear(dim, inner_dim),
                nn.ReLU(),
                nn.Dropout(ffn_dropout),
                nn.Linear(inner_dim, dim)
            )
        elif self.mlp =='conv1d': ##80.0跑出来的setting
            self.rel_pos_bias = CPB(dim // 4, offset_groups = offset_groups, heads = heads, depth = 2, log_distance = cpb_log_distance)
            self.to_q = nn.Conv1d(dim, inner_dim, 1, groups = offset_groups if group_queries else 1, bias = False)
            self.to_k = nn.Conv1d(dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
            self.to_v = nn.Conv1d(dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
        # self.to_out = nn.Conv1d(inner_dim, dim, 1)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(inner_dim, dim_ff),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(dim_ff, dim)
        )
        self.scales_make_attn_diff = nn.Parameter(torch.tensor(70.0))  # 可学习的缩放

        #新增温度参数，attn weight 拉开。 增大scale，offset 拉开。 新增position embedding。 三个77.0 -》 80+
        self.position = True
        if self.position:
            self.pos_encoder_Q = nn.Sequential(
            nn.Embedding(128, 384),
            nn.LayerNorm(384)
        )
            self.pos_encoder_KV = nn.Sequential(
            nn.Embedding(128, 384),
            nn.LayerNorm(384)
        )
        
    def forward(self, q_raw, kv_raw, return_vgrid = False):
        """
        b - batch
        h - heads
        n - sequence dimension
        d - dimension
        g - offset groups
        """
        
        heads, b, n, downsample_factor, device = self.heads, q_raw.shape[0], q_raw.shape[-1], self.downsample_factor, q_raw.device
        # exit()
        # queries
        if self.position:
            position_ids = torch.arange(q_raw.shape[-1], device=device).unsqueeze(0)  # [1, n]
            pos_encoding = self.pos_encoder_Q(position_ids)  # [1, n, d_model]
            pos_encoding = rearrange(pos_encoding, '1 n d -> 1 d n')  # [1, d_model, n]

            q_raw = q_raw + pos_encoding
            position_ids = torch.arange(kv_raw.shape[-1], device=device).unsqueeze(0)  # [1, n]
            pos_encoding = self.pos_encoder_KV(position_ids)  # [1, n, d_model]
            pos_encoding = rearrange(pos_encoding, '1 n d -> 1 d n')  # [1, d_model, n]
            kv_raw = kv_raw + pos_encoding

        if self.mlp =='linear':
            q = self.to_q(q_raw.transpose(1,2)).transpose(1,2)# bs dim len -> bs dim' len
        else:
            q = self.to_q(q_raw)
        # calculate offsets - offset MLP shared across all groups

        group = lambda t: rearrange(t, 'b (g d) n -> (b g) d n', g = self.offset_groups)

        grouped_queries = group(q)
        offsets = self.to_offsets(grouped_queries) #len为4的时候预测一个偏移量
        # print(offsets.shape)
        # print(offsets[0,:,[20,60]])
        # calculate grid + offsets

        grid = torch.arange(offsets.shape[-1], device = device) #原始位置
        vgrid = grid + offsets #偏移之后的位置

        vgrid_scaled = normalize_grid(vgrid) #归一化之后的位置
        
        vgrid_scaled = rearrange(vgrid_scaled, 'b num_k n -> (b num_k) n') #把numk放到bsz里面
        
        kv_feats = grid_sample_1d(
            rearrange(group(kv_raw).unsqueeze(1).repeat(1, self.offset_num, 1, 1), 'b num_k dim n_len-> (b num_k) dim n_len'), #把x分成多个组，维度拆分了
            vgrid_scaled,
        mode = 'bilinear', padding_mode = 'zeros', align_corners = False)  #根据偏移位置取出kv feats #,在这个过程下采样了？
        #下面这个版本，复现不出来当时ckpt的80.1，上面的版本可以，说明有差距
        # kv_feats = grid_sample_1d_direct(
        #     rearrange(group(kv_raw).unsqueeze(1).repeat(1, self.offset_num, 1, 1), 'b num_k dim n_len-> (b num_k) dim n_len'), #把x分成多个组，维度拆分了
        #     vgrid_scaled,
        # mode = 'bilinear', padding_mode = 'zeros', align_corners = False)  #根据偏移位置取出kv feats #,在这个过程下采样了？
        
        # print(torch.max(torch.abs(kv_feats-kv_feats_1d)))

        vgrid_scaled = rearrange(vgrid_scaled, '(b num_k) n -> b num_k n', num_k=self.offset_num) #把numk取出来
        kv_feats = rearrange(kv_feats, '(b num_k) dim n_len-> b num_k dim n_len', num_k=self.offset_num)
        kv_feats = rearrange(kv_feats, '(b g) num_k d n -> (b num_k) (g d) n', b = b) #取出之后，整合回来
        
        # derive key / values
        # kv_feats = rearrange(kv_feats, 'b num_k d n -> (b num_k) d n', b = b) #把numk放到bsz里面
        # print(kv_feats.shape)
        # import ipdb;ipdb.set_trace()
        if self.mlp =='linear':
            k, v = self.to_k(kv_feats.transpose(1,2)).transpose(1,2), self.to_v(kv_feats.transpose(1,2)).transpose(1,2) #linear to的时候需要先head和dim拼回来
        else:
            k, v = self.to_k(kv_feats), self.to_v(kv_feats)
        # print(k.shape)
        # exit()
        # scale queries

        q = q * self.scale

        # split out heads
        # 预先计算缩放因子，避免重复计算
        headdim = q.size(-1)
       
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h n d', h = heads), (q, k, v)) #to之后再拆head和dim

        # query / key similarity
        q = rearrange(q, 'b h (n k) d -> (b n) h k d', k = 1)#把n len拆出来到batchsize去，因为每一个Q所对应的key是不一样的。不能把所有Q放在一起去和k计算，每一个Q之间是独立的，key也是独立的
        k = rearrange(k, '(b num_k) h (n k) d -> (b n) h (num_k k) d', k = 1, num_k = self.offset_num)
        v = rearrange(v, '(b num_k) h (n k) d -> (b n) h (num_k k) d', k = 1, num_k = self.offset_num)
        
        scale = 1.0 / math.sqrt(headdim)
        
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * scale #q bs 128   k bs 32
        # print(sim.shape)
        # relative positional bias

        # seq_range = torch.arange(n, device = device)
        # seq_scaled = normalize_grid(seq_range, dim = 0)
        # rel_pos_bias = self.rel_pos_bias(seq_scaled, vgrid_scaled)
        # sim = sim + rel_pos_bias

        # numerical stability
        # print(sim[[0, 555]][0])
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        
        # attention
        
        attn = (sim * self.scales_make_attn_diff).softmax(dim = -1)
        
        attn = self.dropout(attn)

       

       
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b (h d) n')
        # out = self.to_out(out)
        out = rearrange(out, '(b n_len) dim k ->b dim (n_len k)', n_len=n)
        out = self.norm1(q_raw.transpose(1,2) + out.transpose(1,2))
        out = out + self.ffn(out)
        out = self.norm2(out).transpose(1,2)
        # import ipdb;ipdb.set_trace()
        if return_vgrid:
            return out, vgrid

        return out

if __name__=='__main__':
    import torch
    # from deformable_attention import DeformableAttention1D

    attn = DeformableAttention1D(
        dim = 384,
        downsample_factor = 1,
        offset_scale = 2,
        heads = 8,
        offset_groups = 8,
        offset_num = 8,
        position = False,
        offset_kernel_size = 3
    )

    from thop import profile
    x = torch.randn(1, 384, 128)
    kv = torch.randn(1, 384, 128)

    flops, params = profile(attn, inputs=(x, kv,))

    # 打印计算结果
    print(f"模型参数量: {params/1e6:.2f} M params")  # 转换为M单位
    print(f"模型计算量: {flops/1e9:.2f} G FLOPs")    # 转换为G单位    
    
