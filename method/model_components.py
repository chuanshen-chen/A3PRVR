import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from vim_block import create_block
def _get_atm_loss(
        text_feat,
        v_feat_cat,
    ):

        sim_t2q = torch.matmul(
            F.normalize(text_feat, dim=-1),
            F.normalize(v_feat_cat, dim=-1).transpose(0,1),
        ) #bsz 384 * bsz*2 384

        bs = sim_t2q.shape[0]

        bsz = torch.arange(bs, device=sim_t2q.device).unsqueeze(-1)
        bsz_reverse = bsz + bs
        pair = torch.cat((bsz, bsz_reverse), dim=-1)
        logits = sim_t2q[bsz, pair] / 0.07
        targets = torch.zeros(logits.shape[0],dtype=int).to(logits.device)
        loss_atm= F.cross_entropy(logits, targets, label_smoothing=0.1)
        return loss_atm
    
def _get_contrastive_loss(
        video_feat, 
        # text_feat, 
        # video_feat_all, 
        text_feat_all, 
    ):

        sim_v2t = torch.matmul(
            F.normalize(video_feat, dim=-1), # (batch_size,  latent_dim)  
            F.normalize(text_feat_all, dim=-1).transpose(0,1),  # (batch_size*2 latent_dim)
        ) 

        bs = video_feat.shape[0]

        bsz = torch.arange(bs, device=sim_v2t.device).unsqueeze(-1)
        bsz_reverse = bsz + bs
        pair = torch.cat((bsz, bsz_reverse), dim=-1)
        logits = sim_v2t[bsz, pair] / 0.07
        targets = torch.zeros(logits.shape[0],dtype=int).to(logits.device)
        loss_vac= F.cross_entropy(logits, targets, label_smoothing=0.1)

        return loss_vac
        

        
class MultiheadAttn(nn.Module):
    def __init__(self, dim, nhead, dropout=0.1, ret_att=False):
        super().__init__()
        self.dim = dim
        self.nhead = nhead
        self.head_dim = int(dim // nhead)
        assert self.nhead * self.head_dim == self.dim
        self.linears = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)])
        # dropout=0
        if dropout == 0:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(dropout)
        self.ret_att = ret_att
    
    def attention(self, queries, keys, values, mask=None, dropout=None):
        """
            queries: B x H x S x headdim
            keys: B x H x L x headdim
            values: B x H x L x headdim
            mask: B x 1 x S x L
        """
        headdim = queries.size(-1)
        
        scores = queries @ keys.transpose(-1, -2) / math.sqrt(headdim)  # B x H x S x L
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = torch.softmax(scores, dim=-1)
        
        if dropout is not None:
            scores = dropout(scores)
        return scores @ values, scores  # B x H x S x headdim

    def forward(self, query, key, value, mask=None, sum_seq=False):
        """
            query: B x S x D
            key: B x L x D
            value: B x L x D
            mask: B x S x L
        """
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)  # B x 1 x S x L, 1 for heads
        queries, keys, values = [
            layer(x).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
            for layer, x in zip(self.linears[:3], (query, key, value))
        ]  # B x H x S|L x head_dim
        
        result, att = self.attention(queries, keys, values, mask, self.dropout)  # B x H x S x headdim
        if sum_seq:
            result = result.sum(2, keepdim=True)  # B x H x 1 x headdim
        result = result.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        # B x S x D / (if sum_seq)B x 1 x D
        if self.ret_att:
            return self.linears[-1](result), att.mean(dim=1)
        else:
            return self.linears[-1](result)
        

class AttentionModule(nn.Module):
    def __init__(self, dim, dim_ff, n_head, msa_dropout, ffn_dropout, ret_att=False):
        super().__init__()
        self.dim = dim
        self.msa = MultiheadAttn(dim, n_head, dropout=msa_dropout, ret_att=ret_att)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(dim_ff, dim)
        )
        self.ret_att = ret_att

    def forward(self, q, k, v, mask, add_q=True):
        
        if self.ret_att:
            msa, att = self.msa(q, k, v, mask)
        else:
            msa = self.msa(q, k, v, mask)
            att = None
        if add_q:
            x = self.norm1(q + msa)
        else:
            x = self.norm1(v + msa)

        x = x + self.ffn(x)
        x = self.norm2(x)
        return x, att
        
class TransformerFilter(nn.Module):
    """
                    | CA |      | CA |
                    | CA | 
                    | SA |      | SA |         
                | Obj Feat | | Text Feat | | Point Feat |
        Obj Feat:  B C Pq
        Text Feat: 
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.05, ret_att=False):
        super().__init__()

        self.ca = AttentionModule(d_model, dim_feedforward, nhead, dropout, dropout, ret_att)

        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        # self.ffn = nn.Sequential(
        #     nn.Linear(d_model, dim_feedforward),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(dim_feedforward, d_model)
        # )

    # def with_pos_embed(self, tensor, pos_embed: Optional[Tensor]):
    #     return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, q_feat, kv_feat, q_mask, kv_mask):
        """
        object_feat: [B,K,C]     object_mask: [B,K]      object_pose: [B,K,6]
        point_feat: [B,N,C]                              point_pose: [B,N,3]
        lang_feat: [B,M,C]       lang_mask: [B,M]        lang_pose:[B,M,1]
        """

        # add mask to 1 dim
        q_mask = q_mask.unsqueeze(-1)     # [B, K, 1]
        kv_mask = kv_mask.unsqueeze(-1)         # [B, M, 1]

        # object ca lang layer
        mask = q_mask * kv_mask.transpose(-1, -2)    # [B, K, M]
        cross_object_feat, cross_obj_att = self.ca(q_feat, kv_feat, kv_feat, mask)
        
        return cross_object_feat, cross_obj_att
        


def onehot(indexes, N=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().long().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    return output

def kl_divergence_loss(p, q, epsilon=1e-10):
    # 添加一个微小的数值，以避免除以零的情况
    p = p + epsilon
    q = q + epsilon
    
    # 计算KL散度
    kl_loss = torch.sum(p * (torch.log(p) - torch.log(q)), dim=1)
    
    # 返回平均KL散度
    return torch.mean(kl_loss)


class clip_nce(nn.Module):
    def __init__(self, reduction='mean'):
        super(clip_nce, self).__init__()
        self.reduction = reduction
        
    def topk_indices(self, matrix, k):
        # 对每一行取topk索引
        row_topk_indices = torch.topk(matrix, k, dim=1)[1]
        # 对每一列取topk索引
        col_topk_indices = torch.topk(matrix, k, dim=0)[1]

        return row_topk_indices, col_topk_indices
    
    def calculate_topk_match(self, matrix, row_topk_indices, col_topk_indices):
        # 创建一个与原始矩阵相同形状的矩阵，用于存储匹配结果
        match_matrix1 = torch.zeros_like(matrix, dtype=torch.int)
        match_matrix2 = torch.zeros_like(matrix, dtype=torch.int)
        bsz = torch.arange(matrix.shape[0]).unsqueeze(-1)
        bsz2 = torch.arange(col_topk_indices.shape[-1]).unsqueeze(-1)
        match_matrix1[bsz, row_topk_indices] = 1
        match_matrix2[col_topk_indices.transpose(1,0), bsz2] = 1
        
        match_matrix = torch.logical_and(match_matrix1, match_matrix2)

        return match_matrix
    
    def forward(self,labels, label_dict, topk=0.2, q2ctx_scores=None, contexts=None, queries=None, loss_weight_matrix=None, many2many=False):
        
        if many2many:
            row_topk_indices, col_topk_indices = self.topk_indices(q2ctx_scores.squeeze(), 5)
            # 计算匹配结果/
            topk_reciprocal = self.calculate_topk_match(q2ctx_scores.squeeze(), row_topk_indices, col_topk_indices) #若true，则表示是彼此的topk
            topk_reciprocal = torch.logical_not(topk_reciprocal) + 1 #true的为0，false的为1 然后false的加权重
            t2v_weight = topk_reciprocal * topk_reciprocal.shape[-1] / torch.sum(topk_reciprocal, dim=-1, keepdim=True)
        else:
            t2v_weight = torch.ones_like(q2ctx_scores, device=q2ctx_scores.device)
            
        query_bsz = q2ctx_scores.shape[0]
        vid_bsz = q2ctx_scores.shape[1]
        diagnoal = torch.arange(query_bsz).to(q2ctx_scores.device)
        t2v_nominator = q2ctx_scores[diagnoal, labels] #取query对应的video出来作为正样本

        t2v_nominator = torch.logsumexp(t2v_nominator.unsqueeze(1), dim=1)
        t2v_denominator = torch.logsumexp(q2ctx_scores, dim=1)
        pos_topk = False
        if pos_topk:
            _, top_indices = t2v_nominator.topk(int(topk * t2v_nominator.shape[0]), largest=False)
            pos_t2v_loss_weight = torch.ones_like(t2v_nominator)
            pos_t2v_loss_weight[top_indices] = 0
        
        v2t_nominator = torch.zeros(vid_bsz).to(q2ctx_scores)
        v2t_denominator = torch.zeros(vid_bsz).to(q2ctx_scores)

        for i, label in label_dict.items():
            v2t_nominator[i] = torch.logsumexp(q2ctx_scores[label, i], dim=0)
            if many2many:
                v2t_weight = topk_reciprocal[:, i] * topk_reciprocal[:, i].shape[0] / torch.sum(topk_reciprocal[:, i], dim=-1)
            else:
                v2t_weight = torch.ones_like(q2ctx_scores[:, i], device=q2ctx_scores[:, i].device)
            v2t_denominator[i] = torch.logsumexp(q2ctx_scores[:, i], dim=0)
        if self.reduction:
            if pos_topk:
                pos_t2v_loss_weight += 1
                return torch.sum((t2v_denominator - t2v_nominator) * pos_t2v_loss_weight) / torch.sum(pos_t2v_loss_weight) + torch.mean(v2t_denominator - v2t_nominator)
            else:
                return torch.mean(t2v_denominator - t2v_nominator) + torch.mean(v2t_denominator - v2t_nominator)
        else:
            return denominator - nominator


class frame_nce(nn.Module):
    def __init__(self, reduction='mean'):
        super(frame_nce, self).__init__()
        self.reduction = reduction
    def topk_indices(self, matrix, k):
        # 对每一行取topk索引
        row_topk_indices = torch.topk(matrix, k, dim=1)[1]
        # 对每一列取topk索引
        col_topk_indices = torch.topk(matrix, k, dim=0)[1]

        return row_topk_indices, col_topk_indices
    
    def calculate_topk_match(self, matrix, row_topk_indices, col_topk_indices):
        # 创建一个与原始矩阵相同形状的矩阵，用于存储匹配结果
        match_matrix1 = torch.zeros_like(matrix, dtype=torch.int)
        match_matrix2 = torch.zeros_like(matrix, dtype=torch.int)
        bsz = torch.arange(matrix.shape[0]).unsqueeze(-1)
        match_matrix1[bsz, row_topk_indices] = 1
        match_matrix2[col_topk_indices.transpose(1,0), bsz] = 1
        
        match_matrix = torch.logical_and(match_matrix1, match_matrix2)

        return match_matrix
    
    def forward(self, q2ctx_scores=None, contexts=None, queries=None, loss_weight=None, pos_radio_loss_weight=None, many2many=False):
        if q2ctx_scores is None:
            assert contexts is not None and queries is not None
            x = torch.matmul(contexts, queries.t())
            device = contexts.device
            bsz = contexts.shape[0]
        else:
            x = q2ctx_scores
            device = q2ctx_scores.device
            bsz = q2ctx_scores.shape[0]

        x = x.view(bsz, bsz, -1)
        if many2many:
            row_topk_indices, col_topk_indices = self.topk_indices(x.squeeze(), 5)
            # 计算匹配结果/
            topk_reciprocal = self.calculate_topk_match(x.squeeze(), row_topk_indices, col_topk_indices) #若true，则表示是彼此的topk
        
        nominator = x * torch.eye(x.shape[0], dtype=torch.float32, device=device)[:, :, None]
        
        nominator = nominator.sum(dim=1)
        
        nominator = torch.logsumexp(nominator, dim=1)

        denominator = torch.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1)
        
        if loss_weight is None:
            loss_weight = torch.ones_like(x.squeeze())
        if many2many:
            loss_weight[topk_reciprocal] = 0 #把那些互相匹配的topk，也降低权重，他们不一定是负样本。
            
        new_loss_weight = loss_weight + 1
        new_loss_weight = new_loss_weight.unsqueeze(-1)
        new_loss_weight = torch.cat((new_loss_weight, new_loss_weight.permute(1, 0, 2)), dim=1).view(new_loss_weight.shape[0], -1)
        new_loss_weight = new_loss_weight * new_loss_weight.shape[-1] / torch.sum(new_loss_weight, dim=-1, keepdim=True)
        denominator = torch.logsumexp(denominator * new_loss_weight, dim=1)

        if self.reduction:
            if pos_radio_loss_weight is not None:
                return torch.sum((denominator - nominator) * pos_radio_loss_weight) / torch.sum(pos_radio_loss_weight)
            else:
                return torch.mean(denominator - nominator)
        else:
            return denominator - nominator

class TrainablePositionalEncoding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, max_position_embeddings, hidden_size, dropout=0.1):
        super(TrainablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_feat):
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(input_feat + position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def add_position_emb(self, input_feat):
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        return input_feat + position_embeddings


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""
    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [nn.Dropout(dropout), nn.Linear(in_hsz, out_hsz)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


class BertLayer(nn.Module):
    def __init__(self, config, use_self_attention=True):
        super(BertLayer, self).__init__()
        self.use_self_attention = use_self_attention
        if use_self_attention:
            self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        """
        Args:
            hidden_states:  (N, L, D)
            attention_mask:  (N, L) with 1 indicate valid, 0 indicates invalid
        """
        if self.use_self_attention:
            attention_output = self.attention(hidden_states, attention_mask) 
        else:
            attention_output = hidden_states
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        if config.mamba > 0:
            self.mamba_list = nn.ModuleList()
            for i in range(config.mamba):
                self.mamba_list.append(create_block(
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
                ))
        else:
            self.self = BertSelfAttention(config)
        # self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.mamba = config.mamba
    def forward(self, input_tensor, attention_mask=None):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, L)
        """
        if self.mamba > 0:
            residual = None
            mamba_output = input_tensor
            for i in range(self.mamba):
                mamba_output, residual = self.mamba_list[i](mamba_output, residual)
            attention_output = self.output(mamba_output, input_tensor)
            return attention_output
        else:
            self_output = self.self(input_tensor, input_tensor, input_tensor, attention_mask)
            attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Sequential(nn.Linear(config.hidden_size, config.intermediate_size), nn.ReLU(True))

    def forward(self, hidden_states):
        return self.dense(hidden_states)


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (
                config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query_states, key_states, value_states, attention_mask=None):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)
        """
        # only need to mask the dimension where the softmax (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        # attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)
        # transpose
        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.  # (N, 1, Lq, L)
            attention_scores = attention_scores + attention_mask
        # attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # compute output context
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class DynamicGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0, bidirectional=False):
        super().__init__()
        self.batch_first = batch_first
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, bias=bias,
                          batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        self.gru.flatten_parameters()

    def forward(self, x, seq_len, max_num_frames):
        sorted_seq_len, sorted_idx = torch.sort(seq_len, dim=0, descending=True)
        _, original_idx = torch.sort(sorted_idx, dim=0, descending=False)
        if self.batch_first:
            sorted_x = x.index_select(0, sorted_idx)
        else:
            sorted_x = x.index_select(1, sorted_idx)

        packed_x = nn.utils.rnn.pack_padded_sequence(
            sorted_x, sorted_seq_len.cpu().data.numpy(), batch_first=self.batch_first)

        out, state = self.gru(packed_x)

        unpacked_x, unpacked_len = nn.utils.rnn.pad_packed_sequence(out, batch_first=self.batch_first)

        if self.batch_first:
            out = unpacked_x.index_select(0, original_idx)
            if out.shape[1] < max_num_frames:
                out = F.pad(out, [0, 0, 0, max_num_frames - out.shape[1]])
        else:
            out = unpacked_x.index_select(1, original_idx)
            if out.shape[0] < max_num_frames:
                out = F.pad(out, [0, 0, 0, 0, 0, max_num_frames - out.shape[0]])

        return out
    
    
def margin_ranking_loss(
        similary_matrix, 
        margin=None, 
        direction= 'both', 
        average_batch = True, 
        ):
    
    batch_size = similary_matrix.size(0)
    diagonal = similary_matrix.diag().view(batch_size, 1)
    pos_mask = torch.eye(batch_size,batch_size,device=similary_matrix.device).bool()
    # v2c
    if direction == 'both' or direction == 's2n':
        diagonal_1 = diagonal.expand_as(similary_matrix)
        cost_cap = (margin + similary_matrix - diagonal_1).clamp(min=0)
        cost_cap = cost_cap.masked_fill(pos_mask, 0)
        if average_batch:
            cost_cap = cost_cap / (batch_size * (batch_size - 1))
            cost_cap = torch.sum(cost_cap)
    # c2v
    if direction == 'both' or direction == 'n2s':
        diagonal_2 = diagonal.t().expand_as(similary_matrix)
        cost_vid = (margin + similary_matrix - diagonal_2).clamp(min=0)
        cost_vid = cost_vid.masked_fill(pos_mask,0)
        if average_batch:
            cost_vid = cost_vid / (batch_size * (batch_size - 1))
            cost_vid = torch.sum(cost_vid)
    
    if direction == 'both':
        return cost_cap + cost_vid
    elif direction == 's2n':
        return cost_cap
    else:
        return cost_vid