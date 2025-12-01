import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
from method.model_components import BertAttention, LinearLayer, BertSelfAttention, TrainablePositionalEncoding, TransformerFilter, clip_nce, frame_nce, margin_ranking_loss
#, TransformerFilter, DynamicGRU _get_contrastive_loss, _get_atm_loss,

# import sys
from einops import rearrange
import time
from transformers import RobertaTokenizer, RobertaModel
import json
import os

def l2_normalize_tensor(tensor, eps=1e-5):
    """tensor: torch.Tensor, (*, D), where the last dim will be normalized"""
    return tensor / (torch.norm(tensor, p=2, dim=-1, keepdim=True) + eps)

def gaussian_distribution_batch(a, weight_len=128, sigma=1):

    j = torch.arange(weight_len, device=a.device).unsqueeze(0)
    if len(a.shape)==3:
        j = j.unsqueeze(0)
    # 计算高斯分布
    weight = (1.0 / (2 * 3.1415926)) * torch.exp(-(j - a) ** 2 / (sigma ** 2))
    mask_matrix = torch.abs(j - a) > 1
    weight[mask_matrix] = -1e9
    return weight
def gaussian_distribution_batch_loss(a, weight_len=128, sigma=1):

    j = torch.arange(weight_len, device=a.device).unsqueeze(0)
    if len(a.shape)==3:
        j = j.unsqueeze(0)
    # 计算高斯分布
    weight = (1.0 / (2 * 3.1415926)) * torch.exp(-(j - a) ** 2 / (sigma ** 2))
    return weight
#/mnt/cephfs/home/shenqingwei/ANACONDA3/envs/vim
    
class MS_SL_Net(nn.Module):
    def __init__(self, config):
        super(MS_SL_Net, self).__init__()
        self.config = config
        if self.config.mamba > 0:
            self.query_encoder = BertAttention(edict(mamba=config.mamba, hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                    hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                    attention_probs_dropout_prob=config.drop))
        else:
            self.query_encoder = BertAttention(edict(mamba=config.mamba, hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                    hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                    attention_probs_dropout_prob=config.drop))
            
        self.query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l,
                                                           hidden_size=config.hidden_size, dropout=config.input_drop)
        self.query_input_proj = LinearLayer(config.q_feat_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        
        self.channel_proj = nn.Sequential(
            nn.Linear(1024, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

        self.frame_input_proj = LinearLayer(config.visual_feat_dim, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.frame_encoder = copy.deepcopy(self.query_encoder)
            
        self.frame_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                          hidden_size=config.hidden_size, dropout=config.input_drop)
        
        # self.frame_pos_embed_2 = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
        #                                                   hidden_size=config.hidden_size, dropout=config.input_drop)
        # self.frame_input_proj_2 = LinearLayer(config.hidden_size, config.hidden_size, layer_norm=True,
        #                                      dropout=config.input_drop, relu=True)
        # self.frame_encoder_2 = copy.deepcopy(self.query_encoder)
        
        #
        
        if self.config.cross_branch_fusion:
            #nn.ModuleList
            self.cross_branch_fusion_SA = TransformerFilter(config.hidden_size, 4, dim_feedforward=2048, dropout=0.05, ret_att=False)
            self.cross_branch_fusion_SB = TransformerFilter(config.hidden_size, 4, dim_feedforward=2048, dropout=0.05, ret_att=False)
            if self.config.deformable_attn:
                if self.config.deformable_key == 'shared':
                    #这个是没用到的
                    from method.deformable_module.deformable_1d import DeformableAttention1D
                    self.cross_branch_fusion_A1 = DeformableAttention1D(dim = config.hidden_size, downsample_factor = config.downsample_factor, offset_scale = 2, heads=self.config.deformable_heads, 
                                                                        offset_groups=self.config.deformable_offset_groups, offset_kernel_size = config.deformable_offset_kernel_size,
                                                                        offset_num=1)
                    # self.cross_branch_fusion_B1 = DeformableAttention1D(dim = config.hidden_size, downsample_factor = 1, offset_scale = 2, heads=self.config.deformable_heads, 
                    #                                                     offset_groups=self.config.deformable_offset_groups, offset_kernel_size = self.config.deformable_offset_kernel_size,
                    #                                                     offset_num=1)
                elif self.config.deformable_key == 'private':
                    from method.deformable_module.deformable_1d_self import DeformableAttention1D
                    self.cross_branch_fusion_A1 = DeformableAttention1D(dim = config.hidden_size, downsample_factor = 1, offset_scale = config.deformable_offset_scale, heads=self.config.deformable_heads, 
                                                                        offset_groups=self.config.deformable_offset_groups, offset_kernel_size = self.config.deformable_offset_kernel_size,
                                                                        offset_num=self.config.deformable_offset_num)
                    # self.cross_branch_fusion_B1 = DeformableAttention1D(dim = config.hidden_size, downsample_factor = 1, offset_scale = 2, heads=self.config.deformable_heads, 
                    #                                                     offset_groups=self.config.deformable_offset_groups, offset_kernel_size = self.config.deformable_offset_kernel_size,
                    #                                                     offset_num=self.config.deformable_offset_num)
            else:
                self.cross_branch_fusion_A1 = TransformerFilter(config.hidden_size, config.cross_attn_head, dim_feedforward=2048, dropout=0.05, ret_att=True)
            if self.config.cross_attn_q=='double_q':#只有double的时候使用这个初始化，否则保持原来的
                self.cross_branch_fusion_B1 = DeformableAttention1D(dim = config.hidden_size, downsample_factor = 1, offset_scale = 2, heads=self.config.deformable_heads, 
                                                                        offset_groups=self.config.deformable_offset_groups, offset_kernel_size = self.config.deformable_offset_kernel_size,
                                                                        offset_num=self.config.deformable_offset_num)
            else:
                self.cross_branch_fusion_B1 = TransformerFilter(config.hidden_size, config.cross_attn_head, dim_feedforward=2048, dropout=0.05, ret_att=False)

        if self.config.qformer>0:
            self.qformer_selfa = TransformerFilter(config.hidden_size, 4, dim_feedforward=1024, dropout=0.05, ret_att=False)
            self.qformer_a = TransformerFilter(config.hidden_size, 4, dim_feedforward=1024, dropout=0.05, ret_att=False)
            self.qformer_selfb = TransformerFilter(config.hidden_size, 4, dim_feedforward=1024, dropout=0.05, ret_att=False)
            self.qformer_b = TransformerFilter(config.hidden_size, 4, dim_feedforward=1024, dropout=0.05, ret_att=False)
            self.learnable_action_queries = nn.Parameter(torch.randn(1, self.config.qformer, config.hidden_size))
            self.learnable_object_queries = nn.Parameter(torch.randn(1, self.config.qformer, config.hidden_size))
            self.learnable_queries_mask = torch.ones(1, self.config.qformer)
        if self.config.learnable_text_prompt > 0:
            self.prompt_class = Prompt_class(unified_text_prompt_length=self.config.learnable_text_prompt)
            self.prompt_text_attn = TransformerFilter(config.hidden_size, 4, dim_feedforward=2048, dropout=0.05, ret_att=False)
            self.prompt_class_action = Prompt_class(unified_text_prompt_length=self.config.learnable_text_prompt)
            self.prompt_text_attn_action = TransformerFilter(config.hidden_size, 4, dim_feedforward=2048, dropout=0.05, ret_att=False)
        if self.config.phrase_action_branch:
            self.query_pos_embed_phrase = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l,
                                                           hidden_size=config.hidden_size, dropout=config.input_drop)
            self.query_input_proj_phrase = LinearLayer(config.q_feat_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
            self.modular_vector_mapping_phrase = nn.Linear(config.hidden_size, out_features=1, bias=False)
            self.channel_proj_action = nn.Sequential(
            nn.Linear(config.visual_flow_feat_dim, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(config.hidden_size, config.hidden_size)
            )
            if self.config.mamba > 0:
                self.query_encoder_phrase = BertAttention(edict(mamba=config.mamba, hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                    hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                    attention_probs_dropout_prob=config.drop))
            else:
                self.query_encoder_phrase = BertAttention(edict(mamba=config.mamba, hidden_size=config.hidden_size, intermediate_size=config.hidden_size,
                                                    hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,
                                                    attention_probs_dropout_prob=config.drop))
            
            self.frame_input_proj_action = LinearLayer(config.visual_flow_feat_dim if config.flow_feat else config.visual_feat_dim, config.hidden_size, layer_norm=True,
                                             dropout=config.input_drop, relu=True)
            self.frame_encoder_action = copy.deepcopy(self.query_encoder_phrase)

            self.frame_pos_embed_action = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                          hidden_size=config.hidden_size, dropout=config.input_drop)
            
            self.mapping_linear_phrase_action = nn.ModuleList([nn.Linear(config.hidden_size, out_features=config.hidden_size, bias=False)
                                                for i in range(3)])
        
        self.mapping_linear = nn.ModuleList([nn.Linear(config.hidden_size, out_features=config.hidden_size, bias=False)
                                                for i in range(3)])
        self.modular_vector_mapping = nn.Linear(config.hidden_size, out_features=1, bias=False)

        self.clip_nce_criterion = clip_nce(reduction='mean')
        self.video_nce_criterion = frame_nce(reduction='mean')
        self.reset_parameters()
        self.temp = nn.Parameter(1 / 0.07 * torch.ones([]))
        self.temp.requires_grad = True
        self.temp2 = nn.Parameter(1 / 0.07 * torch.ones([]))
        self.temp2.requires_grad = True
        
        if self.config.neg_query_loss in ['both', 'action', 'object']:
            # if self.config.dataset_name == 'tvr':
            #     self.roberta_model = RobertaModel.from_pretrained('/share/home/chenyaofo/project/chenchuanshen/tvr/output_bs_128/sub_query/roberta-base_tuned_model').cuda()
            #     self.roberta_tokenizer = RobertaTokenizer.from_pretrained('/share/home/chenyaofo/project/chenchuanshen/tvr/output_bs_128/sub_query/roberta-base_tuned_model')
            if self.config.dataset_name != 'tvr':
                self.roberta_model = RobertaModel.from_pretrained('./checkpoints/roberta-large').cuda()
                self.roberta_tokenizer = RobertaTokenizer.from_pretrained('./checkpoints/roberta-large')
            
    def reset_parameters(self):
        """ Initialize the weights."""

        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(re_init)

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        """use_hard_negative: bool; hard_pool_size: int, """
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size
    

    def forward(self, batch,
                epoch_now=0):
        frame_video_feat = batch['frame_video_features'] #bs/2 128 512
        frame_video_mask = batch['videos_mask']
        query_feat = batch['text_feat'] #
        query_mask = batch['text_mask']
        query_labels = batch['text_labels']
        frame_flow_video_feat=batch['frame_video_flow_features']
        frame_flow_video_mask=batch['videos_flow_mask']
        final_neg_inputs = batch['final_neg_inputs']
        # print(final_neg_inputs)
        
        encoded_frame_feat = self.encode_context(
        frame_video_feat, frame_video_mask)
        
        if self.config.phrase_action_branch:
            encoded_frame_feat_action = self.encode_context(
            frame_flow_video_feat if self.config.flow_feat else frame_video_feat, frame_flow_video_mask if self.config.flow_feat else frame_video_mask, phrase_action_branch=True)
        else:
            encoded_frame_feat_action = None

        if self.config.cross_branch_fusion:
            #I3D作为Q clip作为key value
            if self.config.deformable_attn:
                if self.config.cross_attn_q == 'i3d_q':
                    # import ipdb;ipdb.set_trace()
                    cross_encoded_frame_feat = self.cross_branch_fusion_A1(encoded_frame_feat.transpose(1,2), encoded_frame_feat_action.transpose(1,2).detach())
                    encoded_frame_feat = cross_encoded_frame_feat.transpose(1,2)
                elif self.config.cross_attn_q == 'clip_q':
                    cross_encoded_frame_feat_action = self.cross_branch_fusion_A1(encoded_frame_feat_action.transpose(1,2), encoded_frame_feat.transpose(1,2).detach())
                    encoded_frame_feat_action = cross_encoded_frame_feat_action.transpose(1,2)
                elif self.config.cross_attn_q == 'double_q':
                    cross_encoded_frame_feat = self.cross_branch_fusion_A1(encoded_frame_feat.transpose(1,2), encoded_frame_feat_action.transpose(1,2).detach())
                    cross_encoded_frame_feat_action = self.cross_branch_fusion_B1(encoded_frame_feat_action.transpose(1,2), encoded_frame_feat.transpose(1,2).detach())
                    encoded_frame_feat = cross_encoded_frame_feat.transpose(1,2)
                    encoded_frame_feat_action = cross_encoded_frame_feat_action.transpose(1,2)
            else:
                cross_encoded_frame_feat, attn = self.cross_branch_fusion_A1(encoded_frame_feat, encoded_frame_feat_action.detach(), frame_video_mask, frame_flow_video_mask.detach())
                encoded_frame_feat = cross_encoded_frame_feat
        
        if self.config.neg_query_loss in ['action', 'object', 'both']:
            neg_inputs = self.roberta_tokenizer(final_neg_inputs, padding=True, return_tensors="pt")
            input_ids = neg_inputs['input_ids'].to(self.roberta_model.device) #输入的token ids
            query_mask_neg = neg_inputs['attention_mask'].to(self.roberta_model.device)  # attention mask
            with torch.no_grad():  # 在推理时不需要计算梯度
                start_time = time.time()
                self.roberta_model.eval()
                with torch.cuda.amp.autocast():
                    query_feat_neg = self.roberta_model(input_ids, attention_mask=query_mask_neg).last_hidden_state #'hello'
                print(f"forward_roberta time: {time.time() - start_time}")
                query_feat_neg = rearrange(query_feat_neg, '(bs neg_num) max_len dim -> bs neg_num max_len dim', bs=query_feat.shape[0])
                query_mask_neg = rearrange(query_mask_neg, '(bs neg_num) max_len -> bs neg_num max_len', bs=query_feat.shape[0])
                query_feat_neg = query_feat_neg[:,:,:self.config.max_desc_l]
                query_feat_neg = l2_normalize_tensor(query_feat_neg)
                query_mask_neg = query_mask_neg[:,:,:self.config.max_desc_l]
        else:
            query_feat_neg = None
            query_mask_neg = None

       

        return {
            "query_feat" : query_feat,
            "query_mask" : query_mask,
            "query_labels" : query_labels,
            "encoded_frame_feat" : encoded_frame_feat,
            "frame_video_mask" : frame_video_mask,
            "query_feat_neg" : query_feat_neg,
            "query_mask_neg" :query_mask_neg,
            "encoded_frame_feat_action" : encoded_frame_feat_action,
            "frame_flow_video_mask" : frame_flow_video_mask,
        }

    def get_all_loss(self, feat_dict):
        query_feat = feat_dict['query_feat']
        query_mask = feat_dict['query_mask']
        query_labels = feat_dict['query_labels']
        encoded_frame_feat = feat_dict['encoded_frame_feat']
        frame_video_mask = feat_dict['frame_video_mask']
        query_feat_neg = feat_dict['query_feat_neg']
        query_mask_neg = feat_dict['query_mask_neg']
        encoded_frame_feat_action = feat_dict['encoded_frame_feat_action']
        frame_flow_video_mask = feat_dict['frame_flow_video_mask']
        
        clip_scale_scores,  clip_scale_scores_, video_query, neg_query_loss \
            = self.get_pred_from_raw_query(
            query_feat, query_mask, query_labels, encoded_frame_feat, frame_video_mask, 
            return_query_feats=True, neg_query_feat=query_feat_neg, neg_query_mask=query_mask_neg)
       
        if self.config.phrase_action_branch:
            clip_scale_scores_action, clip_scale_scores__action, video_query_action, neg_query_loss_action \
                = self.get_pred_from_raw_query(
                query_feat, query_mask, query_labels, encoded_frame_feat_action, frame_flow_video_mask if self.config.flow_feat else frame_video_mask, 
                return_query_feats=True, phrase_action_branch=True, neg_query_feat=query_feat_neg, neg_query_mask=query_mask_neg)       
        else:
            neg_query_loss_action = 0.0
        label_dict = {}  
        
        for index, label in enumerate(query_labels):
            if label in label_dict:
                label_dict[label].append(index)
            else:
                label_dict[label] = []
                label_dict[label].append(index)
        #lossss
        if self.config.best_frame_loss:
            clip_nce_loss = 0.02 * self.clip_nce_criterion(query_labels, label_dict, q2ctx_scores=clip_scale_scores_)
            
            clip_trip_loss = self.get_clip_triplet_loss(clip_scale_scores, query_labels)
            if self.config.phrase_action_branch:

                clip_nce_loss_action = 0.02 * self.clip_nce_criterion(query_labels, label_dict, q2ctx_scores=clip_scale_scores__action)
                clip_trip_loss_action = self.get_clip_triplet_loss(clip_scale_scores_action, query_labels)
            else:
                clip_nce_loss_action = 0.0
                clip_trip_loss_action = 0.0
        else:
            clip_nce_loss = 0.0
            clip_trip_loss = 0.0
            clip_nce_loss_action = 0.0
            clip_trip_loss_action = 0.0
        
        loss =  clip_nce_loss_action + clip_trip_loss_action + clip_nce_loss + clip_trip_loss + neg_query_loss + neg_query_loss_action

        return loss, {"loss_overall": float(loss), 
                      'clip_nce_loss': clip_nce_loss,'clip_trip_loss': clip_trip_loss,
                      'clip_nce_loss_action': clip_nce_loss_action,'clip_trip_loss_action': clip_trip_loss_action,
                      'neg_query_loss': neg_query_loss, #是i3d的
                      'neg_query_loss_action': neg_query_loss_action #是clip的
                      }
    def encode_query(self, query_feat, query_mask, phrase_action_branch=False):
        if phrase_action_branch:
            encoded_query = self.encode_input(query_feat, query_mask, self.query_input_proj_phrase, self.query_encoder_phrase,
                                                self.query_pos_embed_phrase)  # (N, Lq, D)
            # encoded_query = self.encode_input(query_feat, query_mask, self.query_input_proj_phrase, self.query_encoder_phrase)  # (N, Lq, D)
        else:
            encoded_query = self.encode_input(query_feat, query_mask, self.query_input_proj, self.query_encoder,
                                          self.query_pos_embed)  # (N, Lq, D)
        video_query = self.get_modularized_queries(encoded_query, query_mask, phrase_action_branch=phrase_action_branch)  # (N, D) * 1

        return video_query, encoded_query

    def encode_context(self, frame_video_feat, video_mask=None, phrase_action_branch=False):

        if phrase_action_branch:
            encoded_frame_feat = self.encode_input(frame_video_feat, video_mask, self.frame_input_proj_action,
                                                self.frame_encoder_action, self.frame_pos_embed_action
                                                ) 
        else:
            encoded_frame_feat = self.encode_input(frame_video_feat, video_mask, self.frame_input_proj,
                                                self.frame_encoder, self.frame_pos_embed
                                                ) #
        return encoded_frame_feat

               

    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer=None):
        """
        Args:
            feat: (N, L, D_input), torch.float32
            mask: (N, L), torch.float32, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            pos_embed_layer: positional embedding layer
        """
        feat = input_proj_layer(feat)
        if pos_embed_layer is not None:
            feat = pos_embed_layer(feat)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor
        return encoder_layer(feat, mask)  # (N, L, D_hidden)

    def get_modularized_queries(self, encoded_query, query_mask, phrase_action_branch=False):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        if self.config.learnable_text_prompt > 0:
            if phrase_action_branch:
                prompt_class = self.prompt_class_action
                prompt_text_attn = self.prompt_text_attn_action
            else:
                prompt_class = self.prompt_class
                prompt_text_attn = self.prompt_text_attn
            unified_text_prompt = prompt_class.encode_prompt(batch_size=encoded_query.shape[0], device=encoded_query.device)
            prefix_prompt = unified_text_prompt[:, :unified_text_prompt.shape[1]//2, :]  # bs*4*dim
            suffix_prompt = unified_text_prompt[:, unified_text_prompt.shape[1]//2:, :]  # bs*4*dim
            encoded_query = torch.cat((prefix_prompt, encoded_query, suffix_prompt), dim=1)  # bs*(20+4+4)*dim
            query_mask = torch.cat((torch.ones(encoded_query.shape[0], unified_text_prompt.shape[1]//2, dtype=torch.bool, device=encoded_query.device), query_mask, 
                                   torch.ones(encoded_query.shape[0], unified_text_prompt.shape[1]//2, dtype=torch.bool, device=encoded_query.device)), dim=1)  # bs*(20+4+4)
            encoded_query,_ = prompt_text_attn(encoded_query, encoded_query,query_mask, query_mask)
        if phrase_action_branch:  
            modular_attention_scores = self.modular_vector_mapping_phrase(encoded_query)  # (N, L, 2 or 1)
        else:
            modular_attention_scores = self.modular_vector_mapping(encoded_query)  # (N, L, 2 or 1)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)  # (N, 2 or 1, D)
        return modular_queries.squeeze()

    @staticmethod
    def get_clip_scale_scores(modularied_query, context_feat):
        modularied_query = F.normalize(modularied_query, dim=-1)
        context_feat = F.normalize(context_feat, dim=-1)

        clip_level_query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)
        query_context_scores, indices = torch.max(clip_level_query_context_scores,
                                                  dim=1)  # (N, N) diagonal positions are positive pairs
        
        return query_context_scores, indices, clip_level_query_context_scores

    @staticmethod
    def get_unnormalized_clip_scale_scores(modularied_query, context_feat):

        query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)

        query_context_scores, _ = torch.max(query_context_scores, dim=1)

        return query_context_scores


    def text_guided_attention(self, video_query, frame_feat, feat_mask, query_labels, phrase_action_branch=False, neg_video_query=None, neg_valid_mask=None):


        query = video_query.unsqueeze(-1)

        if self.config.best_frame_loss:
            if phrase_action_branch: 
                mapping_linear = self.mapping_linear_phrase_action[0]
            else:
                mapping_linear = self.mapping_linear[0]
            encoded_frame = mapping_linear(frame_feat)
            # encoded_frame = ori_key
            normalized_encoded_frame = F.normalize(encoded_frame, dim=-1)

            normalized_query = F.normalize(query.squeeze(), dim=-1)
            best_frame_scores = torch.matmul(normalized_encoded_frame, normalized_query.t())#.permute(2, 1, 0)
            best_frame_scores = best_frame_scores.masked_fill(feat_mask.unsqueeze(-1).eq(0), -1e9)#.unsqueeze(1)
            best_frame_scores, best_frame_indices = torch.max(best_frame_scores,dim=1)  # (N, N) diagonal positions are positive pairs
            
            if neg_video_query is not None:
                
                neg_video_query = F.normalize(neg_video_query, dim=-1) #bs * 5 * 512
                bs = torch.arange(normalized_encoded_frame.shape[0])
                bs_text = torch.arange(video_query.shape[0])
                target = torch.zeros(torch.sum(neg_valid_mask), dtype=int).to(query.device)    
                if self.config.max_type =='max_pos':
                    text_best_seg_index = best_frame_indices[query_labels, bs_text] #取出每一个text所对应的视频里面 匹配分数最高的片段
                    expand_normalized_encoded_frame = normalized_encoded_frame[query_labels] #将video进行拓展
                    
                    best_match_seg = expand_normalized_encoded_frame[bs_text, text_best_seg_index].unsqueeze(1)
                    
                    neg_v2t_scores = torch.bmm(best_match_seg, neg_video_query.transpose(1, 2)).squeeze(1) #bs_text 5

                    pos_v2t_scores = best_frame_scores[query_labels, bs_text] #bs_text 1
                    total_v2t_scores = torch.cat([pos_v2t_scores.unsqueeze(1), neg_v2t_scores], dim=1)
                    # print(f"pos_score : {total_v2t_scores[neg_valid_mask][:,0].mean()} neg_score:{total_v2t_scores[neg_valid_mask][:,1:].mean()}")
                    # print(f"随机抽取score : {total_v2t_scores[neg_valid_mask][:3]}")
                    total_v2t_scores = total_v2t_scores[neg_valid_mask] * (self.temp2 if phrase_action_branch else self.temp)
                    
                    neg_query_loss = F.cross_entropy(total_v2t_scores, target, label_smoothing=self.config.neg_loss_label_smoothing) * self.config.neg_query_loss_weight
                    
                elif self.config.max_type == 'max_each':

                    cat_query = torch.cat([normalized_query.unsqueeze(1), neg_video_query], dim=1) #bs n+1 dim
                    cat_query = rearrange(cat_query, 'bs pos_neg dim -> (bs pos_neg) dim')
                    max_frame_scores_posneg = torch.matmul(normalized_encoded_frame, cat_query.t())#.permute(2, 1, 0)
                    max_frame_scores_posneg = rearrange(max_frame_scores_posneg, 'bs_v len_v (bs_t pos_neg)->bs_v len_v bs_t pos_neg', bs_t=query.shape[0])
                    max_frame_scores_posneg = max_frame_scores_posneg.masked_fill(feat_mask.unsqueeze(-1).unsqueeze(-1).eq(0), -1e9)#.unsqueeze(1)
                    best_segment_scores_posneg, best_segment_indices_posneg = torch.max(max_frame_scores_posneg, dim=1) #bs bs_t pos+neg
                    best_segment_scores_posneg_matchvideo = best_segment_scores_posneg[query_labels, bs_text]
                    total_v2t_scores = best_segment_scores_posneg_matchvideo[neg_valid_mask] * (self.temp2 if phrase_action_branch else self.temp)
                    neg_query_loss = F.cross_entropy(total_v2t_scores, target, label_smoothing=self.config.neg_loss_label_smoothing) * self.config.neg_query_loss_weight
                # print(f"neg_query_loss:{neg_query_loss}")
            else:
                neg_query_loss = 0.0
                
            best_frame_scores = best_frame_scores.permute(1, 0)
            best_frame_scores_ = torch.matmul(encoded_frame, query.squeeze().t())#.permute(2, 1, 0)
            best_frame_scores_ = best_frame_scores_.masked_fill(feat_mask.unsqueeze(-1).eq(0), -1e9)#.unsqueeze(1)
            best_frame_scores_, best_frame_indices_ = torch.max(best_frame_scores_,
                                                    dim=1)  # (N, N) diagonal positions are positive pairs
            best_frame_scores_ = best_frame_scores_.permute(1, 0)

        else:
            best_frame_scores, best_frame_scores_ = None, None
        
        return best_frame_scores, best_frame_scores_, neg_query_loss
    
    def text_guided_attention_in_inference(self, video_query, frame_feat, feat_mask, phrase_action_branch=False):
        start_time = time.time()
        key_linear = self.mapping_linear_phrase_action[0] if phrase_action_branch else self.mapping_linear[0]
        key = key_linear(frame_feat)
        normalized_encoded_frame = F.normalize(key, dim=-1)
        query = video_query.unsqueeze(0).repeat(key.shape[0], 1, 1)

        if self.config.best_frame_loss:
            
            # encoded_frame = key
            normalized_query = F.normalize(query.squeeze(), dim=-1)
            end_time = time.time()
            if len(normalized_query.shape) == 1:
                normalized_query = normalized_query.unsqueeze(0).unsqueeze(0)

            best_frame_scores = torch.bmm(normalized_encoded_frame, normalized_query.transpose(1,2))
            end_time = time.time()
            best_frame_scores = best_frame_scores.masked_fill(feat_mask.unsqueeze(-1).eq(0), -1e9)#.unsqueeze(1)
            end_time = time.time()
            best_frame_scores, best_frame_indices = torch.max(best_frame_scores,
                                                    dim=1)  # (N, N) diagonal positions are positive pairs
            end_time = time.time()
            best_frame_scores = best_frame_scores.permute(1, 0)
            end_time = time.time()
            best_frame_scores_ = None
        else:
            best_frame_scores, best_frame_scores_ = None, None
        return best_frame_scores, best_frame_scores_


    def get_pred_from_raw_query(self, query_feat, query_mask, query_labels=None,
                                video_feat=None,
                                video_feat_mask=None,
                                return_query_feats=False,
                                phrase_action_branch=False,
                                zeroshot_match=False,
                                neg_query_feat=None,
                                neg_query_mask=None
                                ):

        #query feat q_len 10 512    video_feat  v_len 64 512 -> (q_len*10  v_len*64) -> (q_len 10 v_len 64) -> q_len v_len 64 -> q_len v_len
        if zeroshot_match:
            total_num_q, num_object, dim = query_feat.shape
            total_num_v, video_len, dim = video_feat.shape
            query_feat = query_feat.reshape(-1, 512)
            video_feat = video_feat.reshape(-1, 512)
            similarity_before_norm = torch.matmul(query_feat, video_feat.transpose(0, 1))
            similarity_before_norm = rearrange(similarity_before_norm, 
                                                                      '(total_num_q num_object) (total_num_v video_len) -> total_num_q num_object total_num_v video_len', total_num_q=total_num_q, total_num_v=total_num_v)
            similarity_before_norm = similarity_before_norm.mean(dim=1) #对多个object求均值
            clip_scale_scores_ = similarity_before_norm.masked_fill(video_feat_mask.unsqueeze(0) == 0, float('-inf')).max(dim=-1)[0]

            query_feat_norm = F.normalize(query_feat, dim=-1)
            video_feat_norm = F.normalize(video_feat, dim=-1)
            similarity_after_norm = torch.matmul(query_feat_norm, video_feat_norm.transpose(0, 1))
            similarity_after_norm = rearrange(similarity_after_norm, 
                                                                      '(total_num_q num_object) (total_num_v video_len) -> total_num_q num_object total_num_v video_len', total_num_q=total_num_q, total_num_v=total_num_v)
            similarity_after_norm = similarity_after_norm.mean(dim=1) #对多个object求均值
            clip_scale_scores = similarity_after_norm.masked_fill(video_feat_mask.unsqueeze(0) == 0, float('-inf')).max(dim=-1)[0] #有normalize的
            return clip_scale_scores
        start_time = time.time()
        video_query, encoded_query = self.encode_query(query_feat, query_mask, phrase_action_branch=phrase_action_branch)
        
        if return_query_feats:
            ok = False
            if (self.config.neg_query_loss_branch=='i3d' and not phrase_action_branch) or \
            (self.config.neg_query_loss_branch=='clip' and phrase_action_branch) or\
            self.config.neg_query_loss_branch=='both':
                ok = True
            
            if ok and self.config.neg_query_loss in ['action','object', 'both']: #and not phrase_action_branch
                #i3d action   clip  object:
                if self.config.neg_query_loss =='both':#both的话就有拆分，不是both就没拆分
                    if not phrase_action_branch: #phrase_action_branch 代表的是clip。 not phrase_action_branch代表的是i3d
                    # neg_query_feat  的顺序是先action  后object，要取action就是:neg_action_num,object就是neg_action_num：
                        print(f'i3d branch 使用的是action')
                        neg_query_mask = neg_query_mask[:, :self.config.neg_action_num, :]
                        neg_query_feat = neg_query_feat[:, :self.config.neg_action_num, :, :]
                    else: #clip使用的是action
                        print(f'clip branch 使用的是object')
                        neg_query_mask = neg_query_mask[:, self.config.neg_action_num:, :]
                        neg_query_feat = neg_query_feat[:, self.config.neg_action_num:, :, :]
                neg_valid_mask = torch.sum(torch.sum(neg_query_mask, dim=-1), dim=-1) > neg_query_mask.shape[1] * 3 #start hello end  #因为无效的时候长度为2
                print(f"{torch.sum(neg_valid_mask==0)}|{torch.sum(neg_valid_mask==1)}")
                neg_query_feat = rearrange(neg_query_feat, 'bsz num_neg max_len dim -> (bsz num_neg) max_len dim')
                neg_query_mask = rearrange(neg_query_mask, 'bsz num_neg max_len -> (bsz num_neg) max_len')
                neg_video_query, neg_encoded_query = self.encode_query(neg_query_feat, neg_query_mask, phrase_action_branch=phrase_action_branch)
                neg_video_query = rearrange(neg_video_query, '(bsz num_neg) dim -> bsz num_neg dim', bsz=query_feat.shape[0])
            else:
                neg_video_query = None
                neg_valid_mask = None
            clip_scale_scores, clip_scale_scores_, neg_query_loss = self.text_guided_attention(video_query, video_feat, video_feat_mask, query_labels, 
                                                                               phrase_action_branch=phrase_action_branch, neg_video_query=neg_video_query, neg_valid_mask=neg_valid_mask)  # (N, D) * 1
            return clip_scale_scores, clip_scale_scores_, video_query, neg_query_loss
        
        else:
            if len(video_query.shape) == 1:
                video_query = video_query.unsqueeze(0)

            clip_scale_scores, clip_scale_scores_ = self.text_guided_attention_in_inference(video_query, video_feat, video_feat_mask, phrase_action_branch=phrase_action_branch)

        return clip_scale_scores

    def get_clip_triplet_loss(self, query_context_scores, labels, loss_weight_matrix=None):
        v2t_scores = query_context_scores.t()
        t2v_scores = query_context_scores
        labels = np.array(labels)

        # cal_v2t_loss
        v2t_loss = 0
        for i in range(v2t_scores.shape[0]):
            pos_pair_scores = torch.mean(v2t_scores[i][np.where(labels == i)])
            neg_pair_scores, _ = torch.sort(v2t_scores[i][np.where(labels != i)[0]], descending=True)
            
            if self.config.use_hard_negative:
                sample_neg_pair_scores = neg_pair_scores[0]
            else:
                v2t_sample_max_idx = neg_pair_scores.shape[0]
                sample_neg_pair_scores = neg_pair_scores[
                    torch.randint(0, v2t_sample_max_idx, size=(1,)).to(v2t_scores.device)]

            v2t_loss += (self.config.margin + sample_neg_pair_scores - pos_pair_scores).clamp(min=0).sum()

        # cal_t2v_loss
        text_indices = torch.arange(t2v_scores.shape[0]).to(t2v_scores.device)
        t2v_pos_scores = t2v_scores[text_indices, labels]
        mask_score = copy.deepcopy(t2v_scores.data)
        mask_score[text_indices, labels] = 999
        _, sorted_scores_indices = torch.sort(mask_score, descending=True, dim=1)
        t2v_sample_max_idx = min(1 + self.config.hard_pool_size,
                                 t2v_scores.shape[1]) if self.config.use_hard_negative else t2v_scores.shape[1]
        sample_indices = sorted_scores_indices[
            text_indices, torch.randint(1, t2v_sample_max_idx, size=(t2v_scores.shape[0],)).to(t2v_scores.device)]

        t2v_neg_scores = t2v_scores[text_indices, sample_indices]

        t2v_loss = (self.config.margin + t2v_neg_scores - t2v_pos_scores).clamp(min=0)

        if loss_weight_matrix is not None:
            return torch.sum(t2v_loss * loss_weight_matrix) / torch.sum(loss_weight_matrix) + v2t_loss / len(v2t_scores)
        else:
            return t2v_loss.sum() / len(t2v_scores) + v2t_loss / len(v2t_scores)

    def get_frame_trip_loss(self, query_context_scores, loss_weight=None, pos_radio_loss_weight=None):
        """ ranking loss between (pos. query + pos. video) and (pos. query + neg. video) or (neg. query + pos. video)
        Args:
            query_context_scores: (N, N), cosine similarity [-1, 1],
                Each row contains the scores between the query to each of the videos inside the batch.
        """

        bsz = len(query_context_scores)

        diagonal_indices = torch.arange(bsz).to(query_context_scores.device)
        pos_scores = query_context_scores[diagonal_indices, diagonal_indices]  # (N, )
        query_context_scores_masked = copy.deepcopy(query_context_scores.data)
        # impossibly large for cosine similarity, the copy is created as modifying the original will cause error
        query_context_scores_masked[diagonal_indices, diagonal_indices] = 999
        pos_query_neg_context_scores, same_video_weight1  = self.get_neg_scores(query_context_scores, query_context_scores_masked, loss_weight)
        neg_query_pos_context_scores, same_video_weight2  = self.get_neg_scores(query_context_scores.transpose(0, 1),
                                                           query_context_scores_masked.transpose(0, 1), loss_weight)
        loss_neg_ctx = self.get_ranking_loss(pos_scores, pos_query_neg_context_scores, same_video_weight1, pos_radio_loss_weight)
        loss_neg_q = self.get_ranking_loss(pos_scores, neg_query_pos_context_scores, same_video_weight2, pos_radio_loss_weight)
        return loss_neg_ctx + loss_neg_q

    def get_neg_scores(self, scores, scores_masked, loss_weight=None):
        """
        scores: (N, N), cosine similarity [-1, 1],
            Each row are scores: query --> all videos. Transposed version: video --> all queries.
        scores_masked: (N, N) the same as scores, except that the diagonal (positive) positions
            are masked with a large value.
        """

        bsz = len(scores)
        batch_indices = torch.arange(bsz).to(scores.device)

        _, sorted_scores_indices = torch.sort(scores_masked, descending=True, dim=1)

        sample_min_idx = 1  # skip the masked positive

        sample_max_idx = min(sample_min_idx + self.config.hard_pool_size, bsz) if self.config.use_hard_negative else bsz

        # sample_max_idx = 2

        # (N, )
        sampled_neg_score_indices = sorted_scores_indices[batch_indices, torch.randint(sample_min_idx, sample_max_idx,
                                                                                       size=(bsz,)).to(scores.device)]

        sampled_neg_scores = scores[batch_indices, sampled_neg_score_indices]  # (N, )
        if loss_weight is not None:
            same_video_weight = loss_weight[batch_indices, sampled_neg_score_indices]
        else:
            same_video_weight = None
        return sampled_neg_scores, same_video_weight

    def get_ranking_loss(self, pos_score, neg_score, same_video_weight=None, pos_radio_loss_weight=None):
        """ Note here we encourage positive scores to be larger than negative scores.
        Args:
            pos_score: (N, ), torch.float32
            neg_score: (N, ), torch.float32
        """
        # if pos_radio_loss_weight is not None or same_video_weight is not None: 
        #     if pos_radio_loss_weight is not None:
        #         weight_final = pos_radio_loss_weight
        #         if same_video_weight is not None:
        #             weight_final = weight_final * torch.logical_not(same_video_weight) #如果不是同一个视频的，那么就有效，否则无效
        if same_video_weight is not None:
            final_weight = same_video_weight + 1 #同一个视频的，降低权重
            return (torch.clamp(self.config.margin + neg_score - pos_score, min=0) * final_weight).sum() / final_weight.sum()
        else:    
            return torch.clamp(self.config.margin + neg_score - pos_score, min=0).sum() / len(pos_score)



def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)
