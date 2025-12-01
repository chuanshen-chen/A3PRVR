import numpy as np
import logging
import torch.backends.cudnn as cudnn
import os
import pickle
from method.model import MS_SL_Net
from torch.utils.data import DataLoader
from method.data_provider import Dataset4MS_SL,VisDataSet4MS_SL,  \
    TxtDataSet4MS_SL, read_video_ids, collate_frame_val, collate_text_val
from tqdm import tqdm
import torch
from utils.basic_utils import BigFile
from method.config import TestOptions
from einops import rearrange
import torch.nn.functional as F
from graph_module.fast_kmeans import batch_fast_kmedoids
import time
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def ap_score(sorted_labels):
    nr_relevant = len([x for x in sorted_labels if x > 0])
    if nr_relevant == 0:
        return 0.0

    length = len(sorted_labels)
    ap = 0.0
    rel = 0

    for i in range(length):
        lab = sorted_labels[i]
        if lab >= 1:
            rel += 1
            ap += float(rel) / (i + 1.0)
    ap /= nr_relevant
    return ap

def get_gt(video_metas, query_metas):
    v2t_gt = []
    for vid_id in video_metas:
        v2t_gt.append([])
        for i, query_id in enumerate(query_metas):
            if query_id.split('#', 1)[0] == vid_id:
                v2t_gt[-1].append(i)

    t2v_gt = {}
    for i, t_gts in enumerate(v2t_gt):
        for t_gt in t_gts:
            t2v_gt.setdefault(t_gt, [])
            t2v_gt[t_gt].append(i)

    return v2t_gt, t2v_gt

def eval_q2m(scores, q2m_gts):
    n_q, n_m = scores.shape

    gt_ranks = np.zeros((n_q,), np.int32)
    aps = np.zeros(n_q)
    for i in range(n_q):
        s = scores[i]
        sorted_idxs = np.argsort(s)
        rank = n_m + 1
        tmp_set = []
        for k in q2m_gts[i]:
            tmp = np.where(sorted_idxs == k)[0][0] + 1
            if tmp < rank:
                rank = tmp

        gt_ranks[i] = rank

    # compute metrics
    r1 = 100.0 * len(np.where(gt_ranks <= 1)[0]) / n_q
    r5 = 100.0 * len(np.where(gt_ranks <= 5)[0]) / n_q
    r10 = 100.0 * len(np.where(gt_ranks <= 10)[0]) / n_q
    r100 = 100.0 * len(np.where(gt_ranks <= 100)[0]) / n_q
    medr = np.median(gt_ranks)
    meanr = gt_ranks.mean()

    return r1, r5, r10, r100, medr, meanr, gt_ranks, 

def t2v_map(c2i, t2v_gts):
    perf_list = []
    for i in range(c2i.shape[0]):
        d_i = c2i[i, :]
        labels = [0]*len(d_i)

        x = t2v_gts[i][0]
        labels[x] = 1

        sorted_labels = [labels[x] for x in np.argsort(d_i)]

        current_score = ap_score(sorted_labels)
        perf_list.append(current_score)
    return np.mean(perf_list)


def compute_context_info(model, eval_dataset, opt):
    model.eval()
    n_total_vid = len(eval_dataset)
    context_dataloader = DataLoader(eval_dataset, collate_fn=collate_frame_val, batch_size=opt.eval_context_bsz,
                                    num_workers=opt.num_workers, shuffle=False, pin_memory=opt.pin_memory)
    
    metas = []  # list(dicts)
    frame_feat, frame_mask = [], []
    metas_action = []  # list(dicts)
    frame_feat_action, frame_mask_action = [], []

    for idx, batch in tqdm(enumerate(context_dataloader), desc="Computing query2video scores",
                           total=len(context_dataloader)):
        frame_videos, videos_mask, idxs, video_ids, frame_flow_videos, videos_flow_mask = batch
        phrase_action_branch=False
        metas.extend(video_ids)
        frame_video_feat_ = frame_videos.to(opt.device) 
        frame_mask_ = videos_mask.to(opt.device)
        
        _frame_feat = model.encode_context(frame_video_feat_, frame_mask_, phrase_action_branch=phrase_action_branch)
        
        if opt.phrase_action_branch:
            phrase_action_branch=True
            metas_action.extend(video_ids)
            frame_video_feat__action = frame_flow_videos.to(opt.device) if opt.flow_feat else frame_video_feat_
            frame_mask__action = videos_flow_mask.to(opt.device) if opt.flow_feat else frame_mask_
            if opt.clip_zero_shot:
                _frame_feat_action = frame_video_feat__action
            else:
                _frame_feat_action  = model.encode_context(frame_video_feat__action, frame_mask__action, phrase_action_branch=phrase_action_branch)
        if opt.qformer > 0:
            video_mask = model.learnable_queries_mask.to(_frame_feat.device).repeat(_frame_feat.shape[0], 1)
            
            object_queries = model.learnable_object_queries.to(_frame_feat.device).repeat(_frame_feat.shape[0], 1, 1)
            # object_queries, _ = model.qformer_selfa(object_queries, object_queries, video_mask, video_mask) #先过一层self attn
            _frame_feat, _ = model.qformer_a(object_queries, _frame_feat,
                                            video_mask, frame_mask_)
            
            action_queries = model.learnable_action_queries.to(_frame_feat_action.device).repeat(_frame_feat_action.shape[0], 1, 1)
            # action_queries,_ = model.qformer_selfb(action_queries, action_queries,video_mask, video_mask)
            _frame_feat_action, _ = model.qformer_b(action_queries, _frame_feat_action, video_mask, frame_mask__action)
            frame_mask_ = video_mask
            frame_mask__action = video_mask
        if opt.cross_branch_fusion:
            if model.config.deformable_attn:
                # cross_encoded_frame_feat = self.cross_branch_fusion_A1(encoded_frame_feat.transpose(1,2), encoded_frame_feat_action.transpose(1,2).detach())
                
                if model.config.cross_attn_q == 'i3d_q':
                    print(f"使用的是i3d_q")
                    cross_frame_feat = model.cross_branch_fusion_A1(_frame_feat.transpose(1,2), _frame_feat_action.transpose(1,2))
                    _frame_feat = cross_frame_feat.transpose(1,2)
                elif model.config.cross_attn_q == 'clip_q':
                    print(f"使用的是clip_q")
                    cross_frame_feat_action = model.cross_branch_fusion_A1(_frame_feat_action.transpose(1,2), _frame_feat.transpose(1,2))
                    _frame_feat_action = cross_frame_feat_action.transpose(1,2)
                elif model.config.cross_attn_q == 'double_q':
                    print(f"使用的是double_q")
                    cross_frame_feat = model.cross_branch_fusion_A1(_frame_feat.transpose(1,2), _frame_feat_action.transpose(1,2))
                    cross_frame_feat_action = model.cross_branch_fusion_B1(_frame_feat_action.transpose(1,2), _frame_feat.transpose(1,2))
                    _frame_feat = cross_frame_feat.transpose(1,2)
                    _frame_feat_action = cross_frame_feat_action.transpose(1,2)
            else:
                cross_frame_feat, attn = model.cross_branch_fusion_A1(_frame_feat, _frame_feat_action, frame_mask_, frame_mask__action)
                if 0:
                    print(attn.shape)
                    print(len(video_ids))
                    attn_save = attn.cpu().numpy()
                    version = '2505140100'
                    np.save(f'/share/home/chenyaofo/project/chenchuanshen/ms-sl_gt-main/q-dta_visual/attn_global_attn_{version}.npy', attn_save)
                    with open(f'/share/home/chenyaofo/project/chenchuanshen/ms-sl_gt-main/q-dta_visual/attn_global_attn_{version}.txt', 'w') as f:
                        for video_id in video_ids:
                            f.write(f"{video_id}\n")  # 写入video_id并添加换行符
                    exit()
                _frame_feat = cross_frame_feat
            # cross_frame_feat_action, _ = model.cross_branch_fusion_B1(_frame_feat_action, _frame_feat, frame_mask__action, frame_mask_)
            
            # _frame_feat_action = cross_frame_feat_action
        
        # _frame_feat = model.encode_context_2(_frame_feat, frame_mask_)
        # _frame_feat_action = model.encode_context_2(_frame_feat_action, frame_mask__action, phrase_action_branch=True)
        
        frame_feat.append(_frame_feat.cpu())
        frame_mask.append(frame_mask_.cpu())
        if opt.phrase_action_branch:
            frame_feat_action.append(_frame_feat_action.cpu())
            frame_mask_action.append(frame_mask__action.cpu())
                
        
        
    def cat_tensor(tensor_list):
        if len(tensor_list) == 0:
            return None
        else:
            seq_l = [e.shape[1] for e in tensor_list]
            b_sizes = [e.shape[0] for e in tensor_list]
            b_sizes_cumsum = np.cumsum([0] + b_sizes)
            if len(tensor_list[0].shape) == 3:
                hsz = tensor_list[0].shape[2]
                res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l), hsz)
            elif len(tensor_list[0].shape) == 2:
                res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l))
            else:
                raise ValueError("Only support 2/3 dimensional tensors")
            for i, e in enumerate(tensor_list):
                res_tensor[b_sizes_cumsum[i]:b_sizes_cumsum[i+1], :seq_l[i]] = e
            return res_tensor

    if opt.phrase_action_branch:
        second_branch = dict(
            video_metas=metas_action,  # list(dict) (N_videos)
            video_feat=cat_tensor(frame_feat_action),
            video_mask=cat_tensor(frame_mask_action)
            )
    else:
        second_branch = None
    return dict(
        video_metas=metas,  # list(dict) (N_videos)
        video_feat=cat_tensor(frame_feat),
        video_mask=cat_tensor(frame_mask)
        ), \
        second_branch
    

def compute_query2ctx_info(model, eval_dataset, opt, ctx_info, phrase_action_branch=False):
    model.eval()

    query_eval_loader = DataLoader(eval_dataset, collate_fn=collate_text_val, batch_size=opt.eval_query_bsz,
                                   num_workers=opt.num_workers, shuffle=False, pin_memory=opt.pin_memory)

    query_metas = []
    query_metas_raw = []
    clip_scale_scores = []
    frame_scale_scores = []
    score_sum = []
    gt_video_indices_list = []
    attn_weight_list = []
    query_idx_list = []
    key_clip_list = []

    total_attn_weight_list = []
    for idx, batch in tqdm(enumerate(query_eval_loader), desc="Computing q embedding", total=len(query_eval_loader)):
        query_feat, query_mask, idxs, _query_metas, zero_shot_captions_tensor, cap_ids_raw = batch
        query_metas.extend(_query_metas)
        query_metas_raw.extend(cap_ids_raw)
        query_feat = query_feat.to(opt.device)
        query_mask = query_mask.to(opt.device)
        query_idx_list.extend(idxs)
        zero_shot_captions_tensor = zero_shot_captions_tensor.to(opt.device)
            
        gt_video_indices = torch.tensor([ctx_info['video_metas'].index(_query_metas[i]) for i in range(len(_query_metas))],device='cpu')#.unsqueeze(-1)

        #attn_weight _frame_scale_scores

        video_feat = ctx_info["video_feat"]
        video_mask = ctx_info["video_mask"]
        
        _clip_scale_scores  = model.get_pred_from_raw_query(
            zero_shot_captions_tensor if opt.clip_zero_shot and phrase_action_branch else query_feat,
            query_mask, None, video_feat, video_mask, phrase_action_branch=phrase_action_branch, zeroshot_match=opt.clip_zero_shot and phrase_action_branch)
        
        _clip_scale_scores= _clip_scale_scores.cpu() # , _frame_scale_scores, attn_weight _frame_scale_scores.cpu(), attn_weight.cpu()
        _score_sum = _clip_scale_scores
        
        
        clip_scale_scores.append(_clip_scale_scores)

        score_sum.append(_score_sum)

        bsz = torch.arange(query_feat.shape[0])

        gt_video_indices_list.extend(gt_video_indices)

    clip_scale_scores = torch.cat(clip_scale_scores, dim=0).numpy().copy()

    score_sum = torch.cat(score_sum, dim=0).numpy().copy()
    
    return clip_scale_scores, score_sum, query_metas, gt_video_indices_list, query_metas_raw
       

def cal_perf(t2v_all_errors, t2v_gt):

    # video retrieval
    t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr, gt_ranks = eval_q2m(t2v_all_errors, t2v_gt)
    t2v_map_score = t2v_map(t2v_all_errors, t2v_gt)

    logging.info(" * Text to Video:")
    logging.info(" * r_1_5_10_100, medr, meanr: {}".format([round(t2v_r1, 1), round(t2v_r5, 1), round(t2v_r10, 1), round(t2v_r100, 1)]))
    logging.info(" * recall sum: {}".format(round(t2v_r1+t2v_r5+t2v_r10+t2v_r100, 1)))
    logging.info(" * mAP: {}".format(round(t2v_map_score, 4)))
    logging.info(" * "+'-'*10)

    return t2v_r1, t2v_r5, t2v_r10,t2v_r100, t2v_medr, t2v_meanr, t2v_map_score, gt_ranks

def eval_epoch(model, val_video_dataset, val_text_dataset, opt, epoch=-1):
    
    model.eval()
    logger.info(f"Epoch:[{epoch}]Computing scores")

    context_info, context_info_action = compute_context_info(model, val_video_dataset, opt)

    context_info['video_feat'] = context_info['video_feat'].to(opt.device)
    context_info['video_mask'] = context_info['video_mask'].to(opt.device)
    query_context_scores, score_sum, query_metas, gt_video_indices_list, query_metas_raw = compute_query2ctx_info(model,
                                                                        val_text_dataset,
                                                                        opt,
                                                                        context_info)
    
    
    if opt.phrase_action_branch:
        context_info_action['video_feat'] = context_info_action['video_feat'].to(opt.device)
        context_info_action['video_mask'] = context_info_action['video_mask'].to(opt.device)
        query_context_scores_action, score_sum_action, query_metas_action, gt_video_indices_list_action, query_metas_raw_action = compute_query2ctx_info(model,
                                                                                                val_text_dataset,
                                                                                                opt,
                                                                                                context_info_action, phrase_action_branch=True)

    video_metas = context_info['video_metas']
    
    
    t2v_gt = {}
    for i in range(len(gt_video_indices_list)):
        if opt.cal_only3part:
            t2v_gt[i] = [0]
        else:
            t2v_gt[i] = [gt_video_indices_list[i].item()]
    
    print('score_sum:')
    t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr, t2v_map_score, gt_ranks = cal_perf(-1 * score_sum, t2v_gt)
    
    if opt.phrase_action_branch:
        t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr, t2v_map_score, gt_ranks = cal_perf(-1 * score_sum_action, t2v_gt)
    if opt.phrase_action_branch:
        t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr, t2v_map_score, gt_ranks = cal_perf(-1 * ((1 - opt.fusion_weight) * score_sum_action + opt.fusion_weight * score_sum), t2v_gt)
    if 0:
    #####下面代码是用于2025iccv rebuttal可视化 qdta和cross attn的差距对比的
        print(len(query_metas))
        target_strings = ['1F4JZ', '2B577', '2z8G8', '7V4NJ', '9EEGQ', '21WN7']
        # 1. 先获取每个字符串的所有匹配索引（存入字典）
        string_indices = {s: [i for i, item in enumerate(query_metas) if item == s] for s in target_strings}

        # 2. 遍历字典，输出每个字符串对应的 gt_ranks 值
        for s, indices in string_indices.items():
            if not indices:  # 处理无匹配的情况（可选）
                print(f"字符串 '{s}' 无匹配索引，gt_ranks 对应值为空")
                continue
            for one_index in indices:
                # 获取 gt_ranks 中的对应值（支持 numpy 数组或列表）
                gt_values = gt_ranks[one_index]
                print(f"字符串 '{s}' 的索引：{one_index}  对应的raw cap id是:{query_metas_raw[one_index]}  对应的排名是:{gt_values}")
        import pdb
        pdb.set_trace()
    if 0:
        top1_rank = np.argsort(-1 * score_sum, axis=1)
        total_attn_weight_list = torch.cat(total_attn_weight_list)
        top1_attn_weight = total_attn_weight_list[torch.arange(top1_rank.shape[0]), top1_rank[:,0]]
        top1_video_id = [context_info['video_metas'][top1_rank[i,0]] for i in range(len(query_metas))]
        import json
        with open(os.path.join('/mnt/cephfs/home/chenchuanshen/ms-sl_gt/ms-sl/activitynet/visual_data/', 'top1_video_id.json'), 'w') as file:
            json.dump(top1_video_id, file, indent=2)
        np.save( '/mnt/cephfs/home/chenchuanshen/ms-sl_gt/ms-sl/activitynet/visual_data/top1_attn_weight', top1_attn_weight)
    currscore = 0
    currscore += (t2v_r1 + t2v_r5 + t2v_r10 + t2v_r100)

    return currscore

def setup_model(opt):
    """Load model from checkpoint and move to specified device"""
    ckpt_filepath = os.path.join(opt.ckpt_filepath)
    checkpoint = torch.load(ckpt_filepath, map_location=opt.device)
    loaded_model_cfg = checkpoint["model_cfg"]
    NAME_TO_MODELS = {'MS_SL_Net':MS_SL_Net}
    model = NAME_TO_MODELS[opt.model_name](loaded_model_cfg)
    
    model.load_state_dict(checkpoint["model"], strict=False)
    logger.info("Loaded model saved at epoch {} from checkpoint: {}".format(checkpoint["epoch"], opt.ckpt_filepath))

    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        if len(opt.device_ids) > 1:
            logger.info("Use multi GPU", opt.device_ids)
            model = torch.nn.DataParallel(model, device_ids=opt.device_ids)  # use multi GPU
    return model

def start_inference(opt=None, epoch=-1):
    logger.info("Setup config, data and model...")
    if opt is None:
        opt = TestOptions().parse()
    cudnn.benchmark = False
    cudnn.deterministic = True
    
    ########eval的时候专用
    
    # opt.cal_only3part = True #如果要计算3part，query bsz得先设置为1
    # opt.replace_key_clip = True
    # opt.global_video_loss = True
    rootpath = opt.root_path
    collection = opt.collection
    testCollection = '%stest' % collection

    cap_file = {'test': '%s.caption_old.txt' % testCollection}

    # caption
    caption_files = {x: os.path.join(rootpath, collection, 'TextData', cap_file[x])
                     for x in cap_file}

    text_feat_path = os.path.join(rootpath, collection, 'TextData', 'roberta_%s_query_feat.hdf5' % collection)
    # Load visual features
    # caption_files['test'] = '/mnt/cephfs/home/chenchuanshen/ms-sl_gt/ms-sl/activitynet/TextData/activitynettest.caption_debug.txt'
    test_video_ids_list = read_video_ids(caption_files['test'])
    
    test_vid_dataset = VisDataSet4MS_SL(opt, video_ids=test_video_ids_list)
    test_text_dataset = TxtDataSet4MS_SL(caption_files['test'], text_feat_path, opt)

    model = setup_model(opt)

    logger.info("Starting inference...")
    with torch.no_grad():
        score = eval_epoch(model, test_vid_dataset, test_text_dataset, opt, epoch=epoch)



if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    start_inference()