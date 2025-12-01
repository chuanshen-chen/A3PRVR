import torch
import h5py
import numpy as np
import time
# video_flow_feat = h5py.File('/mnt/cephfs/home/chenchuanshen/DL-DKD/activitynet/FeatureData/new_clip_vit_32_activitynet_vid_features.hdf5', 'r')
# video_flow_feat2 = h5py.File('/mnt/cephfs/home/chenchuanshen/ms-sl_gt/ms-sl/clip_vit_32_features.hdf5', 'r')
# feat1 = video_flow_feat['v_QOlSCBRmfWY'][...]
# feat2 = video_flow_feat2['v_QOlSCBRmfWY'][...]
# print(feat1.shape)
# print(feat2.shape)
# feat1 = video_flow_feat['v_ehGHCYKzyZ8'][...]
# feat2 = video_flow_feat2['v_ehGHCYKzyZ8'][...]
# print(feat1.shape)
# print(feat2.shape)
# dataset_names = list(video_flow_feat.keys())
# print(dataset_names)

# clip_q_feat = h5py.File('/mnt/cephfs/home/chenchuanshen/DL-DKD/activitynet/TextData/clip_ViT_B_32_activitynet_query_feat.hdf5', 'r')
# roberta_q_feat = h5py.File('/mnt/cephfs/home/chenchuanshen/DL-DKD/activitynet/TextData/roberta_activitynet_query_feat.hdf5', 'r')
# feat1 = clip_q_feat['v_bXdq2zI1Ms0#enc#2'][...]
# feat2 = roberta_q_feat['v_bXdq2zI1Ms0#enc#2'][...]
# print(feat1.shape)
# print(feat2.shape)

def uniform_feature_sampling(features, max_len):
    num_clips = features.shape[0]
    if max_len is None or num_clips <= max_len:
        return features
    idxs = np.arange(0, max_len + 1, 1.0) / max_len * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1
    new_features = []
    for i in range(max_len):
        s_idx, e_idx = idxs[i], idxs[i + 1]
        if s_idx < e_idx:
            new_features.append(np.mean(features[s_idx:e_idx], axis=0))
        else:
            new_features.append(features[s_idx])
    new_features = np.asarray(new_features)
    return new_features

def optimized_uniform_feature_sampling(features, max_len):
    num_clips = features.shape[0]
    if max_len is None or num_clips <= max_len:
        return features
    idxs = np.arange(0, max_len + 1, 1.0) / max_len * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1
    # 计算采样索引
    # idxs = np.linspace(0, num_clips - 1, max_len + 1, dtype=np.int32)

    # 使用列表推导和预先计算的区间来计算平均值
    new_features = np.array([np.mean(features[start:end], axis=0) for start, end in zip(idxs[:-1], idxs[1:])])

    return new_features



# input = np.random.rand(224, 512)
# print(input.shape)
# max_len = 128
# start = time.time()
# output1 = uniform_feature_sampling(input,max_len)
# print(f"time:{time.time()-start}")
# start = time.time()
# output2 = optimized_uniform_feature_sampling(input,max_len)
# print(f"time:{time.time()-start}")
# print((output1-output2).mean())
# print((output1-output2).max())
# print((output1-output2).min())

