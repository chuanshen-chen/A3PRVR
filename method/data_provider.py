import json
import torch
import torch.utils.data as data
import numpy as np
import re
import h5py
import random
import math
import os
import time

def read_json(file_path):
    """
    读取 JSON 文件并返回字典对象。
    
    :param file_path: JSON 文件的路径
    :return: 返回字典对象
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
def is_tensor_with_single_dimension(var):
    if isinstance(var, torch.Tensor):
        return len(var.shape) == 1
    return False
def getVideoId(cap_id):
    vid_id = cap_id.split('#')[0]
    return vid_id

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    return string.strip().lower().split()

def read_video_ids(cap_file):
    video_ids_list = []
    with open(cap_file, 'r') as cap_reader:
        for line in cap_reader.readlines():
            cap_id, caption = line.strip().split(' ', 1)
            video_id = getVideoId(cap_id)
            if video_id not in video_ids_list:
                video_ids_list.append(video_id)
    return video_ids_list

def average_to_fixed_length(visual_input, map_size):
    visual_input = torch.from_numpy(visual_input)
    num_sample_clips = map_size
    num_clips = visual_input.shape[0]
    idxs = torch.arange(0, num_sample_clips + 1, 1.0) / num_sample_clips * num_clips

    idxs = torch.min(torch.round(idxs).long(), torch.tensor(num_clips - 1))

    new_visual_input = []

    for i in range(num_sample_clips):

        s_idx, e_idx = idxs[i].item(), idxs[i + 1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx], dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0).numpy()

    return new_visual_input

def uniform_feature_sampling_wrong(features, max_len):
    num_clips = features.shape[0]
    if max_len is None or num_clips <= max_len:
        return features
    idxs = np.arange(0, max_len + 1, 1.0) / max_len * num_clips
    idxs = np.round(idxs).astype(np.int32)
    idxs[idxs > num_clips - 1] = num_clips - 1
    new_features = np.array([np.mean(features[start:end], axis=0)for start, end in zip(idxs[:-1], idxs[1:])])

    return new_features



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
    
    # Generate indices once and use vectorized operations
    idxs = np.linspace(0, num_clips - 1, max_len + 1, dtype=np.int32)
    
    # Compute all slices at once
    slices = [features[s_idx:e_idx] for s_idx, e_idx in zip(idxs[:-1], idxs[1:])]
    
    # Compute means using vectorized operation
    new_features = np.array([np.mean(slice, axis=0) if slice.shape[0] > 1 else slice[0] for slice in slices])
    
    return new_features

def l2_normalize_np_array(np_array, eps=1e-5):
    """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)

def l2_normalize_tensor(tensor, eps=1e-5):
    """tensor: torch.Tensor, (*, D), where the last dim will be normalized"""
    return tensor / (torch.norm(tensor, dim=-1, keepdim=True) + eps)

def collate_train(data):
    
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True) 
    
    frame_video_features, captions, idxs, cap_ids, video_ids, frame_flow_video_features, final_neg_inputs = zip(*data) 

    #######################
    video_lengths = [len(frame) for frame in frame_video_features]
    frame_vec_len = len(frame_video_features[0][0])
    frame_videos = torch.zeros(len(frame_video_features), max(video_lengths), frame_vec_len)
    videos_mask = torch.zeros(len(frame_video_features), max(video_lengths))
    for i, frames in enumerate(frame_video_features):
        end = video_lengths[i]
        frame_videos[i, :end, :] = frames[:end, :]
        videos_mask[i, :end] = 1.0
        ###########################################
        
    if len(frame_flow_video_features[0].shape) != 1:
        video_flow_lengths = [len(frame) for frame in frame_flow_video_features]
        frame_flow_vec_len = len(frame_flow_video_features[0][0])
        frame_flow_videos = torch.zeros(len(frame_flow_video_features), max(video_flow_lengths), frame_flow_vec_len)
        videos_flow_mask = torch.zeros(len(frame_flow_video_features), max(video_flow_lengths))
        for i, frames in enumerate(frame_flow_video_features):
            end = video_flow_lengths[i]
            frame_flow_videos[i, :end, :] = frames[:end, :]
            videos_flow_mask[i, :end] = 1.0
    else:
        videos_flow_mask = torch.tensor([0.0])
        frame_flow_videos = torch.tensor([0.0])

    #captions
    feat_dim = captions[0][0].shape[-1]

    merge_captions = []
    all_lengths = []
    labels = []

    for index, caps in enumerate(captions):#caps:每一个video所对应的caption list
        labels.extend(index for i in range(len(caps)))
        all_lengths.extend(len(cap) for cap in caps)
        merge_captions.extend(cap for cap in caps) #caption list 里面每一个caption

    target = torch.zeros(len(all_lengths), max(all_lengths), feat_dim)
    words_mask = torch.zeros(len(all_lengths), max(all_lengths))

    for index, cap in enumerate(merge_captions):
        end = all_lengths[index]
        target[index, :end, :] = cap[:end, :]
        words_mask[index, :end] = 1.0
    
    cat_final_neg_inputs = []
    for i in range(len(final_neg_inputs)): #有多少个video 
        for j in range(len(final_neg_inputs[i])): #每一个video 有多少个 caption
            cat_final_neg_inputs.extend(final_neg_inputs[i][j]) #每一个caption有多少个负样本

        
    # print(len(cat_final_neg_inputs))
    return dict(frame_video_features=frame_videos,
                videos_mask=videos_mask,
                frame_video_flow_features=frame_flow_videos,
                videos_flow_mask=videos_flow_mask,
                text_feat=target,
                text_mask=words_mask,
                text_labels=labels,
                final_neg_inputs=cat_final_neg_inputs
                # text_feat_neg=target_neg,
                # text_mask_neg=words_mask_neg
                )

def collate_frame_val(data):
    frame_video_features, idxs, video_ids, frame_flow_video_features = zip(*data)
    video_lengths = [len(frame) for frame in frame_video_features]
    frame_vec_len = len(frame_video_features[0][0])
    frame_videos = torch.zeros(len(frame_video_features), max(video_lengths), frame_vec_len)
    videos_mask = torch.zeros(len(frame_video_features), max(video_lengths))
    for i, frames in enumerate(frame_video_features):
        end = video_lengths[i]
        frame_videos[i, :end, :] = frames[:end, :]
        videos_mask[i, :end] = 1.0

    if len(frame_flow_video_features[0].shape) != 1:
        video_flow_lengths = [len(frame) for frame in frame_flow_video_features]
        frame_flow_vec_len = len(frame_flow_video_features[0][0])
        frame_flow_videos = torch.zeros(len(frame_flow_video_features), max(video_flow_lengths), frame_flow_vec_len)
        videos_flow_mask = torch.zeros(len(frame_flow_video_features), max(video_flow_lengths))
        for i, frames in enumerate(frame_flow_video_features):
            end = video_flow_lengths[i]
            frame_flow_videos[i, :end, :] = frames[:end, :]
            videos_flow_mask[i, :end] = 1.0
    else:
        videos_flow_mask = torch.tensor([0.0])
        frame_flow_videos = torch.tensor([0.0]) 

    return frame_videos, videos_mask, idxs, video_ids, frame_flow_videos, videos_flow_mask

def collate_text_val(data):
    if data[0][0] is not None:
        data.sort(key=lambda x: len(x[0]), reverse=True)
    captions,idxs, cap_ids, zero_shot_captions_tensor, cap_ids_raw = zip(*data)

    if captions[0] is not None:
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        target = torch.zeros(len(captions), max(lengths), captions[0].shape[-1])
        words_mask = torch.zeros(len(captions), max(lengths))
        for i, cap in enumerate(captions):
            end = lengths[i]
            target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
    else:
        target = None
        lengths = None
        words_mask = None
    zero_shot_captions_tensor = torch.cat(zero_shot_captions_tensor, dim=0)

    return target, words_mask, idxs, cap_ids, zero_shot_captions_tensor, cap_ids_raw

class Dataset4MS_SL(data.Dataset):
    """
    Load captions and video frame features by pre-trained CNN model.
    """

    def __init__(self, cap_file, text_feat_path, opt):
        # Captions
        self.captions = {}
        self.cap_ids = []
        self.video_ids = []
        self.vid_caps = {}

        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.split(' ', 1)
                video_id = getVideoId(cap_id)
                self.captions[cap_id] = caption.strip()
                self.cap_ids.append(cap_id)
                if video_id not in self.video_ids:
                    self.video_ids.append(video_id)
                if video_id in self.vid_caps:
                    self.vid_caps[video_id].append(cap_id)
                else:
                    self.vid_caps[video_id] = []
                    self.vid_caps[video_id].append(cap_id)
        self.text_feat_path = text_feat_path
        self.map_size = opt.map_size
        self.max_ctx_len = opt.max_ctx_l
        self.max_desc_len = opt.max_desc_l

        self.length = len(self.vid_caps)
        self.config = opt
        
        if opt.dset_name =='activitynet':
            self.video_feat_path = 'dataset/activitynet_i3d/FeatureData/anet_clip_i3d_numpy.hdf5'
            
            self.video_flow_feat_path = "dataset/activitynet_clip/FeatureData/new_clip_vit_32_activitynet_vid_features.hdf5"
            self.action_list = read_json('./dataset/activitynet_i3d/TextData/anet_total_action_dict.json') 
            self.object_list = read_json('./dataset/activitynet_i3d/TextData/anet_total_object_dict.json') 
          
            self.verb_dict = read_json('./dataset/activitynet_i3d/TextData/anet_id_action_dict.json')
            self.object_dict = read_json('./dataset/activitynet_i3d/TextData/anet_id_object_dict.json')
        elif opt.dset_name =='tvr':
            
            self.video_feat_path = "dataset/tvr_i3d/FeatureData/tvr_clip_i3d_numpy.hdf5"
            
            self.action_list = read_json('./dataset/tvr_i3d/TextData/tvr_total_action_dict.json') 
            self.object_list = read_json('./dataset/tvr_i3d/TextData/tvr_total_object_dict.json') 
           
            self.verb_dict = read_json('./dataset/tvr_i3d/TextData/tvr_id_action_dict.json')
            self.object_dict = read_json('./dataset/tvr_i3d/TextData/tvr_id_object_dict.json')
        elif opt.dset_name =='charades':
            self.video_feat_path = 'dataset/charades_i3d/FeatureData/charades_clip_i3d_numpy.hdf5'
            self.action_list = read_json('./dataset/charades_i3d/TextData/charades_total_action_dict.json')
            self.object_list = read_json('./dataset/charades_i3d/TextData/charades_total_object_dict.json') 
           
            self.verb_dict = read_json('./dataset/charades_i3d/TextData/charades_id_action_dict.json')
            self.object_dict = read_json('./dataset/charades_i3d/TextData/charades_id_object_dict.json')
            
        self.video_feat_file = h5py.File(self.video_feat_path, 'r')
        self.query_feat_file = h5py.File(self.text_feat_path, 'r')
        
        self.neg_query_loss = opt.neg_query_loss
        if self.neg_query_loss == 'object':
            self.action_neg_feat = False
            self.object_neg_feat = True
        elif self.neg_query_loss == 'action':
            self.action_neg_feat = True
            self.object_neg_feat = False
        elif self.neg_query_loss == 'both': #两个都要
            self.action_neg_feat = True
            self.object_neg_feat = True
        else:
            self.action_neg_feat = False
            self.object_neg_feat = False
       
    def replace_verbs(self, sentence, verbs, action_list_tag, neg_num):
        new_sentences = []
        
        total_verb_num = len(verbs)
        
        split_result_random_corrected = neg_num // total_verb_num
        cha = neg_num % total_verb_num
        start_time = time.time()
        for key in verbs.keys():
            special = verbs[key]
            if split_result_random_corrected >= len(action_list_tag[special]):
                print(f"越界！！！{split_result_random_corrected} | {len(action_list_tag[special])}")
                print(f"{action_list_tag[special]} | {special}")
            new_verbs = random.sample(action_list_tag[special], split_result_random_corrected)
            if key in new_verbs:
                new_verbs.remove(key)
                while 1:
                    add_verb = random.sample(action_list_tag[special], 1)
                    if add_verb != key:
                        new_verbs.append(add_verb[0])
                        break
            new_sentences.extend([sentence.replace(key, new_verb) for new_verb in new_verbs])
          
        if cha > 0:
            new_verbs = random.sample(action_list_tag[special], cha)
            if key in new_verbs:
                new_verbs.remove(key)
                while 1:
                    add_verb = random.sample(action_list_tag[special], 1)
                    if add_verb != key:
                        new_verbs.append(add_verb[0])
                        break
            new_sentences.extend([sentence.replace(key, new_verb) for new_verb in new_verbs])   
        return new_sentences

    def __getitem__(self, index):
      
        video_id = self.video_ids[index]
        cap_ids = self.vid_caps[video_id]

        if self.config.dset_name == 'activitynet' or self.config.dset_name == 'charades' or self.config.dset_name == 'tvr':
            frame_video_feature = uniform_feature_sampling(self.video_feat_file[video_id]['i3d_feat'][...], self.max_ctx_len)
            frame_video_feature = torch.from_numpy(l2_normalize_np_array(frame_video_feature))

            flow_feat = uniform_feature_sampling(self.video_feat_file[video_id]['clip_feat'][...], self.max_ctx_len)
            flow_feat = torch.from_numpy(l2_normalize_np_array(flow_feat))

        cap_tensors = []
        final_neg_inputs = []
        for cap_id in cap_ids:
            cap_feat = self.query_feat_file[cap_id][...]
            
            sentence = self.captions[cap_id]
            if self.action_neg_feat or self.object_neg_feat: #要使用action
                ok = 0
                valid_neg_action, valid_neg_object = [], []
                
                if 'NNP' in self.verb_dict[cap_id].values():
                    self.verb_dict[cap_id] = {key: value for key, value in self.verb_dict[cap_id].items() if value != 'NNP'}
                if len(self.verb_dict[cap_id]) == 0 or not self.action_neg_feat: #说明该句子中没有发生替换  或者不适用action
                    neg_samples_action = ['Hello' for i in range(self.config.neg_action_num)]
                else: 
                    ok += 1
                    valid_neg_action.append(True)
                    verb_in_sentences = self.verb_dict[cap_id] #取出该id对应的句子里面的动词，以及词性 {'play':'xxx','walk':'xxx'}
                    neg_samples_action = self.replace_verbs(sentence, verb_in_sentences, self.action_list, self.config.neg_action_num)
                    
                if len(self.object_dict[cap_id]) == 0  or not self.object_neg_feat: #说明该句子中没有发生替换 或者不适用object 
                    neg_samples_object = ['Hello' for i in range(self.config.neg_object_num)]
                else: 
                    ok += 2
                    valid_neg_object.append(True)
                    object_in_sentences = self.object_dict[cap_id] #取出该id对应的句子里面的动词，以及词性 {'play':'xxx','walk':'xxx'}
                    neg_samples_object = self.replace_verbs(sentence, object_in_sentences, self.object_list, self.config.neg_object_num)

                neg_samples = neg_samples_action + neg_samples_object
               

                final_neg_inputs.append(neg_samples)
              
            else:
                final_neg_inputs.append(['Hello'])

            cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]
            cap_tensors.append(cap_tensor)
       
        return frame_video_feature, cap_tensors, index, cap_ids, video_id, flow_feat, final_neg_inputs
    
    def __len__(self):
        return self.length
    
class VisDataSet4MS_SL(data.Dataset):

    def __init__(self, opt, video_ids=None):
        
        self.video_ids = video_ids
        self.length = len(self.video_ids)
        self.map_size = opt.map_size
        self.max_ctx_len = opt.max_ctx_l
        
        self.config = opt
        if opt.dset_name =='activitynet':
            
            self.video_feat_path = 'dataset/activitynet_i3d/FeatureData/anet_clip_i3d_numpy.hdf5'
        elif opt.dset_name =='tvr':
            
            self.video_feat_path = 'dataset/tvr_i3d/FeatureData/tvr_clip_i3d_numpy.hdf5'
            
        elif opt.dset_name =='charades':
            self.video_feat_path = 'dataset/charades_i3d/FeatureData/charades_clip_i3d_numpy.hdf5'
        self.video_feat_file = h5py.File(self.video_feat_path, 'r')
         
            
    def __getitem__(self, index):
        video_id = self.video_ids[index]

        if self.config.dset_name=='activitynet' or self.config.dset_name=='charades' or self.config.dset_name=='tvr':
            frame_video_feature = uniform_feature_sampling(self.video_feat_file[video_id]['i3d_feat'][...], self.max_ctx_len)
            frame_video_feature = torch.from_numpy(l2_normalize_np_array(frame_video_feature))

            flow_feat = uniform_feature_sampling(self.video_feat_file[video_id]['clip_feat'][...], self.max_ctx_len)
            flow_feat = torch.from_numpy(l2_normalize_np_array(flow_feat))
        else:
            frame_video_feature = uniform_feature_sampling(np.load(save_path), self.max_ctx_len)
            frame_video_feature = l2_normalize_np_array(frame_video_feature)
            frame_video_feature = torch.from_numpy(frame_video_feature)

            flow_feat = uniform_feature_sampling(np.load(os.path.join(self.video_flow_feat_path, video_id+'.npy')), self.max_ctx_len)
            flow_feat = l2_normalize_np_array(flow_feat)
            flow_feat = torch.from_numpy(flow_feat)
            
        return frame_video_feature, index, video_id, flow_feat

    def __len__(self):
        return self.length

class TxtDataSet4MS_SL(data.Dataset):
    """
    Load captions
    """

    def __init__(self, cap_file, text_feat_path, opt):
        # Captions
        self.captions = {}
        self.cap_ids = []
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
        self.text_feat_path = text_feat_path
        self.max_desc_len = opt.max_desc_l
        self.length = len(self.cap_ids)

        self.map_size = opt.map_size
    
        self.text_feat = h5py.File(self.text_feat_path, 'r')
    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        cap_feat = self.text_feat[cap_id][...].squeeze()
        cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]

        return cap_tensor, index, cap_id.split('#')[0], torch.zeros(1,1,10,512), cap_id #最后面是没用的东西

    def __len__(self):
        return self.length

if __name__ == '__main__':
    pass




