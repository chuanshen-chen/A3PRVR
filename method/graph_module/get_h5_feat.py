import torch
import h5py
import json
from tqdm import tqdm
def top_k_indices(lst, k: int):
    # 使用 sorted() 对列表进行排序，并获取前三个元素的下标
    indices = sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)[:k]
    return indices
def get_input():
    # 读取整个数据集的处理结果，依据合理的batch size, 划分成batch, 输入main程序
    # 先默认每一帧选取3个bounding box, 根据boxes.json中的分数，选择分数最高的3个box    
    
    input_features_path = "/mnt/cephfs/home/shenqingwei/zhuweikun/Anet_processing/anet_clip_feature/anet_clip_features_128frame.h5"
    box_info_path = "/mnt/cephfs/home/shenqingwei/zhuweikun/Anet_processing/anet_boxes_128frame.json"
    # input_features_path = "/mnt/cephfs/home/shenqingwei/zhuweikun/clip_image_encoder/clip_features_64frame_9848.h5"
    # box_info_path = "/mnt/cephfs/home/shenqingwei/zhuweikun/clip_image_encoder/boxes_64frame_9848.json"
    seq_len = 96
    num_object = 5
    dim = 512
    bsz = 128
    # with h5py.File(input_features_path, 'r') as hf:
    #     a = 1
    with open(box_info_path, 'r', encoding='utf-8') as file:
        box_info = json.load(file)
    
    data_list = [] # 存储一个视频的box feature, shape: [1, seq_len, num_object, dim] dim = 512


    r"""
    先根据box的数量, 对于每一个视频来说, 取box数量最多的32帧出来
    """
    bsz_video_feat = torch.zeros(bsz, seq_len, num_object+1, dim)
    bsz_video_boxes = torch.zeros(bsz, seq_len, num_object, 4)
    bsz_now = 0
    #v_-02DygXbn6w_80.jpg v_0uOMJSUza68.mkv_63.jpg
    if 1:
        with h5py.File('/mnt/cephfs/home/chenchuanshen/ms-sl_gt/ms-sl/activitynet/visual_data/video_clip_feat_boxes=5.h5', 'w') as new_hf:
            # print(new_hf.keys())
            # for vid in tqdm(new_hf.keys(), desc="Reading videos in h5 file..."):
            #     print(vid)
            # return None,None
            with h5py.File(input_features_path, 'r') as hf:
                for vid in tqdm(hf.keys(), desc="Reading videos in h5 file..."): # 每一个video取若干帧，每一帧有N个region, 每个region的特征是512维
                    # if vid != 'Y98NJ':
                    #     continue
                    # if not Y98NJ:
                    #     continue
                    # if not Y98NJ:continue
                    if vid != 'v_0uOMJSUza68':
                        continue
                    # else:           
                    print (vid)
                    group = hf[vid]

                    video_feats = torch.zeros(1, seq_len, num_object + 1, dim) # 存储视频一帧中，所有bounding box的特征, [1, num_object, dim]
                    video_boxes = torch.zeros(1, seq_len, num_object, 4)

                    box_info[vid] = sorted(box_info[vid], key=lambda x: len(x['box']), reverse=True) 
                    box_info[vid] = box_info[vid][:seq_len]  #取了前topk个出来
                    
                    box_info[vid] = sorted(box_info[vid], key=lambda x: int(x['image_name'].split('_')[-1].split('.')[0])) #按照先后顺序排列
                    # box_num = [len(item['box']) for item in box_info[vid]]
                    # indices = top_k_indices(box_num, seq_len)
                    # indices = sorted(indices) # 对下标进行升序排列
                    # frame_name = [box_info[vid][idx]['image_name'] for idx in indices]
                    now_index = 0
                    for idx in range(len(box_info[vid])):
                        # if '.mkv' in frame:
                        #     # 使用replace方法删除子串
                        #     frame = frame.replace('.mkv', '')
                        if box_info[vid][idx]['image_name'] not in group.keys():
                            before, before2, _ = box_info[vid][idx]['image_name'].split('_')
                            before = before + "_" + before2
                            after = '.jpg'
                            num = int(box_info[vid][idx]['image_name'].split('_')[-1].split('.')[0]) - 1
                            final = before + "_" + str(num) + after
                            box_info[vid][idx]['image_name'] = final
                            if final not in group.keys():
                                num += 2
                                final = before + "_" + str(num) + after
                                box_info[vid][idx]['image_name'] = final
                        arr = torch.tensor(group[box_info[vid][idx]['image_name']][:])

                        scores = box_info[vid][idx]['score']
                        if len(scores) < num_object:
                            cha = num_object - len(scores)
                            padding_row = arr[0:1, :].repeat(cha, 1)
                            arr = torch.cat([arr, padding_row], dim=0)
                            if len(scores) > 0:
                                video_boxes[:, now_index, :len(scores)] = torch.tensor(box_info[vid][idx]['box'])
                            for i in range(len(scores), num_object):
                                video_boxes[:, now_index, i] = torch.tensor([0.0, 0.0, 1.0, 1.0])
                            
                        elif len(scores) > num_object:
                            box_indices = top_k_indices(scores, num_object)
                            # box_coorinates = []
                            for i in range(len(box_indices)):
                                video_boxes[:, now_index, i] = torch.tensor(box_info[vid][idx]['box'][box_indices[i]])
                            box_indices = [box_indices[i]+1 for i in range(len(box_indices))]
                            arr = torch.cat((arr[0:1], arr[box_indices]), dim=0)
                        else:
                            # box_coorinates = box_info[vid][idx]['box']
                            video_boxes[:, now_index] = torch.tensor(box_info[vid][idx]['box'])
                        
                        
                        video_feats[:, now_index] = torch.tensor(arr)
                        now_index += 1
                        
                    video_name = new_hf.create_group(vid)
                    video_name.create_dataset('video_clip_feat_32frames', data=video_feats)
                    video_name.create_dataset('video_5boxes_32frames', data=video_boxes)
                    
                    # bsz_video_feat[bsz_now] = video_feats
                    # bsz_video_boxes[bsz_now] = video_boxes
                    # bsz_now += 1
                    # if bsz_now == bsz:
                    #     break
            
    return bsz_video_feat, bsz_video_boxes

if __name__ == "__main__":
    input_features, input_bounding_box = get_input()