import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

import h5py
import json
from tqdm import tqdm

def generate_input():
    # 定义输入数据
    batch_size = 8
    seq_len = 64 # 时间上的序列长度
    num_object = 10 # bounding box的数量
    # 每一个bounding box用4个数字表示，分别是左上角和右下角的x, y坐标
    feature_dim = 512

    # 构造输入数据
    input_features = torch.rand(batch_size, seq_len, num_object, feature_dim).to(device)
    input_bounding_box = torch.rand(batch_size, seq_len, num_object, 4).to(device)
    # input_bounding_box = torch.zeros(batch_size, seq_len, num_object, 4).to(device)  # 减少空间图中的边数量
    # input_bounding_box = torch.ones(batch_size, seq_len, num_object, 4).to(device)  # peak load

    return input_features, input_bounding_box

# 求一个数组中topK元素的下标
def top_k_indices(lst, k: int):
    # 使用 sorted() 对列表进行排序，并获取前三个元素的下标
    indices = sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)[:k]
    return indices

# 获取视频帧中最少和最多的物体数量
def get_max_min_frame_num_object():
    input_features_path = "/mnt/cephfs/home/shenqingwei/zhuweikun/clip_image_encoder/image_features.h5"
    frame_max_num_object = 25
    frame_min_num_object = 1
       
    one_box = []
    exception_path = "/mnt/cephfs/home/shenqingwei/models/AME-Net/one_box.txt"

    with h5py.File(input_features_path, 'r') as hf:
        for vid in tqdm(hf.keys(), desc="Reading videos in h5 file..."): # 每一个video取若干帧，每一帧有N个region, 每个region的特征是512维
            group = hf[vid]  

            # 统计frame内box数量
            for frame in group.keys():
                # arr = group[frame][:] # numpy ndarray, dtype=float16
                if frame == 'mnt':
                    continue
                temp = group[frame]
                num_obj = temp.shape[0] - 1 # 减去原始图片的特征
                if num_obj == 1:
                    one_box.append(f"{vid}/{frame}\n")
            
                if num_obj > frame_max_num_object:
                    frame_max_num_object = num_obj
                if num_obj < frame_min_num_object:
                    frame_min_num_object = num_obj     

        with open(exception_path, 'w', encoding='utf-8') as file:
            file.writelines(one_box)

    print("帧内最大box数量:", frame_max_num_object)
    print("帧内最小box数量:", frame_min_num_object)


def debug_get_input():
    # 读取整个数据集的处理结果，依据合理的batch size, 划分成batch, 输入main程序
    # 统计每一帧中bounding box的最大，最小数量
    # 先默认每一帧选取3个bounding box, 根据boxes.json中的分数，选择分数最高的3个box    

    input_features_path = "/mnt/cephfs/home/shenqingwei/zhuweikun/clip_image_encoder/image_features.h5"    
    box_info_path = "/mnt/cephfs/home/shenqingwei/zhuweikun/clip_image_encoder/boxes.json"

    with open(box_info_path, 'r', encoding='utf-8') as file:
        box_info = json.load(file)
    
    data_list = [] # 存储一个视频的box feature, shape: [1, seq_len, num_object, dim] dim = 512 

    cnt = 0
    r"""
    先根据box的数量, 对于每一个视频来说, 取box数量最多的48帧出来
    """
    with h5py.File(input_features_path, 'r') as hf:
        for vid in tqdm(hf.keys(), desc="Reading videos in h5 file..."): # 每一个video取若干帧，每一帧有N个region, 每个region的特征是512维
            group = hf[vid]

            frame_feats = [] # 存储视频一帧中，所有bounding box的特征, [1, num_object, dim]
            # 测试阶段，每个视频只取一帧, 每一帧只取3个box
            first_frame = next(iter(group.keys())) # 注意: 真正的程序是for循环读取每一帧, 如下
            # for frame in group.keys():
            #     arr = group[frame][:] # 读取numpy ndarray

            if first_frame == 'mnt': # 出现图片名称划分错误，跳过
                continue

            arr = group[first_frame][:]
            # 接下来要选取其中3个box
            # arr中的第一个是原始图片的特征，没有bounding box, 不用于建图
            scores = box_info[vid][0]['score']
            if len(scores) < 3: # 如果box数量小于3，跳过
                continue
            indices = top_k_indices(scores, 3)
            indices = [idx + 1 for idx in indices] # 因为第一个是整张图片的特征，所以从1开始计数
            # box_features = torch.tensor(arr[indices, :], dtype=torch.float32, device=device).unsqueeze(0) # 原版代码[1, 3, 512] 
                       
            box_features = torch.tensor(arr[indices, :], dtype=torch.float32, device=device) # [3, 512]   
            box_features = torch.cat([box_features, box_features], dim=0).unsqueeze(0)         
            # frame_feats.append(box_features)  # 原版代码
             
            # 测试一个视频选取48帧的显存占用
            for i in range(48):
                frame_feats.append(box_features)   

            video_features = torch.cat(frame_feats, dim=0)
            data_list.append(video_features.unsqueeze(0))
            
            cnt += 1
            if(cnt >= 8): # 先用64个video, 每个video选取一帧来测试
                break

    
    # debug
    input_features = torch.cat(data_list, dim=0).to(device) # 
    debug_input_features = []
    for i in range(16):
        debug_input_features.append(input_features)
    
    debug_input_features = torch.cat(debug_input_features, dim=0)
    input_features = debug_input_features

    # 定义输入数据
    batch_size, seq_len, num_object, _ = input_features.shape
    # 每一个bounding box用4个数字表示，分别是左上角和右下角的x, y坐标

    # 构造输入数据
    input_bounding_box = torch.rand(batch_size, seq_len, num_object, 4).to(device)
    return input_features, input_bounding_box


def get_input():
    # 读取整个数据集的处理结果，依据合理的batch size, 划分成batch, 输入main程序
    # 先默认每一帧选取3个bounding box, 根据boxes.json中的分数，选择分数最高的3个box    

    input_features_path = "/mnt/cephfs/home/shenqingwei/zhuweikun/clip_image_encoder/clip_features_64frame.h5"    
    box_info_path = "/mnt/cephfs/home/shenqingwei/zhuweikun/clip_image_encoder/boxes_64frame.json"
    seq_len = 48
    num_object = 3

    with open(box_info_path, 'r', encoding='utf-8') as file:
        box_info = json.load(file)
    
    data_list = [] # 存储一个视频的box feature, shape: [1, seq_len, num_object, dim] dim = 512


    r"""
    先根据box的数量, 对于每一个视频来说, 取box数量最多的48帧出来
    """
    with h5py.File(input_features_path, 'r') as hf:
        for vid in tqdm(hf.keys(), desc="Reading videos in h5 file..."): # 每一个video取若干帧，每一帧有N个region, 每个region的特征是512维
            group = hf[vid]

            frame_feats = [] # 存储视频一帧中，所有bounding box的特征, [1, num_object, dim]
            frame_boxs = []
            # 测试阶段，每个视频只取一帧, 每一帧只取3个box
            # 选取64帧中box数量最多的48帧
            box_num = [len(item['box']) for item in box_info[vid]]
            indices = top_k_indices(box_num, seq_len)
            indices = sorted(indices) # 对下标进行升序排列
            frame_name = [box_info[vid][idx]['image_name'] for idx in indices]
            for idx, frame in zip(indices, frame_name):
                if frame == 'mnt': # 出现图片名称划分错误，跳过
                    print(f'Error:{frame}, 图片名称错误')
                    continue

                arr = group[frame][:]
                scores = box_info[vid][idx]['score']
                if len(scores) < num_object:
                    print(f'{vid}/{frame}, {len(scores)} < {num_object}')                    
                    continue
                idxs = top_k_indices(scores, num_object)
                idxs = [i + i for i in idxs]
                box_features = torch.tensor(arr[indices, :], dtype=torch.float32, device=device).unsqueeze(0) # 原版代码[1, 3, 512] 

                frame_feats.append(box_features)

                # 从box_info中获取bounding box信息
                box_coorinates = torch.tensor([1, 2, 3, 4], dtype=torch.float, device=device).unsqueeze(0)


            video_features = torch.cat(frame_feats, dim=0)
            data_list.append(video_features.unsqueeze(0))
                
    input_features = torch.cat(data_list, dim=0).to(device)

    # 定义输入数据
    batch_size, seq_len, num_object, _ = input_features.shape
    # 每一个bounding box用4个数字表示，分别是左上角和右下角的x, y坐标

    # 构造输入数据
    input_bounding_box = torch.rand(batch_size, seq_len, num_object, 4).to(device)
    return input_features, input_bounding_box
