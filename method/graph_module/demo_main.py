import torch
from graph_construction import construct_temporal_graph, construct_temporal_graph2 # 两种不同的构建时间图的方式
from graph_construction import construct_spatial_graph
from valid_graph_conv import spatial_conv
from valid_graph_conv import temporal_conv
from gated_embedding_module import gated_embedding_module
from fast_kmeans import batch_fast_kmedoids
import h5py
import json
from tqdm import tqdm
import numpy as np 
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

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




def get_input():
    # 读取整个数据集的处理结果，依据合理的batch size, 划分成batch, 输入main程序
    # 先默认每一帧选取3个bounding box, 根据boxes.json中的分数，选择分数最高的3个box    

    input_features_path = "/mnt/cephfs/home/shenqingwei/zhuweikun/clip_image_encoder/clip_features_64frame.h5"
    box_info_path = "/mnt/cephfs/home/shenqingwei/zhuweikun/clip_image_encoder/boxes_64frame.json"
    seq_len = 32
    num_object = 5
    dim = 512
    bsz = 128
    with open(box_info_path, 'r', encoding='utf-8') as file:
        box_info = json.load(file)
    
    data_list = [] # 存储一个视频的box feature, shape: [1, seq_len, num_object, dim] dim = 512


    r"""
    先根据box的数量, 对于每一个视频来说, 取box数量最多的32帧出来
    """
    bsz_video_feat = torch.zeros(bsz, seq_len, num_object+1, dim)
    bsz_video_boxes = torch.zeros(bsz, seq_len, num_object, 4)
    bsz_now = 0
    wrong_file_list = ['IGEU5-000269.jpg', 'K0FAG-000023.jpg', 'KDYNB-000026.jpg', 'Y98NJ-000032', 'ZOLVU-000013']  #Y98NJ-000032只有5个box 但是有7个tensor！？
    # IGEU5 = False #读不出
    # K0FAG = False #读不出
    # KDYNB = False #读不出
    # Y98NJ = False
    if 1:
        with h5py.File('/mnt/cephfs/home/chenchuanshen/ms-sl_gt/ms-sl/charades/visual_data/video_clip_feat_boxes=5.h5', 'w') as new_hf:
            with h5py.File(input_features_path, 'r') as hf:
                for vid in tqdm(hf.keys(), desc="Reading videos in h5 file..."): # 每一个video取若干帧，每一帧有N个region, 每个region的特征是512维
                    # if vid != 'Y98NJ':
                    #     continue
                    # if not Y98NJ:
                    #     continue
                    # if not Y98NJ:continue
                    if vid != 'LGS4C':
                        continue
                    else:
                        while 1:
                            print (vid)
                    
                    group = hf[vid]

                    video_feats = torch.zeros(1, seq_len, num_object + 1, dim) # 存储视频一帧中，所有bounding box的特征, [1, num_object, dim]
                    video_boxes = torch.zeros(1, seq_len, num_object, 4)
                    # 测试阶段，每个视频只取一帧, 每一帧只取3个box
                    # 选取64帧中box数量最多的48帧
                    box_num = [len(item['box']) for item in box_info[vid]]
                    indices = top_k_indices(box_num, seq_len)
                    indices = sorted(indices) # 对下标进行升序排列
                    frame_name = [box_info[vid][idx]['image_name'] for idx in indices]
                    now_index = 0
                    for idx, frame in zip(indices, frame_name):
                        # if frame == 'mnt': # 出现图片名称划分错误，跳过
                        #     print(f'Error:{frame}, 图片名称错误')
                        #     continue
                        try:
                            arr = torch.tensor(group[frame][:])
                        except KeyError:
                            # 处理 KeyError，即对象不存在的情况
                            print(frame)
                            wrong_file_list.append(frame)
                            continue
                        scores = box_info[vid][idx]['score']
                        if len(scores) < num_object:
                            # print(f'{vid}/{frame}, {len(scores)} < {num_object}')                    
                            
                            #补全 补的是整张图片的特征
                            cha = num_object - len(scores)
                            padding_row = arr[0:1, :].repeat(cha, 1)
                            arr = torch.cat([arr, padding_row], dim=0)
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
             


def main(input_features, input_bounding_box, spatial_node_features, temporal_node_features, spatial_channels, temporal_channels):
    # hyperparameters
    eta = 0.3 # 空间图中IoU的阈值
    fai = 0.5 # 中心距离与整张图片对角长度的比值的阈值
    tao = 0.3 # temporal graph中IoU的阈值
    miu = 0.4 # appearance feature相似度的阈值
    img_width = torch.tensor(224, dtype=torch.float32, device=device)
    img_height = torch.tensor(224, dtype=torch.float32, device=device)

    # ===> debug代码 <===
    # valid_s_index = torch.zeros(input_bounding_box.shape[1] * input_bounding_box.shape[2])  # valid spatial index, dim: [128, 10],[seq_len, num_object]
    # for i in range(input_bounding_box.shape[1]):
        # valid_s_index[i*input_bounding_box.shape[2]:(i+1)*input_bounding_box.shape[2]] = 1  # 这一步在做什么？其实就是全为'1'的tensor
    # ===> debug代码 <===
    
    s_graph = construct_spatial_graph(input_bounding_box, eta, fai, img_width, img_height)  # 显存占用168M  2. 186M 3. 208M 4. 228M
    t_graph = construct_temporal_graph2(input_bounding_box, input_features, tao, miu)  # dim: [batch_size, seq_len, num_object, num_object]
    # 显存占用198M 2. 200M 3. 262 4. 400M 

    sp_conv_out = spatial_conv(input_features, s_graph, spatial_node_features) # 显存占用200M  2. 206M 3. 302M 4. 478M
    t_conv_out = temporal_conv(input_features, t_graph, temporal_node_features) # 显存占用202M 2.  210M 3. 328M 5. 528M

    print("spatial conv output shape", sp_conv_out.shape)
    print("temporal conv output shape", t_conv_out.shape)
    
    batch_size, seq_len, num_object, feature_dim = sp_conv_out.shape
    sp_conv_out = sp_conv_out.view(batch_size, -1, feature_dim)
    spatial_feature = gated_embedding_module(sp_conv_out, spatial_channels)
    temporal_feature = gated_embedding_module(t_conv_out, temporal_channels)

    print("spatial feature:", spatial_feature.shape)
    print("temporal feature:", temporal_feature.shape)

    return spatial_feature, temporal_feature




if __name__ == "__main__":
    input_features, input_bounding_box = get_input()
    # 根据输入的特征bounding box构建空间图和时间图
    # 进行图卷积操作
    # 用gated embedding module进行处理
    spatial_node_features = 512 # 图卷积后，时间图中每个结点的特征维度
    temporal_node_features = 512 # 图卷积后，空间图中每个结点的特征维度
    spatial_channels = 384 # 最终用于表达视频空间关系的特征维度
    temporal_channels = 384 # 最终用于表达视频时间关系的特征维度, 以上4个参数可以根据实际情况调整

    input_features = input_features.to(device)
    input_bounding_box = input_bounding_box.to(device)
    global_feat = input_features[:,:,0]
    graph_feat = input_features[:,:,1:]
    # 主程序
    spatial_feature, temporal_feature = main(graph_feat, input_bounding_box, spatial_node_features, temporal_node_features, spatial_channels, temporal_channels)
    
    bsz, _, _ = temporal_feature.shape
    assign, medoids = batch_fast_kmedoids(temporal_feature, 5)
    bsz_range = torch.arange(bsz, device=temporal_feature.device).unsqueeze(-1)
    temporal_event = temporal_feature[bsz_range, medoids]
    
    assign, medoids = batch_fast_kmedoids(spatial_feature, 5) #.reshape(bsz, -1, spatial_channels)
    spatial_event = spatial_feature[bsz_range, medoids]
    
    a = 'finished'