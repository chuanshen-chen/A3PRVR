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
    先根据box的数量, 对于每一个视频来说, 取box数量最多的32帧出来
    """
    with h5py.File(input_features_path, 'r') as hf:
        for vid in tqdm(hf.keys(), desc="Reading videos in h5 file..."): # 每一个video取若干帧，每一帧有N个region, 每个region的特征是512维
            group = hf[vid]

            frame_feats = [] # 存储视频一帧中，所有bounding box的特征, [1, num_object, dim]
            # 测试阶段，每个视频只取一帧, 每一帧只取3个box
            first_frame = next(iter(group.keys())) # 注意: 真正的程序是for循环读取每一帧, 如下
            # for frame in group.keys():
            #     arr = group[frame][:] # 读取numpy ndarray

            # if first_frame == 'mnt': # 出现图片名称划分错误，跳过
            #     continue

            arr = group[first_frame][:]
            # 接下来要选取其中3个box
            # arr中的第一个是原始图片的特征，没有bounding box, 不用于建图
            scores = box_info[vid][0]['score']
            # if len(scores) < 3: # 如果box数量小于3，跳过
            #     continue
            indices = top_k_indices(scores, 3)
            indices = [idx + 1 for idx in indices] # 因为第一个是整张图片的特征，所以从1开始计数
            # box_features = torch.tensor(arr[indices, :], dtype=torch.float32, device=device).unsqueeze(0) # 原版代码[1, 3, 512] 
                       
            box_features = torch.tensor(arr[indices, :], dtype=torch.float32, device=device) # [3, 512]   
            box_features = torch.cat([box_features, box_features], dim=0).unsqueeze(0)         
            # frame_feats.append(box_features)  # 原版代码
             
            # 测试一个视频选取48帧的显存占用
            for i in range(32):
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



       
                # idxs = top_k_indices(scores, num_object)
                # idxs = [i + i for i in idxs]
                # box_features = torch.tensor(arr[indices, :], dtype=torch.float32, device=device).unsqueeze(0) # 原版代码[1, 3, 512] 

#                 frame_feats.append(box_features)

#                 # 从box_info中获取bounding box信息
#                 box_coorinates = torch.tensor([1, 2, 3, 4], dtype=torch.float, device=device).unsqueeze(0)


#             video_features = torch.cat(frame_feats, dim=0)
#             data_list.append(video_features.unsqueeze(0))
                
#     input_features = torch.cat(data_list, dim=0).to(device)

#     # 定义输入数据
#     batch_size, seq_len, num_object, _ = input_features.shape
#     # 每一个bounding box用4个数字表示，分别是左上角和右下角的x, y坐标

#     # 构造输入数据
#     input_bounding_box = torch.rand(batch_size, seq_len, num_object, 4).to(device)
#     return input_features, input_bounding_box






# 用于调试的程序
def debug(input_features, input_bounding_box, spatial_node_features, temporal_node_features, spatial_channels, temporal_channels): 
    # s_graph = construct_spatial_graph(input_bounding_box, eta, fai, img_width, img_height)  # dim: [batch_size, seq_len, num_object, num_object] [128, 48, 5, 5]    
    # 修改代码让s_graph中边的数量大约为结点数的一半
    # 创建形状为 (128, 48, 5, 5) 的随机邻接矩阵
    batch_size, seq_len, num_object, _ = input_features.shape
    adj_matrix = torch.zeros(batch_size, seq_len, num_object, num_object, dtype=torch.bool, device=device)

    # 每个邻接矩阵中生成3条无向边
    num_edges = num_object // 2
    for _ in range(num_edges):
        # 随机选择两个不同的结点作为边的起点和终点
        start_node = torch.randint(0, num_object, (batch_size, seq_len))
        end_node = torch.randint(0, num_object, (batch_size, seq_len))

        # 将边的两个方向都设置为1，表示无向边
        adj_matrix[:, :, start_node, end_node] = 1
        adj_matrix[:, :, end_node, start_node] = 1
    
    diagonal = torch.eye(num_object, dtype=torch.bool, device=device) # 添加自反边
    adj_matrix = adj_matrix | diagonal
    s_graph = adj_matrix

    # t_graph = construct_temporal_graph2(input_bounding_box, input_features, tao, miu)  # dim: [batch_size, seq_len * num_object, seq_len * num_object] [128, 288, 288]
    # 显存占用198M 2. 200M 3. 262 4. 400M 
    product = seq_len * num_object
    result = torch.zeros(batch_size, product, product, dtype=torch.bool, device=device)
    small_graph = torch.tensor([[0, 0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0],
                                [0, 1, 0, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0]], dtype=torch.bool, device=device)
    mask = torch.zeros(result.shape[1], result.shape[1], dtype=torch.bool, device=device)
    for start in range(0, result.shape[1] - num_object, num_object):
        mask[start: start + num_object, start + num_object: start + num_object * 2] = small_graph
    mask = torch.logical_or(mask, mask.t())
    result = torch.logical_or(result, mask)
    t_graph = result

    sp_conv_out = spatial_conv(input_features, s_graph, spatial_node_features) # 显存占用200M  2. 206M 3. 302M 4. 478M 5. 516M 6. 592M
    t_conv_out = temporal_conv(input_features, t_graph, temporal_node_features) # 显存占用202M 2.  210M 3. 328M 4. 528M 5. 752M 6. 918M

    print("spatial conv output shape", sp_conv_out.shape)
    print("temporal conv output shape", t_conv_out.shape)

    
    batch_size, seq_len, num_object, feature_dim = sp_conv_out.shape
    sp_conv_out = sp_conv_out.view(batch_size, -1, feature_dim)
    spatial_feature = gated_embedding_module(sp_conv_out, spatial_channels) # 显存占用 5. 1022M 6. 1188M
    temporal_feature = gated_embedding_module(t_conv_out, temporal_channels) # 显存占用 5. 1236M 6. 1404M

    print("spatial feature:", spatial_feature.shape)
    print("temporal feature:", temporal_feature.shape)

    return spatial_feature, temporal_feature




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
