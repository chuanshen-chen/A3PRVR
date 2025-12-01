import torch
import torch.nn.functional as F

from einops import rearrange
import math
eta = 0.3 # 空间图中IoU的阈值
fai = 0.5 # 中心距离与整张图片对角长度的比值的阈值

tao = 0.3 # temporal graph中IoU的阈值
miu = 0.4 # appearance feature相似度的阈值
image_width = 30  # 图片大小，暂时随机设置
image_height = 40 

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 把输入数据数据的格式从center + size转换到左上角坐标和右下角坐标
def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

def compute_iou(box):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4]. 第一组bbox中有N个框, 
      box2: (tensor) bounding boxes, sized [M,4]. 第二组bbox中有M个框
    Return:
      (tensor) iou, sized [N,M].
      求解其中两两之间对应的IOU值,并输出一个大小为[N, M]的矩阵,
      其中每一个下标ij都表示位于第一组bbox list中下标为i的bbox和第二组bbox list中下标为j的bbox的IOU
    """
    # 首先计算两个box左上角点坐标的最大值和右下角坐标的最小值，然后计算交集面积，最后把交集面积除以对应的并集面积
    
    BATCH, N, _ = box.shape
    lt = torch.max(  # 左上角的点
        box[..., :2].unsqueeze(1),
        box[..., :2].unsqueeze(2)
    )

    rb = torch.min(  # 右下角的点
        box[..., 2:].unsqueeze(1),
        box[..., 2:].unsqueeze(2)
    )

    wh = rb - lt  
    wh[wh < 0] = 0
    inter = wh[..., 0] * wh[..., 1]  # [Big Batch, N, N]

    area1 = (box[..., 2] - box[..., 0]) * (box[..., 3] - box[..., 1])  # [Big Batch, N]
    area2 = (box[..., 2] - box[..., 0]) * (box[..., 3] - box[..., 1])  # [Big Batch, N]

    area1 = area1.unsqueeze(-1)#.expand(BATCH, N, N)
    area2 = area2.unsqueeze(-2)#.expand(BATCH, N, N)
    iou = inter / (area1 + area2 - inter)
    return iou

def compute_cover(box):
    lt_diff = box[..., :2].unsqueeze(1) - box[..., :2].unsqueeze(2) # [N, M, 2]
    rb_diff = box[..., 2:].unsqueeze(2) - box[..., 2:].unsqueeze(1) # [N, M, 2]

    # 如果lt_diff中的
    lt_a_cover_b = torch.min(lt_diff, dim=-1)[0] >=0
    rb_a_cover_b = torch.min(rb_diff, dim=-1)[0] >=0     
    a_cover_b = torch.logical_and(lt_a_cover_b, rb_a_cover_b)
    
    # 类似的方法计算b_cover_a, 两者逐元素相加得到cover的结果
    lt_b_cover_a = torch.max(lt_diff, dim=-1)[0] <= 0
    rb_b_cover_a = torch.max(rb_diff, dim=-1)[0] <= 0
    b_cover_a = torch.logical_and(lt_b_cover_a, rb_b_cover_a)

    return a_cover_b | b_cover_a

def compute_ratio(box, threshold):
    r"""
    Args:
        img_width: tensor, dtype = float
        img_height: tensor, dtype = float    
    """
    # # diagonal_length = torch.sqrt(torch.square(img_width) + torch.square(img_height)).to(device)
    # center1 = box[..., :2].unsqueeze(2)
    # center2 = box[..., :2].unsqueeze(1)
    center1 = ((box[..., :2] + box[..., 2:]) / 2) .unsqueeze(2)
    center2 = ((box[..., :2] + box[..., 2:]) / 2) .unsqueeze(1)
    
    result = torch.sum((center1 - center2) ** 2, dim=-1).sqrt() / math.sqrt(2)# dim: [8192, 10, 10]
    ratio = result < threshold

    return ratio

def compute_cosine_similarity(features):
    r"""
    Args: 
        features1: [N, dim]
        features2: [M, dim]
        计算N个和M个特诊向量之间的余弦相似度, 范围在[0, 1]
    Return:
        [N, M]
        返回值的类型是Float不能够downcast to Long
    """
    simi = torch.bmm(F.normalize(features, dim=-1), F.normalize(features, dim=-1).transpose(1, 2))
    simi = (simi + 1) / 2
    return simi

"""    tensor([[[1.0000, 0.9919, 0.9867],
         [0.9919, 1.0000, 0.9993],
         [0.9867, 0.9993, 1.0000]],

        [[1.0000, 1.0000, 0.9999],
         [1.0000, 1.0000, 1.0000],
         [0.9999, 1.0000, 1.0000]]], dtype=torch.float64)"""

"""tensor([[[0.4472, 0.8944],
         [0.6000, 0.8000],
         [0.6402, 0.7682]]], dtype=torch.float64)"""

    
# 构建空间图
# def construct_spatial_graph(input_bounding_box, eta, fai, img_width, img_height):
#     r"""
#     Args:
#         img_width: float tensor
#         img_height: float tensor
#     """

#     batch_size, seq_len, num_object, _ = input_bounding_box.shape
#     input_bounding_box = input_bounding_box.view(-1, input_bounding_box.shape[2], 4) # [batch_size * seq_len, 10, 4] 
#     # reshape to (2, 640, 4)

#     # batch_sp_graph = [] # 记录batch中每一个视频的spatial graph, 每个元素的形状: [1, seq_len, num_object, num_object]
        
#     ratio_matrix = compute_ratio(input_bounding_box, img_width, img_height, fai)
                    
#     input_bounding_box = box_center_to_corner(input_bounding_box) # 转换boounding box的格式
#     iou_matrix = compute_iou(input_bounding_box) # 640 * 640, 每20行或者列代表一帧图像，
#     iou_matrix = iou_matrix > eta

#     # 判断bounding_box之间是否完全覆盖，或者完全被覆盖
#     cover_matrix = compute_cover(input_bounding_box)   

#     result = torch.eye(cover_matrix.shape[-1], dtype=cover_matrix.dtype, device=device)
#     # ===> 原代码 <===
#     result = ratio_matrix | iou_matrix | cover_matrix | result 
#     batch_spatial_graph = rearrange(result, '(bsz temporal) n m->bsz temporal n m', bsz=batch_size) # 相当于view操作
#     # sp_graph = []
#     # for start in range(0, N, num_object):  # (640, 640) ===> (32, 20, 20)
#     #     sp_graph.append(result[start:start + num_object, start: start + num_object].unsqueeze(0))  # 获取对角线位置上的矩阵
    
#     # spatial_graph = torch.cat(sp_graph)  # [seq_len, num_object, num_object] == [32, 20, 20]
#     # batch_sp_graph.append(spatial_graph.unsqueeze(0))
    
#     # # 一个batch的空间图,最终的处理结果
#     # batch_spatial_graph = torch.cat(batch_sp_graph) # [batch_size, seq_len, num_object, num_object] == [batch_size, 32, 20, 20]
    
#     return batch_spatial_graph
    
#     # 构建时间图和空间图时
#     # 计算IoU的时候, 首先对输入的bounding_box进行reshape, (batch_size, seq_len * num_object, 4)
#     # 然后，取batch中的一个样本  矩阵转置(4, seq_len * num_object), 计算后得到(batch_size, seq_len * num_object, seq_len * num_object)的关于IoU的矩阵
#     # 
#     # reshape, 获取每一对bounding box之间的IOU

def construct_spatial_graph(input_bounding_box, eta, fai):
    r"""
    Args:
        img_width: float tensor
        img_height: float tensor
    """

    batch_size, seq_len, num_object, _ = input_bounding_box.shape
    input_bounding_box = input_bounding_box.view(-1, input_bounding_box.shape[2], 4) # [batch_size * seq_len, 10, 4] 
    # reshape to (2, 640, 4)

    # batch_sp_graph = [] # 记录batch中每一个视频的spatial graph, 每个元素的形状: [1, seq_len, num_object, num_object]
        
    ratio_matrix = compute_ratio(input_bounding_box, fai)
                    
    # input_bounding_box = box_center_to_corner(input_bounding_box) # 转换boounding box的格式
    iou_matrix = compute_iou(input_bounding_box) # 640 * 640, 每20行或者列代表一帧图像，
    iou_matrix = iou_matrix > eta

    # 判断bounding_box之间是否完全覆盖，或者完全被覆盖
    cover_matrix = compute_cover(input_bounding_box)   

    result = torch.eye(cover_matrix.shape[-1], dtype=cover_matrix.dtype, device=device)
    # ===> 原代码 <===
    result = ratio_matrix | iou_matrix | cover_matrix | result 
    batch_spatial_graph = rearrange(result, '(bsz temporal) n m->bsz temporal n m', bsz=batch_size) # 相当于view操作
    # sp_graph = []
    # for start in range(0, N, num_object):  # (640, 640) ===> (32, 20, 20)
    #     sp_graph.append(result[start:start + num_object, start: start + num_object].unsqueeze(0))  # 获取对角线位置上的矩阵
    
    # spatial_graph = torch.cat(sp_graph)  # [seq_len, num_object, num_object] == [32, 20, 20]
    # batch_sp_graph.append(spatial_graph.unsqueeze(0))
    
    # # 一个batch的空间图,最终的处理结果
    # batch_spatial_graph = torch.cat(batch_sp_graph) # [batch_size, seq_len, num_object, num_object] == [batch_size, 32, 20, 20]
    
    return batch_spatial_graph
    
    # 构建时间图和空间图时
    # 计算IoU的时候, 首先对输入的bounding_box进行reshape, (batch_size, seq_len * num_object, 4)
    # 然后，取batch中的一个样本  矩阵转置(4, seq_len * num_object), 计算后得到(batch_size, seq_len * num_object, seq_len * num_object)的关于IoU的矩阵
    # 
    # reshape, 获取每一对bounding box之间的IOU

# 构建时间图，保留了t and t + 2帧之间的边
def construct_temporal_graph(input_bounding_box, input_features, tao, miu):
    batch_size, seq_len, num_object, _ = input_bounding_box.shape
    input_bounding_box = input_bounding_box.view(batch_size, -1, 4) # reshape into (batch_size, 640, 4)
    _, _, _, feature_dim = input_features.shape
    input_features = input_features.view(batch_size, -1, feature_dim)
    
    # for batch in range(batch_size):
    # input_bounding_box = box_center_to_corner(input_bounding_box)
    
    # 计算帧之间的bounding box IoU
    iou_matrix = compute_iou(input_bounding_box)
    iou_matrix = iou_matrix > tao

    appearance_similarity = compute_cosine_similarity(input_features)
    appearance_similarity = appearance_similarity > miu

    result = iou_matrix & appearance_similarity 

    # # 对角线位置上 20 * 20的矩阵需要全部置为零
    for start in range(0, result.shape[1], num_object):
        result[:, start: start + num_object, start: start + num_object] = 0

    return result # 判断返回值的数据类型

# 构建时间图，只保留相邻两帧之间的边，即t - 1, t, t + 1
def construct_temporal_graph2(input_bounding_box, input_features, tao, miu):
    batch_size, seq_len, num_object, _ = input_bounding_box.shape
    input_bounding_box = input_bounding_box.reshape(batch_size, -1, 4) # reshape into (batch_size, 640, 4)
    _, _, _, feature_dim = input_features.shape
    input_features = input_features.reshape(batch_size, -1, feature_dim)
    
    # input_bounding_box = box_center_to_corner(input_bounding_box)
    
    # 计算帧之间的bounding box IoU
    iou_matrix = compute_iou(input_bounding_box)
    iou_matrix = iou_matrix > tao

    appearance_similarity = compute_cosine_similarity(input_features)
    appearance_similarity = appearance_similarity > miu

    result = iou_matrix & appearance_similarity

    # 对角线位置上 20 * 20的矩阵需要全部置为零
    mask = torch.zeros(result.shape[1], result.shape[1], dtype=torch.bool, device=device)
    for start in range(0, result.shape[1] - num_object, num_object):
        mask[start: start + num_object, start + num_object: start + num_object * 2] = True
    mask = torch.logical_or(mask, mask.t())
    result = torch.logical_and(result, mask)

    return result # 判断返回值的数据类型

def test_iou():
    # 样例1
    # box1 = torch.tensor([[1.5, 1.5, 3, 3],
    #                      [3.5, 3.5, 3, 3],
    #                      [5.5, 5.5, 3, 3]], dtype=torch.float).to(device)
    
    # box2 = torch.tensor([[.5, .5, 1, 1],
    #                      [1.5, 1.5, 1, 1],
    #                      [2.5, 2.5, 1, 1]], dtype=torch.float).to(device)
    
    # input_bounding_box = torch.cat([box1.unsqueeze(0), box2.unsqueeze(0)])
    # input_bounding_box = box_center_to_corner(input_bounding_box)

    # 样例2
    box1 = torch.tensor([[0, 0, 3, 3],
                         [2, 2, 4, 4],
                         [3, 3, 4, 4]], dtype=torch.float).to(device)
    input_bounding_box = box1.unsqueeze(0)   # dim: [1, 3, 4]

    iou_matrix = compute_iou(input_bounding_box) # dim: [1, 2, 3, 4]

def test_cover():
    box1 = torch.tensor([[1, 1, 2, 2],
                         [2, 2, 4, 4],
                         [3, 3, 2, 2]], dtype=torch.float).to(device)
    
    box2 = torch.tensor([[.5, .5, 1, 1],
                         [1.5, 1.5, 1, 1],
                         [2.5, 2.5, 1, 1]], dtype=torch.float).to(device)
    
    input_bounding_box = torch.cat([box1.unsqueeze(0), box2.unsqueeze(0)])
    input_bounding_box = box_center_to_corner(input_bounding_box)
    cover = compute_cover(input_bounding_box)

# if __name__ == "__main__":
# #     # 构造输入
# #     input_features = torch.rand(batch_size, seq_len, num_object, feature_dim).to(device)
# #     input_bounding_box = torch.ones(batch_size, seq_len, num_object, 4).to(device)
    
# #     s_graph = construct_spatial_graph(input_bounding_box)   # dim: [batch_size, seq_len, num_object, num_ojbect]
# #     # 进行图卷积时，reshape into [B, 20, 20], B = batch_size * seq_len     
# #     # 输入的特征: [batch_size, seq_len, num_object, dim], reshape into [batch_size * seq_len, 20, dim]   保持(B, N, N) 与 (B, N, D)统一

# #     # test_ratio()
# #     t_graph = construct_temporal_graph(input_bounding_box, input_features)  # dim: [batch_size, seq_len * num_object, seq_len * num_object]
# #     # 输入的特征: [batch_size, seq_len, num_object, dim], reshape into [B, 640, dim]
#     test_cover()
