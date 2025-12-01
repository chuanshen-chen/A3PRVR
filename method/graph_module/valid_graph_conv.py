import torch
import torch.nn as nn
# from torch_geometric.nn import GCNConv
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
import torch.nn.functional as F
# from torch_geometric.data.batch import Batch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gcn = GCNConv(in_channels, out_channels, add_self_loops=False, normalize=True)
    
    def forward(self, data):
        r"""
        Args:
            x: feature, [N, dim]
            edge_index: 邻接矩阵, [N, N]            
        """
        x, edge_index = data.x, data.edge_index
        h = self.gcn(x, edge_index)
        return h
    
# 定义图卷积
r"""构建多层图卷积神经网络参见链接：
https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
"""
class TwoLayerGCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)

    def forward(self, data):
        r"""
        Args:
            x: feature, [N, dim]
            edge_index: 邻接矩阵, [N, N]            
        """
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x) # ReLU激活函数
        x = self.conv2(x, edge_index)

        return x

def spatial_conv(sp_model, input_features, s_graph, out_channels):
    r"""
        Args:
            input_features: dim[batch_size, seq_len, num_object, feature_dim]
            s_graph: dim[batch_size, seq_len, num_object, num_object]
        
    """
    batch_size, seq_len, num_object, feature_dim = input_features.shape
    input_features = input_features.view(-1, num_object, feature_dim)
    s_graph = s_graph.reshape(-1, num_object, num_object)  # [B, N, N]
    B = s_graph.size(0)

    """
    # sg = []
    # for i in range(s_graph.size(0)):
    #     temp = s_graph[i].to_sparse().indices()
    #     sg.append(temp.unsqueeze(0))
    
    # s_graph = torch.cat(sg)
    
    # s_graph = s_graph.to_sparse().indices()
    # s_graph, _ = s_graph.to_sparse().to("cpu")
    # s_graph = sp.csr_matrix(s_graph.to_dense().numpy())
    # s_graph, _ = from_scipy_sparse_matrix(s_graph)
    # s_graph = s_graph.to(device)
    """

    data_list = []
    for i in range(B):
        edge_index = s_graph[i].to_sparse().indices()
        x = input_features[i]
        data = Data(x=x, edge_index=edge_index)
        data_list.append(data)

    loader = DataLoader(data_list, batch_size=32)
    # model = GCN(feature_dim, out_channels).to(device)
    # model = TwoLayerGCN(feature_dim, out_channels).to(device)

    # 批处理代码
    batch_output = []
    for data in loader:
        out = sp_model(data)
        batch_output.append(out)

    batch_output = torch.cat(batch_output)
    batch_output = batch_output.view(batch_size, seq_len, num_object, out_channels)

    return batch_output

def temporal_conv(temporal_graph_model, input_features, t_graph, out_channels):
    batch_size, seq_len, num_object, feature_dim = input_features.shape
    input_features = input_features.reshape(batch_size, -1, feature_dim)
    # temporal_graph_model = GCN(feature_dim, out_channels).to(device)
    # temporal_graph_model = TwoLayerGCN(feature_dim, out_channels).to(device)

    data_list = []  # 创建图, 构造Data对象
    for i in range(batch_size):
        x = input_features[i]
        edge_index = t_graph[i].to_sparse().indices()
        data = Data(x = x, edge_index=edge_index)
        data_list.append(data)
    
    loader = DataLoader(data_list, batch_size=32) # 默认使用batch_size=1, 可以根据实际情况调整
    # loader = DataLoader(data_list) # 默认使用batch_size=1, 可以根据实际情况调整
    
    batch_output = []
    for data in loader:
        out = temporal_graph_model(data) # dim[mini_batch * 1280, 512]
        batch_output.append(out)
    
    batch_output = torch.cat(batch_output)
    batch_output = batch_output.view(batch_size, -1, out_channels)

    # print(type(batch_output))
    # print(batch_output.shape)
    return batch_output
