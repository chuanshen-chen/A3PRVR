import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Gated_Embedding_Unit和Context_Gating类构成了Gated Embedding Module
# input: dim * 1, output: dim2 * 1
class Gated_Embedding_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension, gating=True):
        super(Gated_Embedding_Unit, self).__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = Context_Gating(output_dimension, add_batch_norm=False)
        self.gating = gating
  
    def forward(self,x):
        r"""
        Args:
            x: batch_size, dim
        """
        # x: Z_0,  
        x = self.fc(x) # Z_1  dim:[10, 2, 1024]
        if self.gating:
            x = self.cg(x) # Z_2
        x = F.normalize(x) # Z

        return x

class Context_Gating(nn.Module):
    def __init__(self, dimension, add_batch_norm=True):
        super(Context_Gating, self).__init__()
        self.fc = nn.Linear(dimension, dimension) # bias term默认为True
        self.add_batch_norm = add_batch_norm
        self.batch_norm = nn.BatchNorm1d(dimension)
        
    def forward(self,x):
        # x: Z_1
        x1 = self.fc(x) # right term

        if self.add_batch_norm:
            x1 = x1.transpose(1, 2) # 维度转换为[batch_size, feature_dim, len], 符合batch_norm的要求
            x1 = self.batch_norm(x1) # 原代码的输入是2维，现在输入是3维，要修改batch norm 1d的细节
            x1 = x1.transpose(1, 2) # 恢复原来的维度

        x = torch.cat((x, x1), dim=-1)
        
        return F.glu(x, -1) # gated linear unit, 相当于sigmoid + element-wise multiply

def gated_embedding_module(graph_feature, output_dim):
    feature_dim = graph_feature.size(-1)
    geu = Gated_Embedding_Unit(feature_dim, output_dim).to(device)
    feature = geu(graph_feature)
    # feature, _ = torch.max(feature, dim=1)

    return feature

def gated_embedding_module_old(s_graph_feature, t_graph_feature):
    # B = s_graph_feature.size(0)
    batch_size, seq_len, num_object, feature_dim = s_graph_feature.shape
    s_graph_feature = s_graph_feature.view(batch_size, -1, feature_dim)
    output_dim = 256
    
    spatial_geu = Gated_Embedding_Unit(feature_dim, output_dim, gating=True).to(device)
    temporal_geu = Gated_Embedding_Unit(feature_dim, output_dim, gating=True).to(device)
    
    spatial_feature = spatial_geu(s_graph_feature)
    temporal_feature = temporal_geu(t_graph_feature)

    # 对GEU处理结果进行max pooling降维
    spatial_feature, _ = torch.max(spatial_feature, dim=1)
    temporal_feature, _ = torch.max(temporal_feature, dim=1)

    # print("spatial feature:", spatial_feature.shape)
    # print("temporal feature:", temporal_feature.shape)

    return spatial_feature, temporal_feature
