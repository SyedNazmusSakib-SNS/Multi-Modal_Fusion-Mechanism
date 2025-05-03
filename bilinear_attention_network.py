import numpy as np
import torch
import torch.nn as nn

# Bilinear Attention Networks (BAN)
class BilinearAttention(nn.Module):
    def __init__(self, v_dim, q_dim, output_dim, glimpse=8):
        super(BilinearAttention, self).__init__()
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.output_dim = output_dim
        self.glimpse = glimpse
        
        self.v_nets = nn.ModuleList([
            nn.Linear(v_dim, output_dim) for _ in range(glimpse)
        ])
        self.q_nets = nn.ModuleList([
            nn.Linear(q_dim, output_dim) for _ in range(glimpse)
        ])
        self.dropout = nn.Dropout(0.5)
        
        # Fusing features
        self.fusion = nn.Sequential(
            nn.Linear(glimpse * output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
    def forward(self, v, q):
        batch_size = v.size(0)
        v_num = v.size(1)
        
        q_repeated = q.unsqueeze(1).repeat(1, v_num, 1)  
        
        attentions = []
        for glimpse_idx in range(self.glimpse):
            v_proj = self.v_nets[glimpse_idx](v)  
            q_proj = self.q_nets[glimpse_idx](q_repeated)  
            
            joint_repr = v_proj * q_proj  
            
            attention = torch.sum(joint_repr, dim=1)  
            attentions.append(attention)
        

        concat_attentions = torch.cat(attentions, dim=1)  
        
        output = self.fusion(concat_attentions)  
        
        return output