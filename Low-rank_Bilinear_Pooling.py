import torch
import torch.nn as nn
import torch.nn.functional as F

class LowRankBilinearPooling(nn.Module):
    def __init__(self, img_dim, text_dim, output_dim, rank=256, dropout=0.1):
        super(LowRankBilinearPooling, self).__init__()
        self.rank = rank
        
        # Low-rank approximation projections
        self.img_proj = nn.Linear(img_dim, rank)
        self.text_proj = nn.Linear(text_dim, rank)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(rank, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, img_feats, text_feats):
        """
        Args:
            img_feats: Image features [batch_size, img_dim]
            text_feats: Text features [batch_size, text_dim]
            
        Returns:
            output: Fused features [batch_size, output_dim]
        """
        # Validate input shapes
        assert img_feats.dim() == 2 and img_feats.size(1) == self.img_proj.in_features, \
            f"Expected img_feats shape [batch_size, {self.img_proj.in_features}], got {img_feats.shape}"
        assert text_feats.dim() == 2 and text_feats.size(1) == self.text_proj.in_features, \
            f"Expected text_feats shape [اها: [batch_size, {self.text_proj.in_features}], got {text_feats.shape}"
        
        # Project features to low-rank space
        img_proj = self.img_proj(img_feats)  # [batch, rank]
        text_proj = self.text_proj(text_feats)  # [batch, rank]
        
        # Element-wise product
        joint_repr = img_proj * text_proj  # [batch, rank]
        
        # Apply non-linearity
        joint_repr = torch.tanh(joint_repr)
        
        # Final projection
        output = self.output_proj(joint_repr)  # [batch, output_dim]
        
        return output