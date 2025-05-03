import torch
import torch.nn as nn
import torch.nn.functional as F

class FactorizedHighOrderPooling(nn.Module):
    def __init__(self, img_dim, text_dim, output_dim, factor_dim=256, num_factors=2, dropout=0.3):
        super(FactorizedHighOrderPooling, self).__init__()
        self.output_dim = output_dim
        self.factor_dim = factor_dim
        self.num_factors = num_factors
        
        # Projection matrices for factorized bilinear pooling
        self.img_factors = nn.ModuleList([
            nn.Linear(img_dim, factor_dim) for _ in range(num_factors)
        ])
        self.text_factors = nn.ModuleList([
            nn.Linear(text_dim, factor_dim) for _ in range(num_factors)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(num_factors * factor_dim, output_dim),
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
        assert img_feats.dim() == 2 and img_feats.size(1) == self.img_factors[0].in_features, \
            f"Expected img_feats shape [batch_size, {self.img_factors[0].in_features}], got {img_feats.shape}"
        assert text_feats.dim() == 2 and text_feats.size(1) == self.text_factors[0].in_features, \
            f"Expected text_feats shape [batch_size, {self.text_factors[0].in_features}], got {text_feats.shape}"
        
        # Apply factorized bilinear pooling for each factor
        factor_outputs = []
        for i in range(self.num_factors):
            img_proj = self.img_factors[i](img_feats)  # [batch, factor_dim]
            text_proj = self.text_factors[i](text_feats)  # [batch, factor_dim]
            
            # Element-wise product
            factor_output = img_proj * text_proj  # [batch, factor_dim]
            
            # Signed square root normalization
            factor_output = torch.sign(factor_output) * torch.sqrt(torch.clamp(torch.abs(factor_output), min=1e-5))
            
            # L2 normalization
            factor_output = F.normalize(factor_output, p=2, dim=1)
            
            factor_outputs.append(factor_output)
        
        # Concatenate all factor outputs
        concat_factors = torch.cat(factor_outputs, dim=1)  # [batch, num_factors*factor_dim]
        
        # Project to output dimension
        output = self.output_proj(concat_factors)  # [batch, output_dim]
        
        return output