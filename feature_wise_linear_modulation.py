import torch
import torch.nn as nn

class FiLMLayer(nn.Module):
    def __init__(self, img_dim, text_dim, output_dim, dropout=0.3):
        super(FiLMLayer, self).__init__()
        
        self.condition_net = nn.Linear(text_dim, output_dim * 2)
        
        self.img_proj = nn.Linear(img_dim, output_dim)
        
        self.output_layer = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, img_feats, text_feats):
        assert img_feats.dim() == 2 and img_feats.size(1) == self.img_proj.in_features, \
            f"Expected img_feats shape [batch_size, {self.img_proj.in_features}], got {img_feats.shape}"
        assert text_feats.dim() == 2 and text_feats.size(1) == self.condition_net.in_features, \
            f"Expected text_feats shape [batch_size, {self.condition_net.in_features}], got {text_feats.shape}"
        
        # Project image features
        img_proj = self.img_proj(img_feats)  # [batch_size, output_dim]
        
        # Generate FiLM parameters (gamma and beta)
        film_params = self.condition_net(text_feats)  # [batch_size, output_dim*2]
        gamma, beta = torch.split(film_params, film_params.size(1)//2, dim=1)  # [batch_size, output_dim] each
        
        # Apply feature-wise modulation: gamma * x + beta
        modulated = gamma * img_proj + beta
        
        # Apply normalization and non-linearity
        output = self.output_layer(modulated)
        
        return output