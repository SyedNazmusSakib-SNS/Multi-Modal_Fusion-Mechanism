import torch
import torch.nn as nn

class MultimodalBottleneckTransformer(nn.Module):
    def __init__(self, img_dim, text_dim, output_dim, bottleneck_dim=64, num_heads=4):
        super(MultimodalBottleneckTransformer, self).__init__()
        self.output_dim = output_dim
        
        # Project image and text to common dimension
        self.img_proj = nn.Linear(img_dim, output_dim)
        self.text_proj = nn.Linear(text_dim, output_dim)
        
        # Bottleneck tokens (learnable)
        self.num_bottleneck = bottleneck_dim
        self.bottleneck = nn.Parameter(torch.randn(1, bottleneck_dim, output_dim))
        
        # Self-attention for bottleneck
        self.self_attn = nn.MultiheadAttention(output_dim, num_heads, batch_first=True)
        
        # Cross-attention: bottleneck to modalities
        self.cross_attn_img = nn.MultiheadAttention(output_dim, num_heads, batch_first=True)
        self.cross_attn_text = nn.MultiheadAttention(output_dim, num_heads, batch_first=True)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 4, output_dim)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.norm3 = nn.LayerNorm(output_dim)
        self.norm4 = nn.LayerNorm(output_dim)
        
        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(bottleneck_dim * output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
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
            f"Expected text_feats shape [batch_size, {self.text_proj.in_features}], got {text_feats.shape}"
        
        batch_size = img_feats.size(0)
        
        # Project input features
        img_proj = self.img_proj(img_feats).unsqueeze(1)  # [batch, 1, output_dim]
        text_proj = self.text_proj(text_feats).unsqueeze(1)  # [batch, 1, output_dim]
        
        # Expand bottleneck tokens for batch
        bottleneck = self.bottleneck.expand(batch_size, -1, -1)  # [batch, bottleneck_dim, output_dim]
        
        # Self-attention within bottleneck
        bottleneck_attn, _ = self.self_attn(bottleneck, bottleneck, bottleneck)
        bottleneck = bottleneck + bottleneck_attn
        bottleneck = self.norm1(bottleneck)
        
        # Cross-attention: bottleneck -> image
        bottleneck_img, _ = self.cross_attn_img(bottleneck, img_proj, img_proj)
        bottleneck = bottleneck + bottleneck_img
        bottleneck = self.norm2(bottleneck)
        
        # Cross-attention: bottleneck -> text
        bottleneck_text, _ = self.cross_attn_text(bottleneck, text_proj, text_proj)
        bottleneck = bottleneck + bottleneck_text
        bottleneck = self.norm3(bottleneck)
        
        # Feedforward
        bottleneck_ffn = self.ffn(bottleneck)
        bottleneck = bottleneck + bottleneck_ffn
        bottleneck = self.norm4(bottleneck)
        
        # Flatten bottleneck and project to output
        bottleneck_flat = bottleneck.reshape(batch_size, -1)
        output = self.output_mlp(bottleneck_flat)
        
        return output