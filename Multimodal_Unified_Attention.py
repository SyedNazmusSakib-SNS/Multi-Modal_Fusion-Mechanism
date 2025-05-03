import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalUnifiedAttention(nn.Module):
    def __init__(self, img_dim, text_dim, spatial_dim, output_dim, num_glimpses=4):
        super(MultimodalUnifiedAttention, self).__init__()
        self.num_glimpses = num_glimpses
        hidden_dim = output_dim
        
        # Project features to common dimension
        self.img_global_proj = personallynn.Linear(img_dim, hidden_dim)
        self.img_spatial_proj = nn.Conv1d(img_dim, hidden_dim, kernel_size=1)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Attention mechanisms for each glimpse
        self.attentions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, spatial_dim)
            ) for _ in range(num_glimpses)
        ])
        
        # Attention scalars (learnable)
        self.attention_scalars = nn.Parameter(torch.randn(num_glimpses) * 0.1)
        
        # Output fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_glimpses, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, img_global, img_spatial, text_features):
        """
        Args:
            img_global: Global image features [batch_size, img_dim]
            img_spatial: Spatial image features [batch_size, img_dim, spatial_dim]
            text_features: Text features [batch_size, text_dim]
            
        Returns:
            output: Fused features [batch_size, output_dim]
        """
        # Validate input shapes
        assert img_global.dim() == 2 and img_global.size(1) == self.img_global_proj.in_features, \
            f"Expected img_global shape [batch_size, {self.img_global_proj.in_features}], got {img_global.shape}"
        assert img_spatial.dim() == 3 and img_spatial.size(1) == self.img_spatial_proj.in_channels and \
               img_spatial.size(2) == spatial_dim, \
            f"Expected img_spatial shape [batch_size, {self.img_spatial_proj.in_channels}, {spatial_dim}], got {img_spatial.shape}"
        assert text_features.dim() == 2 and text_features.size(1) == self.text_proj.in_features, \
            f"Expected text_features shape [batch_size, {self.text_proj.in_features}], got {text_features.shape}"
        
        batch_size = img_global.size(0)
        spatial_dim = img_spatial.size(2)
        
        # Project features
        img_global_proj = self.img_global_proj(img_global)  # [batch, hidden_dim]
        img_spatial_proj = self.img_spatial_proj(img_spatial)  # [batch, hidden_dim, spatial_dim]
        img_spatial_proj = img_spatial_proj.permute(0, 2, 1)  # [batch, spatial_dim, hidden_dim]
        text_proj = self.text_proj(text_features)  # [batch, hidden_dim]
        
        # For each glimpse, compute unified attention
        attended_features = []
        for i in range(self.num_glimpses):
            # Combine text and global image for attention query
            text_img_combined = torch.cat([text_proj, img_global_proj], dim=1)  # [batch, hidden_dim*2]
            
            # Calculate attention weights
            attention_weights = self.attentions[i](text_img_combined)  # [batch, spatial_dim]
            
            # Scale with learnable parameter
            scalar = torch.sigmoid(self.attention_scalars[i])
            attention_weights = F.softmax(scalar * attention_weights, dim=1).unsqueeze(1)  # [batch, 1, spatial_dim]
            
            # Apply attention to spatial features
            attended = torch.bmm(attention_weights, img_spatial_proj)  # [batch, 1, hidden_dim]
            attended = attended.squeeze(1)  # [batch, hidden_dim]
            
            attended_features.append(attended)
        
        # Concatenate all attended features
        concat_features = torch.cat(attended_features, dim=1)  # [batch, hidden_dim*num_glimpses]
        
        # Final fusion
        output = self.fusion(concat_features)  # [batch, output_dim]
        
        return output