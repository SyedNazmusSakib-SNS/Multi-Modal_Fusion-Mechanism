import torch
import torch.nn as nn
import torch.nn.init as init

class MultimodalTransformerFusion(nn.Module):
    def __init__(self, img_dim, text_dim, output_dim, num_heads=8, num_layers=2):
        super(MultimodalTransformerFusion, self).__init__()
        self.output_dim = output_dim
        hidden_dim = output_dim
        
        # Project image and text features to the same dimension
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Positional embeddings for image and text (2 special tokens)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 2, hidden_dim))
        init.xavier_uniform_(self.pos_embedding)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output MLP (simplified)
        self.output_mlp = nn.Linear(hidden_dim, output_dim)
        
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
        
        # Project to common dimension
        img_proj = self.img_proj(img_feats).unsqueeze(1)  # [batch, 1, hidden_dim]
        text_proj = self.text_proj(text_feats).unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Concatenate image and text tokens
        multimodal_features = torch.cat([img_proj, text_proj], dim=1)  # [batch, 2, hidden_dim]
        
        # Add positional embeddings
        multimodal_features = multimodal_features + self.pos_embedding
        
        # Pass through transformer
        transformer_output = self.transformer(multimodal_features)  # [batch, 2, hidden_dim]
        
        # Use the output corresponding to the first position
        fusion_output = transformer_output[:, 0, :]  # [batch, hidden_dim]
        
        # Final projection
        output = self.output_mlp(fusion_output)  # [batch, output_dim]
        
        return output