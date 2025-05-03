import torch
import torch.nn as nn

class DynamicParameterPrediction(nn.Module):
    def __init__(self, img_dim, text_dim, output_dim, hidden_dim=512, dropout=0.3):
        super(DynamicParameterPrediction, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Dynamic parameter prediction network
        self.weight_predictor = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, img_dim * hidden_dim)
        )
        
        # Bias prediction
        self.bias_predictor = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
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
        assert img_feats.dim() == 2 and img_feats.size(1) == self.weight_predictor[2].in_features // self.hidden_dim, \
            f"Expected img_feats shape [batch_size, {self.weight_predictor[2].in_features // self.hidden_dim}], got {img_feats.shape}"
        assert text_feats.dim() == 2 and text_feats.size(1) == self.weight_predictor[0].in_features, \
            f"Expected text_feats shape [batch_size, {self.weight_predictor[0].in_features}], got {text_feats.shape}"
        
        batch_size = img_feats.size(0)
        img_dim = img_feats.size(1)
        
        # Predict dynamic weights and bias
        predicted_weights = self.weight_predictor(text_feats)  # [batch, img_dim*hidden_dim]
        predicted_weights = predicted_weights.view(batch_size, self.hidden_dim, img_dim)  # [batch, hidden_dim, img_dim]
        
        predicted_bias = self.bias_predictor(text_feats)  # [batch, hidden_dim]
        
        # Apply dynamic transformation
        dynamic_output = torch.bmm(predicted_weights, img_feats.unsqueeze(2)).squeeze(2)  # [batch, hidden_dim]
        dynamic_output = dynamic_output + predicted_bias  # [batch, hidden_dim]
        
        # Final output layer
        output = self.output_layer(dynamic_output)  # [batch, output_dim]
        
        return output