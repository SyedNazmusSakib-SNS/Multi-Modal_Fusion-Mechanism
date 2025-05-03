import torch
import torch.nn as nn
import torch.nn.functional as F

class CompactBilinearPooling(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim, sum_pool=True):
        super(CompactBilinearPooling, self).__init__()
        self.output_dim = output_dim
        self.sum_pool = sum_pool
        
        # Random projection matrices for Count Sketch (non-learnable)
        self.h1 = nn.Parameter(torch.randint(0, output_dim, (input_dim1,)), requires_grad=False)
        self.h2 = nn.Parameter(torch.randint(0, output_dim, (input_dim2,)), requires_grad=False)
        self.s1 = nn.Parameter(torch.randint(0, 2, (input_dim1,)).float() * 2 - 1, requires_grad=False)
        self.s2 = nn.Parameter(torch.randint(0, 2, (input_dim2,)).float() * 2 - 1, requires_grad=False)
        
    def count_sketch(self, x, h, s):
        """
        Vectorized Count Sketch projection.
        Args:
            x: Input tensor [batch_size, input_dim] or [batch_size, input_dim, ...]
            h: Hash indices [input_dim]
            s: Sign values [input_dim]
        Returns:
            y: Sketch projection [batch_size, output_dim, ...]
        """
        # Flatten spatial dimensions if present
        if x.dim() > 2:
            batch_size, input_dim = x.size(0), x.size(1)
            x = x.view(batch_size, input_dim, -1)  # [batch_size, input_dim, prod(...)]
        else:
            batch_size, input_dim = x.shape
            x = x.unsqueeze(-1)  # [batch_size, input_dim, 1]
        
        # Initialize output
        y = torch.zeros(batch_size, self.output_dim, x.size(-1), device=x.device)
        
        # Vectorized scatter-add
        h = h.view(1, -1, 1).expand(batch_size, -1, x.size(-1))  # [batch_size, input_dim, prod(...)]
        s = s.view(1, -1, 1).expand(batch_size, -1, x.size(-1))  # [batch_size, input_dim, prod(...)]
        x = x * s  # Apply signs
        y.scatter_add_(1, h, x)  # Accumulate into output_dim bins
        
        # Reshape to [batch_size, output_dim] if no spatial dims
        if x.size(-1) == 1:
            y = y.squeeze(-1)
        
        return y
    
    def forward(self, x1, x2):
        """
        Args:
            x1: First input tensor [batch_size, input_dim1] or [batch_size, input_dim1, ...]
            x2: Second input tensor [batch_size, input_dim2] or [batch_size, input_dim2, ...]
            
        Returns:
            z: Compact bilinear pooled feature [batch_size, output_dim]
        """
        # Validate input shapes
        assert x1.dim() >= 2, f"Expected x1 to have at least 2 dims, got {x1.shape}"
        assert x2.dim() >= 2, f"Expected x2 to have at least 2 dims, got {x2.shape}"
        assert x1.size(0) == x2.size(0), f"Batch sizes must match: {x1.shape} vs {x2.shape}"
        assert x1.size(1) == self.h1.size(0), f"x1 dim1 must match input_dim1: {x1.shape} vs {self.h1.shape}"
        assert x2.size(1) == self.h2.size(0), f"x2 dim2 must match input_dim2: {x2.shape} vs {self.h2.shape}"
        
        # Move parameters to input device
        h1, s1 = self.h1.to(x1.device), self.s1.to(x1.device)
        h2, s2 = self.h2.to(x2.device), self.s2.to(x2.device)
        
        # Count Sketch projections
        phi1 = self.count_sketch(x1, h1, s1)  # [batch_size, output_dim, ...] or [batch_size, output_dim]
        phi2 = self.count_sketch(x2, h2, s2)  # [batch_size, output_dim, ...] or [batch_size, output_dim]
        
        # Sum pool spatial dimensions if needed
        if self.sum_pool and phi1.dim() > 2:
            phi1 = phi1.sum(dim=-1)  # [batch_size, output_dim]
            phi2 = phi2.sum(dim=-1)  # [batch_size, output_dim]
        
        # FFT, element-wise product, and IFFT
        fft1 = torch.fft.rfft(phi1, dim=1)  # Use rfft for real inputs
        fft2 = torch.fft.rfft(phi2, dim=1)
        fft_product = fft1 * fft2
        z = torch.fft.irfft(fft_product, n=self.output_dim, dim=1)  # Ensure output length matches
        
        # L2 normalization
        z = F.normalize(z, p=2, dim=1)
        
        return z