# Multi-Modal Fusion Mechanism

This repository implements a collection of multimodal fusion mechanisms designed to combine features from multiple modalities (e.g., image and text) for tasks such as visual question answering, image captioning, and multimodal classification. Each mechanism is implemented as a PyTorch module, providing flexible and reusable components for multimodal deep learning research and applications.

## Table of Contents

- [Overview](#overview)
- [Implemented Models](#implemented-models)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

Multimodal fusion is a critical component in systems that process and integrate information from multiple sources, such as images and text. This repository provides implementations of various state-of-the-art fusion techniques, each designed to capture different types of interactions between modalities. The models are implemented in PyTorch and are designed to be modular, allowing easy integration into larger deep learning pipelines.

## Implemented Models

The repository includes the following multimodal fusion mechanisms:

### Feature-wise Linear Modulation (FiLM)
- **File**: `feature_wise_linear_modulation.py`
- **Description**: Modulates image features using text-conditioned scaling and shifting parameters
- **Use case**: Suitable for tasks requiring fine-grained feature modulation

### Compact Bilinear Pooling (MCB)
- **File**: `Compact_Bilinear_Pooling.py`
- **Description**: Approximates bilinear pooling using Count Sketch and FFT for efficient high-order interactions
- **Use case**: Ideal for capturing complex interactions with reduced computational cost

### Multimodal Factorized High-order Pooling (MFH)
- **File**: `Multimodal_Factorized_High-order_Pooling.py`
- **Description**: Factorizes bilinear pooling into multiple low-rank projections with signed square root normalization
- **Use case**: Balances expressiveness and efficiency for high-order interactions

### Multimodal Transformer Fusion
- **File**: `Multimodal_Transformer_Fusion.py`
- **Description**: Uses a transformer encoder to fuse projected image and text features with positional embeddings
- **Use case**: Effective for modeling long-range dependencies between modalities

### Dynamic Parameter Prediction Network (DPPNet)
- **File**: `Dynamic_Parameter_Prediction_Network.py`
- **Description**: Generates dynamic weights and biases from text features to transform image features
- **Use case**: Useful for tasks requiring text-conditioned transformations

### Low-rank Bilinear Pooling (MLB)
- **File**: `Low-rank_Bilinear_Pooling.py`
- **Description**: Projects features to a low-rank space for efficient bilinear interactions with tanh non-linearity
- **Use case**: Lightweight and effective for simple bilinear fusion

### Multimodal Bottleneck Transformer (MBT)
- **File**: `Multimodal_Bottleneck_Transformer.py`
- **Description**: Uses learnable bottleneck tokens with self- and cross-attention to fuse modalities
- **Use case**: Suitable for compact and expressive multimodal representations

### Multimodal Unified Attention (MUA)
- **File**: `Multimodal_Unified_Attention.py`
- **Description**: Applies multiple attention glimpses to spatial and global image features conditioned on text
- **Use case**: Ideal for tasks requiring spatial attention in images

### Bilinear Attention Network (BAN)
- **File**: `bilinear_attention_network.py`
- **Description**: Combines bilinear pooling with attention mechanisms to focus on relevant feature interactions
- **Use case**: Effective for tasks requiring attention-driven fusion

## Installation

To use the models in this repository, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SyedNazmusSakib-SNS/Multi-Modal_Fusion-Mechanism.git
   cd Multi-Modal_Fusion-Mechanism
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install torch torchvision
   ```
   
   Ensure your PyTorch version is compatible with your system (CPU/GPU). Check [PyTorch's official website](https://pytorch.org/) for installation instructions tailored to your setup.

## Usage

Each model is implemented as a standalone PyTorch `nn.Module`, making it easy to integrate into your own projects. Below is an example of how to use the Multimodal Transformer Fusion model:

```python
import torch
from Multimodal_Transformer_Fusion import MultimodalTransformerFusion

# Define model parameters
img_dim = 2048  # e.g., ResNet-152 features
text_dim = 768  # e.g., BERT embeddings
output_dim = 512
num_heads = 8
num_layers = 2

# Initialize model
model = MultimodalTransformerFusion(
    img_dim=img_dim,
    text_dim=text_dim,
    output_dim=output_dim,
    num_heads=num_heads,
    num_layers=num_layers
)

# Sample input features
batch_size = 32
img_feats = torch.randn(batch_size, img_dim)
text_feats = torch.randn(batch_size, text_dim)

# Forward pass
output = model(img_feats, text_feats)  # Shape: [batch_size, output_dim]
print(output.shape)  # Expected: torch.Size([32, 512])
```

To use other models, import the corresponding class from its file and follow a similar pattern. Refer to each model's source file for specific input requirements (e.g., Multimodal_Unified_Attention requires spatial image features).

## Repository Structure

```
Multi-Modal_Fusion-Mechanism/
├── Compact_Bilinear_Pooling.py              # Compact Bilinear Pooling (MCB)
├── Dynamic_Parameter_Prediction_Network.py  # Dynamic Parameter Prediction Network (DPPNet)
├── Low-rank_Bilinear_Pooling.py             # Low-rank Bilinear Pooling (MLB)
├── Multimodal_Bottleneck_Transformer.py     # Multimodal Bottleneck Transformer (MBT)
├── Multimodal_Factorized_High-order_Pooling.py  # Multimodal Factorized High-order Pooling (MFH)
├── Multimodal_Transformer_Fusion.py         # Multimodal Transformer Fusion
├── Multimodal_Unified_Attention.py          # Multimodal Unified Attention (MUA)
├── bilinear_attention_network.py            # Bilinear Attention Network (BAN)
├── feature_wise_linear_modulation.py        # Feature-wise Linear Modulation (FiLM)
├── README.md                                # This file
```

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes and commit them (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request with a detailed description of your changes

Please ensure your code follows the repository's style (e.g., PEP 8 for Python) and includes appropriate documentation.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

**Author**: SyedNazmusSakib-SNS  
**Last Updated**: May 4, 2025
