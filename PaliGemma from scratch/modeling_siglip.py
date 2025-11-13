from typing import Optional, Tuple
import torch
import torch.nn as nn

"""
SigLIP Vision Transformer - A model for processing images

Overall Architecture Flow:
Image → Patch Embeddings → Transformer Encoder → Output Features

Visual representation of the full pipeline:
┌─────────────────┐
│  Input Image    │  224x224x3 (RGB image)
│   [B, 3, 224,   │
│      224]       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Patch Embedding │  Splits image into patches and embeds them
│  (Conv2d)       │  Patches: 16x16 pixels each
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Position Embed  │  Adds positional information to each patch
│  [B, 196, 768]  │  (224/16)² = 196 patches
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Transformer     │  12 layers of self-attention + MLP
│  Encoder        │  Processes relationships between patches
│  (12 layers)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Layer Norm      │  Final normalization
│  [B, 196, 768]  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Output Features │  Rich representations of the image
│  [B, 196, 768]  │  Each patch has a 768-dim vector
└─────────────────┘
"""

class SiglipVisionConfig:
    """
    Configuration class to store the hyperparameters for the SigLIP vision model.
    
    Think of this as the "blueprint" that defines:
    - How big the model is (hidden_size, layers)
    - How images are processed (image_size, patch_size)
    - Architectural details (num_heads, intermediate_size)
    """

    def __init__(
        self,
        hidden_size=768,              # Embedding dimension for each token/patch
        intermediate_size=3072,       # Size of MLP hidden layer (usually 4x hidden_size)
        num_hidden_layers=12,         # Number of transformer encoder layers
        num_attention_heads=12,       # Number of attention heads (hidden_size should be divisible by this)
        num_channels=3,               # RGB images have 3 channels
        image_size=224,               # Input image dimensions (224x224)
        patch_size=16,                # Each patch is 16x16 pixels
        layer_norm_eps=1e-6,          # Small constant for numerical stability in LayerNorm
        attention_dropout=0.0,        # Dropout probability in attention (0.0 = no dropout)
        num_image_tokens: int = None, # Number of image tokens (calculated as (image_size/patch_size)²)
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens

class SiglipVisionEmbeddings(nn.Module):
    """
    Converts an image into a sequence of embedded patches with positional information.
    
    Visual Process:
    ┌──────────────────────────────────────┐
    │         Original Image               │
    │         224 x 224 pixels             │
    │  ┌────┬────┬────┬────┬─────┬────┐    │
    │  │ 16 │ 16 │ 16 │ 16 │ ... │ 16 │    │ Each patch: 16x16 pixels
    │  │x16 │x16 │x16 │x16 │     │x16 │    │ Total patches: 14x14 = 196
    │  ├────┼────┼────┼────┼─────┼────┤    │
    │  │ P1 │ P2 │ P3 │ P4 │ ... │PN  │    │ P = Patch
    │  └────┴────┴────┴────┴─────┴────┘    │
    └──────────────────────────────────────┘
                    ↓
    ┌──────────────────────────────────────┐
    │      Patch Embeddings (Conv2d)       │
    │  Each 16x16x3 patch → 768-dim vector │
    └──────────────────────────────────────┘
                    ↓
    ┌──────────────────────────────────────┐
    │      Add Position Embeddings         │
    │  Tell model where each patch is      │
    │  Patch 1 at position 0               │
    │  Patch 2 at position 1, etc.         │
    └──────────────────────────────────────┘
                    ↓
          [Batch, 196, 768]
    """
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # Conv2d with kernel_size=stride=patch_size acts as a "patchifier"
        # It extracts non-overlapping patches and linearly projects them
        # Input: [B, 3, 224, 224] → Output: [B, 768, 14, 14]
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,   # 3 for RGB
            out_channels=self.embed_dim,       # 768 embedding dimensions
            kernel_size=self.patch_size,       # 16x16 kernel
            stride=self.patch_size,            # 16 stride (no overlap)
            padding="valid",                   # No padding added
        )

        # Calculate number of patches: (224/16) * (224/16) = 14 * 14 = 196
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        
        # Learnable position embeddings: one 768-dim vector for each of 196 positions
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        
        # Position IDs: [0, 1, 2, ..., 195] to index the position embeddings
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),  # Shape: [1, 196]
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [Batch_Size, 3, 224, 224] - RGB images
            
        Returns:
            embeddings: [Batch_Size, 196, 768] - Embedded patches with positions
        """
        _, _, height, width = pixel_values.shape  # [B, 3, 224, 224]
        
        # Step 1: Extract patches using convolution
        # Visual: Sliding 16x16 window with stride 16 across the image
        # [B, 3, 224, 224] → [B, 768, 14, 14]
        patch_embeds = self.patch_embedding(pixel_values)
        
        # Step 2: Reshape from 2D grid to sequence
        # [B, 768, 14, 14] → [B, 768, 196]
        embeddings = patch_embeds.flatten(2)
        
        # Step 3: Transpose to get standard sequence format
        # [B, 768, 196] → [B, 196, 768]
        embeddings = embeddings.transpose(1, 2)
        
        # Step 4: Add positional embeddings
        # Each patch gets its position-specific vector added
        # Position 0: [768-dim vector], Position 1: [768-dim vector], etc.
        # This tells the model WHERE each patch came from in the original image
        embeddings = embeddings + self.position_embedding(self.position_ids)
        
        # Output: [Batch_Size, Num_Patches, Embed_Dim] = [B, 196, 768]
        return embeddings
    
class SiglipAttention(nn.Module):
    """
    Multi-headed Self-Attention mechanism.
    
    Core Idea: Let each patch "attend to" (look at) all other patches to understand context.
    
    Visual representation of attention:
    
    Query Patch (asking): "What should I focus on?"
          │
          ▼
    ┌─────────────────────────────────────────┐
    │  Compute similarity with all patches:   │
    │                                         │
    │  Key Patches (being asked):             │
    │  Patch 1: How relevant am I? → score_1  │
    │  Patch 2: How relevant am I? → score_2  │
    │  ...                                    │
    │ Patch 196: How relevant am I?→ score_196│
    └─────────────────────────────────────────┘
          │
          ▼ Softmax (convert to probabilities)
    ┌─────────────────────────────────────────┐
    │  Attention Weights (how much to focus): │
    │  [0.1, 0.05, 0.3, ..., 0.02]            │
    └─────────────────────────────────────────┘
          │
          ▼ Weighted sum
    ┌─────────────────────────────────────────┐
    │  Value Patches (content to aggregate):  │
    │  0.1 × Value_1 + 0.05 × Value_2 + ...   │
    └─────────────────────────────────────────┘
          │
          ▼
      Updated representation of the query patch
    
    Multi-head: We do this process multiple times (12 heads) in parallel,
                each head learning different types of relationships.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size        # 768
        self.num_heads = config.num_attention_heads # 12
        self.head_dim = self.embed_dim // self.num_heads  # 768 / 12 = 64
        self.scale = self.head_dim**-0.5           # 1/sqrt(64) ≈ 0.125 for scaled dot-product
        self.dropout = config.attention_dropout

        # Linear projections to create queries, keys, and values
        # Each takes [B, N, 768] and outputs [B, N, 768]
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)  # Keys: "I am this content"
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)  # Values: "Use this info"
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)  # Queries: "I'm looking for..."
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)  # Final output projection

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: [Batch_Size, Num_Patches, Embed_Dim] = [B, 196, 768]
            
        Returns:
            attn_output: [Batch_Size, Num_Patches, Embed_Dim] = [B, 196, 768]
            attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        """

        batch_size, seq_len, _ = hidden_states.size()  # B, 196, 768
        
        # Step 1: Project to queries, keys, values
        # All shapes: [B, 196, 768]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Step 2: Split into multiple heads
        # [B, 196, 768] → [B, 196, 12, 64] → [B, 12, 196, 64]
        # Now we have 12 separate attention mechanisms operating in parallel
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Step 3: Compute attention scores
        # Q @ K^T: [B, 12, 196, 64] @ [B, 12, 64, 196] → [B, 12, 196, 196]
        # Each element [i,j] represents: "How much should patch i attend to patch j?"
        # Scale by 1/sqrt(head_dim) to prevent large values
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # Step 4: Apply softmax to get attention probabilities
        # For each query (row), softmax over all keys (columns)
        # Result: Each row sums to 1.0, representing a probability distribution
        # [B, 12, 196, 196] - now values are in [0, 1] and each row sums to 1
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Step 5: Apply dropout (regularization during training)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Step 6: Apply attention weights to values
        # [B, 12, 196, 196] @ [B, 12, 196, 64] → [B, 12, 196, 64]
        # For each query position, we get a weighted combination of all value vectors
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        # Step 7: Concatenate heads back together
        # [B, 12, 196, 64] → [B, 196, 12, 64] → [B, 196, 768]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        
        # Step 8: Final linear projection
        # [B, 196, 768] → [B, 196, 768]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights
    
class SiglipMLP(nn.Module):
    """
    Feed-Forward Network (MLP) applied to each patch independently.
    
    Architecture:
    Input [768] → Linear [3072] → GELU activation → Linear [768] → Output [768]
    
    Purpose: Add non-linearity and increase model capacity
    - Expands to 4x size (768 → 3072)
    - Applies non-linear activation (GELU)
    - Projects back to original size (3072 → 768)
    
    Think of it as: "After attention gathered information, now transform it in complex ways"
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Expand: 768 → 3072 (4x expansion)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # Contract: 3072 → 768 (back to original)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [Batch_Size, Num_Patches, Embed_Dim] = [B, 196, 768]
            
        Returns:
            hidden_states: [Batch_Size, Num_Patches, Embed_Dim] = [B, 196, 768]
        """
        # Step 1: Expand and apply first linear layer
        # [B, 196, 768] → [B, 196, 3072]
        hidden_states = self.fc1(hidden_states)
        
        # Step 2: Apply GELU activation (smooth, non-linear function)
        # GELU(x) ≈ x * Φ(x), where Φ is the cumulative distribution function
        # It's like ReLU but smoother, helping with gradient flow
        # [B, 196, 3072] → [B, 196, 3072]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        
        # Step 3: Project back to original dimension
        # [B, 196, 3072] → [B, 196, 768]
        hidden_states = self.fc2(hidden_states)

        return hidden_states

class SiglipEncoderLayer(nn.Module):
    """
    A single Transformer Encoder Layer combining Self-Attention and MLP.
    
    Architecture (Pre-Norm variant):
    ┌─────────────────────────────────────┐
    │        Input [B, 196, 768]          │
    └────────────────┬────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         │      Layer Norm       │
         │                       │
         │    Self-Attention     |
         │                       │
         └───────────┬───────────┘
                     │ (Add residual)
                     ▼
         ┌───────────────────────┐
         │   [B, 196, 768]       │
         └───────────┬───────────┘
         ┌───────────┴───────────┐
         │                       │
         │      Layer Norm       │
         │                       │
         │          MLP          │
         │                       │
         └───────────┬───────────┘
                     │ (Add residual)
                     ▼
         ┌───────────────────────┐
         │  Output [B, 196, 768] │
         └───────────────────────┘
    
    Key Design Choices:
    - Pre-normalization (norm before attention/MLP, not after)
    - Residual connections (skip connections) around each sub-layer
    - These help with gradient flow in deep networks
    """
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        
        # Sub-layer 1: Multi-head Self-Attention
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        
        # Sub-layer 2: Feed-Forward MLP
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [Batch_Size, Num_Patches, Embed_Dim] = [B, 196, 768]
            
        Returns:
            hidden_states: [Batch_Size, Num_Patches, Embed_Dim] = [B, 196, 768]
        """
        # ===== First Sub-layer: Self-Attention =====
        # Save input for residual connection
        residual = hidden_states  # [B, 196, 768]
        
        # Normalize before attention (Pre-Norm)
        hidden_states = self.layer_norm1(hidden_states)  # [B, 196, 768]
        
        # Apply self-attention: patches interact with each other
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)  # [B, 196, 768]
        
        # Add residual connection (helps with gradient flow)
        # This allows the model to learn incremental changes
        hidden_states = residual + hidden_states  # [B, 196, 768]
        
        # ===== Second Sub-layer: MLP =====
        # Save for second residual connection
        residual = hidden_states  # [B, 196, 768]
        
        # Normalize before MLP (Pre-Norm)
        hidden_states = self.layer_norm2(hidden_states)  # [B, 196, 768]
        
        # Apply MLP: non-linear transformation of each patch independently
        hidden_states = self.mlp(hidden_states)  # [B, 196, 768]
        
        # Add second residual connection
        hidden_states = residual + hidden_states  # [B, 196, 768]
        
        return hidden_states
    
class SiglipEncoder(nn.Module):
    """
    Stack of Transformer Encoder Layers (default: 12 layers).
    
    Visual representation:
    ┌────────────────────┐
    │  Input Embeddings  │  [B, 196, 768]
    └─────────┬──────────┘
              │
    ┌─────────▼──────────┐
    │   Encoder Layer 1  │  Self-Attention + MLP
    └─────────┬──────────┘
              │
    ┌─────────▼──────────┐
    │   Encoder Layer 2  │  Self-Attention + MLP
    └─────────┬──────────┘
              │
          ... (10 more layers)
              │
    ┌─────────▼──────────┐
    │  Encoder Layer 12  │  Self-Attention + MLP
    └─────────┬──────────┘
              │
    ┌─────────▼──────────┐
    │  Output Features   │  [B, 196, 768]
    └────────────────────┘
    
    Each layer refines the representations, learning increasingly
    complex patterns and relationships between patches.
    """
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        # Create a list of num_hidden_layers encoder layers (default: 12)
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim] = [B, 196, 768]
            
        Returns:
            hidden_states: [Batch_Size, Num_Patches, Embed_Dim] = [B, 196, 768]
        """
        hidden_states = inputs_embeds

        # Pass through each encoder layer sequentially
        # Each layer takes [B, 196, 768] and outputs [B, 196, 768]
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states

class SiglipVisionTransformer(nn.Module):
    """
    Complete Vision Transformer architecture.
    
    Combines all components:
    1. Embeddings: Convert image to patch embeddings
    2. Encoder: Process patches through transformer layers
    3. Post Layer Norm: Final normalization
    
    Full Pipeline:
    Image [B,3,224,224] → Patches [B,196,768] → Encoder → Normalized Output [B,196,768]
    """
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        # Component 1: Convert images to embedded patches
        self.embeddings = SiglipVisionEmbeddings(config)
        
        # Component 2: Stack of transformer encoder layers
        self.encoder = SiglipEncoder(config)
        
        # Component 3: Final layer normalization
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [Batch_Size, Channels, Height, Width] = [B, 3, 224, 224]
            
        Returns:
            last_hidden_state: [Batch_Size, Num_Patches, Embed_Dim] = [B, 196, 768]
        """
        # Step 1: Convert image to embedded patches with positions
        # [B, 3, 224, 224] → [B, 196, 768]
        hidden_states = self.embeddings(pixel_values)

        # Step 2: Process through transformer encoder (12 layers)
        # Each patch attends to all other patches, building contextual understanding
        # [B, 196, 768] → [B, 196, 768]
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)

        # Step 3: Final normalization for stable outputs
        # [B, 196, 768] → [B, 196, 768]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state
    

class SiglipVisionModel(nn.Module):
    """
    Top-level wrapper for the SigLIP Vision Model.
    
    This is the main entry point for using the vision transformer.
    It wraps the SiglipVisionTransformer and provides a clean interface.
    
    Usage:
        config = SiglipVisionConfig()
        model = SiglipVisionModel(config)
        features = model(images)  # [B, 196, 768]
    
    The output features can be used for:
    - Image classification (by pooling and adding a classifier head)
    - Image-text matching (SigLIP's main use case)
    - Dense prediction tasks (using individual patch features)
    """

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        # The actual transformer model
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        """
        Args:
            pixel_values: [Batch_Size, Channels, Height, Width] = [B, 3, 224, 224]
            
        Returns:
            Patch-level features: [Batch_Size, Num_Patches, Embed_Dim] = [B, 196, 768]
            
        Each of the 196 output vectors represents the learned features for one
        16x16 patch of the input image, enriched with context from all other patches.
        """
        # [B, 3, 224, 224] → [B, 196, 768]
        return self.vision_model(pixel_values=pixel_values)