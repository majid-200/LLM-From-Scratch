import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

"""
PaliGemma: A Multimodal Vision-Language Model

Architecture Overview:
┌────────────────────────────────────────────────────────────────┐
│                        PaliGemma Model                         │
│                                                                │
│  ┌──────────────┐         ┌─────────────┐                      │
│  │   Vision     │         │   Linear    │                      │
│  │   Encoder    │────────>│  Projector  │                      │
│  │  (SigLIP)    │         │             │                      │
│  └──────────────┘         └──────┬──────┘                      │
│   Image [B,3,224,224]            │ Image Features              │
│         ↓                        │ [B,256,2048]                │
│   Features [B,256,768]           │                             │
│                                  ▼                             │
│                         ┌─────────────────┐                    │
│   Text Tokens           │  Merge Features │                    │
│   [B,Seq_Len] ─────────>│  Text + Images  │                    │
│                         └────────┬────────┘                    │
│                                  │                             │
│                                  ▼                             │
│                         ┌─────────────────┐                    │
│                         │  Gemma Language │                    │
│                         │     Model       │                    │
│                         │  (Decoder-only  │                    │
│                         │   Transformer)  │                    │
│                         └────────┬────────┘                    │
│                                  │                             │
│                                  ▼                             │
│                         ┌─────────────────┐                    │
│                         │  Output Logits  │                    │
│                         │  [B,Seq,Vocab]  │                    │
│                         └─────────────────┘                    │
└────────────────────────────────────────────────────────────────┘

Key Innovation: Merges visual and textual information into a unified sequence
that a language model can process to generate text about images.
"""

class GemmaConfig():
    """
    Configuration class for the Gemma language model.
    
    Gemma is Google's open-source language model (like GPT but decoder-only).
    This config defines the model architecture hyperparameters.
    """

    def __init__(
        self,
        vocab_size,                    # Size of vocabulary (e.g., 256,000 tokens)
        hidden_size,                   # Dimension of embeddings (e.g., 2048)
        intermediate_size,             # MLP hidden size (typically 4x hidden_size)
        num_hidden_layers,             # Number of transformer layers (e.g., 18)
        num_attention_heads,           # Number of query heads (e.g., 8)
        num_key_value_heads,           # Number of key/value heads for GQA (e.g., 1)
        head_dim=256,                  # Dimension per attention head
        max_position_embeddings=8192,  # Maximum sequence length
        rms_norm_eps=1e-6,            # Epsilon for RMS normalization
        rope_theta=10000.0,           # Base for rotary position embeddings
        attention_bias=False,          # Whether to use bias in attention projections
        attention_dropout=0.0,         # Dropout in attention (0.0 = no dropout)
        pad_token_id=None,            # ID for padding token
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class PaliGemmaConfig():
    """
    Configuration for the complete PaliGemma multimodal model.
    
    Combines:
    - Vision encoder config (SigLIP)
    - Language model config (Gemma)
    - Projection layer config (vision → language space)
    """

    def __init__(
        self,
        vision_config=None,           # SigLIP vision encoder configuration
        text_config=None,             # Gemma language model configuration
        ignore_index=-100,            # Label index to ignore in loss (for padding)
        image_token_index=256000,     # Token ID representing <image> token
        vocab_size=257152,            # Total vocabulary size (text + special tokens)
        projection_dim=2048,          # Dimension to project vision features to
        hidden_size=2048,             # Hidden size of language model
        pad_token_id=None,           # Padding token ID
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False  # PaliGemma is decoder-only
        self.pad_token_id = pad_token_id

        # Initialize vision and text configs
        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        # Calculate number of image tokens based on patch size
        # For 224x224 image with 16x16 patches: (224/16)² = 196 tokens
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim

class GemmaRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Simpler alternative to LayerNorm that only rescales (no mean centering).
    
    Standard LayerNorm: y = (x - mean) / std * weight + bias
    RMSNorm: y = x / rms(x) * weight
    
    Visual comparison:
    ┌─────────────────────────────────────────┐
    │  LayerNorm (more complex)               │
    │  1. Compute mean: μ = Σx / n            │
    │  2. Compute std: σ = sqrt(Σ(x-μ)² / n)  │
    │  3. Normalize: (x - μ) / σ              │
    │  4. Scale & shift: y * weight + bias    │
    └─────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────┐
    │  RMSNorm (simpler, faster)              │
    │  1. Compute RMS: rms = sqrt(Σx² / n)    │
    │  2. Normalize: x / rms                  │
    │  3. Scale: y * (1 + weight)             │
    └─────────────────────────────────────────┘
    
    Benefits: Faster, fewer operations, works well in practice
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim: Dimension of the input features
            eps: Small constant for numerical stability (avoid division by zero)
        """
        super().__init__()
        self.eps = eps
        # Learnable scale parameter (initialized to zeros)
        # Will be used as (1 + weight) for scaling
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        """
        Compute RMS normalization: x / sqrt(mean(x²) + eps)
        
        Args:
            x: Input tensor [..., dim]
            
        Returns:
            Normalized tensor with same shape
        """
        # x.pow(2): Square each element
        # .mean(-1, keepdim=True): Average over last dimension
        # torch.rsqrt: 1 / sqrt(x) - reciprocal square root (more efficient)
        # Final: x / sqrt(mean(x²) + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor [Batch_Size, Seq_Len, Hidden_Size]
            
        Returns:
            Normalized and scaled tensor [Batch_Size, Seq_Len, Hidden_Size]
        """
        # Normalize in float32 for numerical stability
        output = self._norm(x.float())
        
        # Scale by (1 + weight) - Gemma-specific choice
        # Note: Llama does x.to(float16) * w, but Gemma does (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        
        # Convert back to original dtype
        return output.type_as(x)
    
class GemmaMLP(nn.Module):
    """
    Gated MLP (Multi-Layer Perceptron) used in Gemma.
    
    Architecture:
    ┌────────────────────────────────────────────┐
    │           Input [Hidden_Size]              │
    └──────────────────┬─────────────────────────┘
                       │
            ┌──────────┴──────────┐
            │                     │
            ▼                     ▼
    ┌──────────────┐      ┌──────────────┐
    │  gate_proj   │      │   up_proj    │
    │ (Linear)     │      │  (Linear)    │
    └──────┬───────┘      └──────┬───────┘
           │                     │
           │ [Intermediate]      │ [Intermediate]
           │                     │
           ▼                     │
    ┌──────────────┐             │
    │    GELU      │             │
    │ (Activation) │             │
    └──────┬───────┘             │
           │                     │
           └──────────┬──────────┘
                      │ Element-wise multiply (gating)
                      ▼
              ┌──────────────┐
              │  down_proj   │
              │  (Linear)    │
              └──────┬───────┘
                     │
                     ▼
            ┌────────────────┐
            │ Output [Hidden]│
            └────────────────┘
    
    This is called "SwiGLU" variant:
    - gate_proj + GELU creates the "gate"
    - up_proj creates the "value"
    - Multiply gate * value (element-wise gating)
    - Project back down
    
    Why gating? Allows the network to control information flow dynamically
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size            # E.g., 2048
        self.intermediate_size = config.intermediate_size # E.g., 8192 (4x)
        
        # Three linear projections (no bias)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        """
        Apply gated MLP.
        
        Formula: down_proj(GELU(gate_proj(x)) * up_proj(x))
        
        Args:
            x: Input [Batch_Size, Seq_Len, Hidden_Size]
            
        Returns:
            Output [Batch_Size, Seq_Len, Hidden_Size]
        """
        # Equivalent to the expanded version:
        # y = self.gate_proj(x)  # [B, Seq, Hidden] → [B, Seq, Intermediate]
        # y = torch.gelu(y, approximate="tanh")  # Apply activation
        # j = self.up_proj(x)  # [B, Seq, Hidden] → [B, Seq, Intermediate]
        # z = y * j  # Element-wise gating
        # z = self.down_proj(z)  # [B, Seq, Intermediate] → [B, Seq, Hidden]
        
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))
    
class GemmaAttention(nn.Module):
    """
    Multi-Head Attention with Grouped Query Attention (GQA) and Rotary Position Embeddings.
    
    Key innovations:
    1. GQA: Fewer K/V heads than Q heads (memory efficient)
    2. RoPE: Rotary position embeddings (better than learned positions)
    3. KV Cache: Efficient autoregressive generation
    
    Architecture comparison:
    ┌────────────────────────────────────────────────────────┐
    │  Standard Multi-Head Attention (MHA)                   │
    │  ┌──────┐ ┌──────┐ ┌──────┐                            │
    │  │ Q1   │ │ K1   │ │ V1   │  Head 1                    │
    │  └──────┘ └──────┘ └──────┘                            │
    │  ┌──────┐ ┌──────┐ ┌──────┐                            │
    │  │ Q2   │ │ K2   │ │ V2   │  Head 2                    │
    │  └──────┘ └──────┘ └──────┘                            │
    │  ... 8 separate K,V heads                              │
    └────────────────────────────────────────────────────────┘
    
    ┌────────────────────────────────────────────────────────┐
    │  Grouped Query Attention (GQA) - Used by Gemma         │
    │  ┌──────┐                                              │
    │  │ Q1   │ ┐                                            │
    │  └──────┘ │                                            │
    │  ┌──────┐ │                                            │
    │  │ Q2   │ ├─ Share same K,V                            │
    │  └──────┘ │                                            │
    │  ...      │  ┌──────┐ ┌──────┐                         │
    │  ┌──────┐ │  │ K1   │ │ V1   │  Single KV head         │
    │  │ Q8   │ │  └──────┘ └──────┘                         │
    │  └──────┘ ┘                                            │
    │  Only 1 K,V head for 8 Q heads!                        │
    └────────────────────────────────────────────────────────┘
    """

    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size              # E.g., 2048
        self.num_heads = config.num_attention_heads        # E.g., 8 query heads
        self.head_dim = config.head_dim                    # E.g., 256
        self.num_key_value_heads = config.num_key_value_heads  # E.g., 1 KV head (GQA)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # E.g., 8/1 = 8
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True  # Causal attention (can't attend to future tokens)

        assert self.hidden_size % self.num_heads == 0

        # Linear projections for Q, K, V
        # Q has more heads than K,V (Grouped Query Attention)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        # Rotary position embeddings
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Perform multi-head attention with GQA and RoPE.
        
        Flow:
        1. Project to Q, K, V
        2. Split into multiple heads
        3. Apply rotary position embeddings
        4. Update/use KV cache (if generating)
        5. Repeat K,V to match Q heads (GQA)
        6. Compute attention scores
        7. Apply attention to values
        8. Concatenate heads and project
        
        Args:
            hidden_states: Input [Batch_Size, Seq_Len, Hidden_Size]
            attention_mask: Mask [Batch_Size, 1, Seq_Len_Q, Seq_Len_KV]
            position_ids: Positions [Batch_Size, Seq_Len]
            kv_cache: Optional KV cache for generation
            
        Returns:
            attn_output: Output [Batch_Size, Seq_Len, Hidden_Size]
            attn_weights: Attention weights [Batch_Size, Num_Heads, Seq_Len_Q, Seq_Len_KV]
        """
        bsz, q_len, _ = hidden_states.size()  # [Batch_Size, Seq_Len, Hidden_Size]
        
        # Step 1: Project to queries, keys, values
        # [B, Seq_Len, Hidden] → [B, Seq_Len, Num_Heads_Q * Head_Dim]
        query_states = self.q_proj(hidden_states)
        # [B, Seq_Len, Hidden] → [B, Seq_Len, Num_Heads_KV * Head_Dim]
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Step 2: Reshape to separate heads
        # [B, Seq, Num_Heads_Q * Head_Dim] → [B, Seq, Num_Heads_Q, Head_Dim] → [B, Num_Heads_Q, Seq, Head_Dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [B, Seq, Num_Heads_KV * Head_Dim] → [B, Num_Heads_KV, Seq, Head_Dim]
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Step 3: Apply rotary position embeddings
        # Get cos and sin for each position: [B, Seq_Len, Head_Dim]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        # Apply rotation to Q and K: rotates vectors based on position
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Step 4: Update KV cache (if generating)
        if kv_cache is not None:
            # Concatenate new keys/values with cached ones
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        # Step 5: Repeat K,V to match number of Q heads (GQA)
        # [B, 1 KV head, Seq, Head_Dim] → [B, 8 Q heads, Seq, Head_Dim]
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Step 6: Compute attention scores Q @ K^T / sqrt(head_dim)
        # [B, Num_Heads, Seq_Q, Head_Dim] @ [B, Num_Heads, Head_Dim, Seq_KV]
        # → [B, Num_Heads, Seq_Q, Seq_KV]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Step 7: Apply attention mask (for causal masking and padding)
        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        # Step 8: Apply softmax to get attention probabilities
        # [B, Num_Heads, Seq_Q, Seq_KV]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Step 9: Apply dropout (regularization during training)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        # Step 10: Apply attention to values
        # [B, Num_Heads, Seq_Q, Seq_KV] @ [B, Num_Heads, Seq_KV, Head_Dim]
        # → [B, Num_Heads, Seq_Q, Head_Dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        # Step 11: Rearrange: put sequence length back as second dimension
        # [B, Num_Heads, Seq, Head_Dim] → [B, Seq, Num_Heads, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # Step 12: Concatenate all heads
        # [B, Seq, Num_Heads, Head_Dim] → [B, Seq, Num_Heads * Head_Dim]
        attn_output = attn_output.view(bsz, q_len, -1)
        
        # Step 13: Final output projection
        # [B, Seq, Num_Heads * Head_Dim] → [B, Seq, Hidden_Size]
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

class GemmaDecoderLayer(nn.Module):
    """
    Single transformer decoder layer in Gemma.
    
    Architecture (Pre-Norm with RMSNorm):
    ┌─────────────────────────────────────┐
    │      Input [B, Seq, Hidden]         │
    └────────────────┬────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         │        RMSNorm        |
         │                       │
         │    Self-Attention     |
         |     (GQA + RoPE)      |
         │                       │
         └───────────┬───────────┘
                     │ (Add residual)
                     ▼
         ┌───────────────────────┐
         │   [B, Seq, Hidden]    │
         └───────────┬───────────┘
         ┌───────────┴───────────┐
         │                       │
         │        RMSNorm        |
         │                       │
         │       Gated MLP       |
         │                       │
         └───────────┬───────────┘
                     │ (Add residual)
                     ▼
         ┌───────────────────────┐
         │  Output [B, Seq, Hid] │
         └───────────────────────┘
    
    Key features:
    - Pre-normalization (RMSNorm before each sub-layer)
    - Residual connections around attention and MLP
    - Causal self-attention (can't see future tokens)
    """

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Sub-components
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
        self.mlp = GemmaMLP(config)
        
        # Normalization layers
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Process input through attention and MLP with residual connections.
        
        Args:
            hidden_states: Input [Batch_Size, Seq_Len, Hidden_Size]
            attention_mask: Attention mask [Batch_Size, 1, Seq_Len_Q, Seq_Len_KV]
            position_ids: Position indices [Batch_Size, Seq_Len]
            kv_cache: Optional KV cache
            
        Returns:
            hidden_states: Output [Batch_Size, Seq_Len, Hidden_Size]
        """
        # ===== First Sub-layer: Self-Attention =====
        residual = hidden_states  # Save for residual connection
        
        # Pre-normalization: normalize before attention
        # [B, Seq, Hidden] → [B, Seq, Hidden]
        hidden_states = self.input_layernorm(hidden_states)

        # Apply self-attention with GQA and RoPE
        # [B, Seq, Hidden] → [B, Seq, Hidden]
        hidden_states, _, = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        
        # Add residual connection
        # [B, Seq, Hidden]
        hidden_states = residual + hidden_states

        # ===== Second Sub-layer: MLP =====
        residual = hidden_states  # Save for second residual connection
        
        # Pre-normalization: normalize before MLP
        # [B, Seq, Hidden] → [B, Seq, Hidden]
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Apply gated MLP
        # [B, Seq, Hidden] → [B, Seq, Hidden]
        hidden_states = self.mlp(hidden_states)
        
        # Add second residual connection
        # [B, Seq, Hidden]
        hidden_states = residual + hidden_states

        return hidden_states

class GemmaModel(nn.Module):
    """
    Complete Gemma language model (decoder-only transformer).
    
    Architecture:
    ┌────────────────────────────────┐
    │  Token IDs [B, Seq_Len]        │
    └────────────┬───────────────────┘
                 │
                 ▼
    ┌────────────────────────────────┐
    │  Embedding Layer               │
    │  [B, Seq_Len, Hidden_Size]     │
    └────────────┬───────────────────┘
                 │
                 ▼ Scale by sqrt(Hidden_Size)
    ┌────────────────────────────────┐
    │  Decoder Layer 1               │
    │  (Attention + MLP)             │
    └────────────┬───────────────────┘
                 │
                 ▼
    ┌────────────────────────────────┐
    │  Decoder Layer 2               │
    │  (Attention + MLP)             │
    └────────────┬───────────────────┘
                 │
            ... (more layers)
                 │
                 ▼
    ┌────────────────────────────────┐
    │  Decoder Layer N               │
    │  (Attention + MLP)             │
    └────────────┬───────────────────┘
                 │
                 ▼
    ┌────────────────────────────────┐
    │  Final RMSNorm                 │
    └────────────┬───────────────────┘
                 │
                 ▼
    ┌────────────────────────────────┐
    │  Hidden States                 │
    │  [B, Seq_Len, Hidden_Size]     │
    └────────────────────────────────┘
    """

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Token embedding layer: converts token IDs to vectors
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Stack of decoder layers
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        # Final normalization
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        """Get the embedding layer (used by PaliGemma to merge text and image embeddings)"""
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        """
        Process embeddings through all decoder layers.
        
        Args:
            attention_mask: Attention mask [B, 1, Seq_Len_Q, Seq_Len_KV]
            position_ids: Position indices [B, Seq_Len]
            inputs_embeds: Input embeddings [B, Seq_Len, Hidden_Size]
            kv_cache: Optional KV cache for generation
            
        Returns:
            hidden_states: Output [B, Seq_Len, Hidden_Size]
        """
        # [B, Seq_Len, Hidden_Size]
        hidden_states = inputs_embeds
        
        # Scale embeddings by sqrt(hidden_size) - Gemma-specific normalization
        # Helps with training stability
        # [B, Seq_Len, Hidden_Size]
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        # Pass through all decoder layers sequentially
        for decoder_layer in self.layers:
            # [B, Seq_Len, Hidden_Size] → [B, Seq_Len, Hidden_Size]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        # Final normalization
        # [B, Seq_Len, Hidden_Size]
        hidden_states = self.norm(hidden_states)

        return hidden_states

class GemmaForCausalLM(nn.Module):
    """
    Gemma model with language modeling head for text generation.
    
    Adds a linear layer on top of GemmaModel to predict next tokens.
    
    Architecture:
    ┌────────────────────────────────┐
    │  GemmaModel                    │
    │  (All decoder layers)          │
    │  Output: [B, Seq, Hidden]      │
    └────────────┬───────────────────┘
                 │
                 ▼
    ┌────────────────────────────────┐
    │  Language Modeling Head        │
    │  Linear: Hidden → Vocab_Size   │
    │  Output: [B, Seq, Vocab_Size]  │
    └────────────┬───────────────────┘
                 │
                 ▼
    ┌────────────────────────────────┐
    │  Logits for each token         │
    │  [B, Seq_Len, Vocab_Size]      │
    │  Apply softmax to get probs    │
    └────────────────────────────────┘
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        
        # Language modeling head: projects hidden states to vocabulary logits
        # Note: Often tied with embedding weights (weight sharing)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        """Get embedding layer from the base model"""
        return self.model.embed_tokens
    
    def tie_weights(self):
        """
        Tie the weights of embedding and LM head (weight sharing).
        
        Benefits:
        - Reduces parameters (vocab_size * hidden_size fewer parameters)
        - Improves generalization
        - Standard practice in language models
        """
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        """
        Generate logits for next token prediction.
        
        Args:
            attention_mask: Attention mask
            position_ids: Position indices
            inputs_embeds: Input embeddings [B, Seq_Len, Hidden_Size]
            kv_cache: Optional KV cache
            
        Returns:
            Dictionary with:
            - logits: [B, Seq_Len, Vocab_Size]
            - kv_cache: Updated cache (if provided)
        """

        # Process through all decoder layers
        # input_embeds: [B, Seq_Len, Hidden_Size]
        # outputs: [B, Seq_Len, Hidden_Size]
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs
        
        # Project to vocabulary size
        # [B, Seq_Len, Hidden_Size] → [B, Seq_Len, Vocab_Size]
        logits = self.lm_head(hidden_states)
        logits = logits.float()  # Ensure float32 for numerical stability

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            # Return the updated cache for next generation step
            return_data["kv_cache"] = kv_cache

        return return_data

class PaliGemmaMultiModalProjector(nn.Module):
    """
    Projects vision features to language model's embedding space.
    
    Purpose: Bridge between vision and language modalities
    
    Visual representation:
    ┌───────────────────────────────────────────────────┐
    │  Vision Encoder Output                            │
    │  [Batch, 256 patches, 768 vision_dim]             │
    └────────────────────┬──────────────────────────────┘
                         │
                         ▼
    ┌───────────────────────────────────────────────────┐
    │  Linear Projection                                │
    │  768 → 2048                                       │
    └────────────────────┬──────────────────────────────┘
                         │
                         ▼
    ┌───────────────────────────────────────────────────┐
    │  Language Model Compatible                        │
    │  [Batch, 256 patches, 2048 language_dim]          │
    │  Ready to merge with text embeddings!             │
    └───────────────────────────────────────────────────┘
    
    Why needed?
    - Vision encoder outputs 768-dim features (SigLIP)
    - Language model expects 2048-dim embeddings (Gemma)
    - This projector aligns the dimensions
    """
    
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        # Linear projection with bias
        # vision_config.hidden_size (768) → vision_config.projection_dim (2048)
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        """
        Project vision features to language embedding space.
        
        Args:
            image_features: Vision encoder output [B, Num_Patches, Embed_Dim]
                           Typically [B, 256, 768] from SigLIP
            
        Returns:
            hidden_states: Projected features [B, Num_Patches, Projection_Dim]
                          Typically [B, 256, 2048] for Gemma
        """
        # [B, Num_Patches, 768] → [B, Num_Patches, 2048]
        hidden_states = self.linear(image_features)
        return hidden_states

class PaliGemmaForConditionalGeneration(nn.Module):
    """
    Complete PaliGemma model: Vision + Language for multimodal understanding.
    
    High-level architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    Input Processing                     │
    │  ┌────────────┐              ┌──────────────┐           │
    │  │   Image    │              │  Text Tokens │           │
    │  │ [B,3,224²] │              │  [B, Seq]    │           │
    │  └──────┬─────┘              └──────┬───────┘           │
    │         │                           │                   │
    │         ▼                           ▼                   │
    │  ┌────────────┐              ┌──────────────┐           │
    │  │  Vision    │              │  Text Embed  │           │
    │  │  Encoder   │              │  [B,Seq,2048]│           │
    │  │ [B,256,768]│              └──────┬───────┘           │
    │  └──────┬─────┘                     │                   │
    │         │                           │                   │
    │         ▼                           │                   │
    │  ┌────────────┐                     │                   │
    │  │ Projector  │                     │                   │
    │  │[B,256,2048]│                     │                   │
    │  └──────┬─────┘                     │                   │
    │         │                           │                   │
    │         └───────────┬───────────────┘                   │
    │                     │                                   │
    │                     ▼                                   │
    │         ┌───────────────────────┐                       │
    │         │  Merge Image + Text   │                       │
    │         │  Replace <image>      │                       │
    │         │  tokens with vision   │                       │
    │         │  features             │                       │
    │         └───────────┬───────────┘                       │
    │                     │                                   │
    │                     ▼                                   │
    │         ┌───────────────────────┐                       │
    │         │  Gemma Language Model │                       │
    │         │  Process unified      │                       │
    │         │  sequence             │                       │
    │         └───────────┬───────────┘                       │
    │                     │                                   │
    │                     ▼                                   │
    │         ┌───────────────────────┐                       │
    │         │  Output Logits        │                       │
    │         │  [B, Seq, Vocab]      │                       │
    │         └───────────────────────┘                       │
    └─────────────────────────────────────────────────────────┘
    
    Key idea: Convert images to a sequence of "visual tokens" that can be
    processed alongside text tokens by the language model.
    """

    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        
        # Component 1: Vision encoder (SigLIP)
        self.vision_tower = SiglipVisionModel(config.vision_config)
        
        # Component 2: Projection layer (vision → language space)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        
        self.vocab_size = config.vocab_size

        # Component 3: Language model (Gemma)
        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        """Tie embedding and LM head weights in the language model"""
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self, 
        image_features: torch.Tensor, 
        inputs_embeds: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        kv_cache: Optional[KVCache] = None
    ):
        """
        Merge text embeddings with image features into a unified sequence.
        
        This is the KEY function that makes PaliGemma work!
        
        Detailed process:
        ┌─────────────────────────────────────────────────────────────┐
        │  Step 1: Input tokens with special image tokens             │
        │  input_ids: [<img>, <img>, ..., <img>, <bos>, "cat", ...]   │
        │             └────────┬────────┘                             │
        │                 256 image tokens                            │
        └─────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────────────────┐
        │  Step 2: Create masks to identify token types               │
        │  text_mask:  [False, False, ..., True, True, True, ...]     │
        │  image_mask: [True, True, ..., False, False, False, ...]    │
        │  pad_mask:   [False, False, ..., False, False, False, ...]  │
        └─────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────────────────┐
        │  Step 3: Build final embedding sequence                     │
        │  - For text tokens: Use text embeddings                     │
        │  - For image tokens: Use vision features                    │
        │  - For padding: Use zeros                                   │
        │                                                             │
        │  Result: [vision_feat, vision_feat, ..., text_emb, ...]     │
        │          └──────────┬──────────┘      └─────┬─────┘         │
        │              256 visual tokens          Text tokens         │
        └─────────────────────────────────────────────────────────────┘
        
        Args:
            image_features: Projected vision features [B, Num_Patches, Hidden_Size]
            inputs_embeds: Text token embeddings [B, Seq_Len, Hidden_Size]
            input_ids: Token IDs [B, Seq_Len]
            attention_mask: Original attention mask [B, Seq_Len]
            kv_cache: Optional KV cache for generation
            
        Returns:
            final_embedding: Merged embeddings [B, Seq_Len, Hidden_Size]
            causal_mask: Attention mask [B, 1, Seq_Len_Q, Seq_Len_KV]
            position_ids: Position indices [B, Seq_Len]
        """
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        
        # Scale image features by 1/sqrt(hidden_size) - Gemma-specific normalization
        # This matches the scaling applied to text embeddings in GemmaModel
        # [B, Num_Patches, Hidden_Size]
        scaled_image_features = image_features / (self.config.hidden_size**0.5)
    
        # Create a zero tensor to hold the final merged embeddings
        # [B, Seq_Len, Hidden_Size]
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        
        # Create boolean masks to identify different token types
        # text_mask: True for text tokens (not image tokens, not padding)
        # [B, Seq_Len]
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        
        # image_mask: True for image tokens
        # [B, Seq_Len]
        image_mask = input_ids == self.config.image_token_index
        
        # pad_mask: True for padding tokens
        # [B, Seq_Len]
        pad_mask = input_ids == self.pad_token_id

        # Expand masks to match embedding dimension for broadcasting
        # We need this because torch.where operates element-wise
        # [B, Seq_Len] → [B, Seq_Len, Hidden_Size]
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Step 1: Add text embeddings where text_mask is True
        # torch.where(condition, value_if_true, value_if_false)
        # [B, Seq_Len, Hidden_Size]
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        
        # Step 2: Insert image embeddings where image_mask is True
        # We can't use torch.where here because scaled_image_features has a different
        # sequence length (256 patches) than final_embedding (full sequence)
        # masked_scatter places values from scaled_image_features wherever mask is True
        # [B, Seq_Len, Hidden_Size]
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        
        # Step 3: Zero out padding tokens
        # [B, Seq_Len, Hidden_Size]
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        #### CREATE THE ATTENTION MASK ####
        # Attention mask prevents tokens from attending to certain positions
        # - Causal mask: Can't attend to future tokens
        # - Padding mask: Can't attend to padding tokens

        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min  # Large negative value for masking
        q_len = inputs_embeds.shape[1]
    
        if kv_cache is None or kv_cache.num_items() == 0:
            # Prefill phase: Processing the initial prompt
            # No masking needed in this implementation (assumes no padding)
            # Shape: [B, Q_Len, Q_Len]
            # All zeros means "attend to everything" (will be added to attention scores)
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Generation phase: Generating one token at a time
            # Query is only the new token (q_len = 1)
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            
            # New token can attend to all previous tokens
            # Shape: [B, 1, KV_Len]
            # All zeros means "attend to everything"
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # Add head dimension for multi-head attention
        # [B, Q_Len, KV_Len] → [B, 1, Q_Len, KV_Len]
        # The '1' will broadcast across all attention heads
        causal_mask = causal_mask.unsqueeze(1)

        #### CREATE POSITION IDs ####
        # Position IDs tell RoPE what position each token is at
        
        if kv_cache is not None and kv_cache.num_items() > 0:
            # Generation phase: New token's position is just after cached tokens
            # cumsum on attention_mask gives the position of each non-padding token
            # We only need the last position (for the new token)
            # [B, Seq_Len] → [B, 1]
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Prefill phase: Create positions for all tokens
            # cumsum creates: [1, 2, 3, 4, ...] for non-padding tokens
            # For padding tokens (attention_mask == 0), use position 1
            # [B, Seq_Len]
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        """
        Process images and text through PaliGemma to generate output.
        
        Complete forward pass:
        1. Embed text tokens
        2. Encode image through vision tower
        3. Project image features to language space
        4. Merge image and text embeddings
        5. Process through language model
        6. Return logits for next token prediction
        
        Example flow with actual shapes:
        ┌─────────────────────────────────────────────────────────┐
        │ Input:                                                  │
        │ - pixel_values: [1, 3, 224, 224] (one image)            │
        │ - input_ids: [1, 300] (256 <image> + text tokens)       │
        └─────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────────────┐
        │ Text embeddings: [1, 300, 2048]                         │
        │ Vision features: [1, 256, 768]                          │
        └─────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────────────┐
        │ Projected vision: [1, 256, 2048]                        │
        └─────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────────────┐
        │ Merged: [1, 300, 2048]                                  │
        │ (first 256 positions have vision features)              │
        └─────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────────────┐
        │ Language model processing...                            │
        └─────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────────────┐
        │ Output logits: [1, 300, 256128]                         │
        │ (probabilities for next token at each position)         │
        └─────────────────────────────────────────────────────────┘
        
        Args:
            input_ids: Token IDs [B, Seq_Len] (includes <image> tokens)
            pixel_values: Images [B, Channels, Height, Width]
            attention_mask: Attention mask [B, Seq_Len]
            kv_cache: Optional KV cache for generation
            
        Returns:
            Dictionary with:
            - logits: [B, Seq_Len, Vocab_Size]
            - kv_cache: Updated cache (if provided)
        """

        # Sanity check: This implementation assumes no padding
        # (All tokens should be attended to)
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # Step 1: Extract text embeddings from token IDs
        # [B, Seq_Len] → [B, Seq_Len, Hidden_Size]
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Step 2: Process image through vision encoder
        # [B, Channels, Height, Width] → [B, Num_Patches, Embed_Dim]
        # Example: [1, 3, 224, 224] → [1, 256, 768]
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        
        # Step 3: Project vision features to language model's embedding space
        # [B, Num_Patches, Embed_Dim] → [B, Num_Patches, Hidden_Size]
        # Example: [1, 256, 768] → [1, 256, 2048]
        image_features = self.multi_modal_projector(selected_image_feature)

        # Step 4: Merge image and text embeddings
        # - Replace <image> token embeddings with actual image features
        # - Create appropriate attention mask
        # - Generate position IDs
        # Returns:
        # - inputs_embeds: [B, Seq_Len, Hidden_Size] - merged embeddings
        # - attention_mask: [B, 1, Seq_Len_Q, Seq_Len_KV] - causal mask
        # - position_ids: [B, Seq_Len] - position indices for RoPE
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, attention_mask, kv_cache
        )
        
        # Step 5: Process merged embeddings through language model
        # The language model sees a unified sequence of:
        # [vision_feat_1, vision_feat_2, ..., vision_feat_256, text_emb_1, text_emb_2, ...]
        # It processes this just like it would process pure text!
        # Returns dictionary with 'logits' and optionally 'kv_cache'
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs


"""
Summary of PaliGemma Architecture:

1. **Vision Encoder (SigLIP)**:
   - Takes image [B, 3, 224, 224]
   - Outputs patch features [B, 256, 768]
   - Each of 256 patches represents a 16x16 region

2. **Multimodal Projector**:
   - Linear layer: 768 → 2048
   - Aligns vision features with language embedding space

3. **Embedding Merger**:
   - Text tokens → embeddings [B, Seq_Len, 2048]
   - Replace <image> tokens with projected vision features
   - Creates unified sequence for language model

4. **Language Model (Gemma)**:
   - Decoder-only transformer (like GPT)
   - Key features:
     * Grouped Query Attention (memory efficient)
     * Rotary Position Embeddings (better position encoding)
     * RMSNorm (simpler normalization)
     * Gated MLP (SwiGLU activation)
   - Processes merged vision+text sequence
   - Outputs logits for next token prediction

5. **Generation**:
   - Uses KV cache for efficient autoregressive decoding
   - Can describe images, answer questions, etc.

Key Innovation:
Images become "visual tokens" that the language model can process
alongside text, enabling true multimodal understanding!
"""