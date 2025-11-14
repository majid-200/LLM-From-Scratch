from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os

"""
Utility Functions for Loading PaliGemma Models

Purpose: Load pretrained models from HuggingFace format into our custom implementation

HuggingFace Model Directory Structure:
┌────────────────────────────────────────────────────────┐
│  model_directory/                                      │
│  ├── config.json              ← Model configuration    │
│  ├── tokenizer.json           ← Tokenizer vocab/rules  │
│  ├── tokenizer_config.json   ← Tokenizer settings      │
│  ├── special_tokens_map.json ← Special token mappings  │
│  ├── model-00001-of-00002.safetensors ← Weights part 1 │
│  └── model-00002-of-00002.safetensors ← Weights part 2 │
└────────────────────────────────────────────────────────┘

Why SafeTensors format?
- Safer than pickle (no arbitrary code execution)
- Faster to load (memory-mapped)
- Better for large models (can load parts selectively)
- Becoming the standard for model weights
"""


def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    """
    Load a PaliGemma model from HuggingFace format.
    
    Complete loading process:
    ┌────────────────────────────────────────────────────────┐
    │  Step 1: Load Tokenizer                                │
    │  - Reads vocabulary (256k+ tokens)                     │
    │  - Loads special tokens (<bos>, <eos>, <image>, etc.)  │
    │  - Sets padding side to "right"                        │
    └────────────────┬───────────────────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────────────────────┐
    │  Step 2: Find Weight Files                             │
    │  - Scan directory for *.safetensors files              │
    │  - May be split across multiple files for large models │
    │    Example: model-00001-of-00003.safetensors           │
    │             model-00002-of-00003.safetensors           │
    │             model-00003-of-00003.safetensors           │
    └────────────────┬───────────────────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────────────────────┐
    │  Step 3: Load All Weight Tensors                       │
    │  - Open each safetensors file                          │
    │  - Extract tensor by name                              │
    │  - Merge into single dictionary                        │
    │    {                                                   │
    │      "vision_tower.embeddings.patch_embedding.weight", │
    │      "language_model.model.layers.0.self_attn.q_proj", │
    │      ...                                               │
    │    }                                                   │
    └────────────────┬───────────────────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────────────────────┐
    │  Step 4: Load Configuration                            │
    │  - Read config.json                                    │
    │  - Parse into PaliGemmaConfig object                   │
    │  - Contains architecture hyperparameters               │
    └────────────────┬───────────────────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────────────────────┐
    │  Step 5: Initialize Model                              │
    │  - Create model with random weights                    │
    │  - Move to device (CPU/GPU)                            │
    └────────────────┬───────────────────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────────────────────┐
    │  Step 6: Load Pretrained Weights                       │
    │  - Match tensor names to model parameters              │
    │  - Copy pretrained values into model                   │
    │  - strict=False: OK if some tensors don't match        │
    └────────────────┬───────────────────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────────────────────┐
    │  Step 7: Tie Weights                                   │
    │  - Share embedding and LM head weights                 │
    │  - Reduces parameters, improves performance            │
    └────────────────┬───────────────────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────────────────────┐
    │  Return: (model, tokenizer)                            │
    │  - Model ready for inference                           │
    │  - Tokenizer ready to process text                     │
    └────────────────────────────────────────────────────────┘
    
    Args:
        model_path: Path to model directory (local) or HuggingFace model ID
                   Examples: 
                   - "./models/paligemma-3b"
                   - "google/paligemma-3b-pt-224"
        device: Target device ("cpu", "cuda", "mps")
        
    Returns:
        Tuple of (model, tokenizer) ready for inference
    """
    
    # ===== STEP 1: LOAD TOKENIZER =====
    """
    The tokenizer converts text to token IDs and vice versa.
    
    Example tokenization:
    "Hello world" → [15496, 1917] → [Hello, world]
    
    Special tokens in PaliGemma:
    - <bos>: Beginning of sequence (marks start)
    - <eos>: End of sequence (marks completion)
    - <image>: Placeholder for image patches (256 of these)
    - <loc####>: Location tokens for bounding boxes (1024 total)
    - <seg###>: Segmentation tokens (128 total)
    
    Why padding_side="right"?
    - Padding is added to the RIGHT of sequences
    - Example: [token1, token2, <pad>, <pad>]
    - Important for batch processing with variable lengths
    - Left padding would shift position embeddings incorrectly
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    
    # Verify padding side is correct (critical for proper model behavior)
    assert tokenizer.padding_side == "right", "Padding side must be 'right' for PaliGemma"

    # ===== STEP 2: FIND WEIGHT FILES =====
    """
    Large models are often split across multiple files for easier downloading/storage.
    
    Example for a 3B parameter model:
    - model-00001-of-00002.safetensors (1.5B parameters, ~6GB)
    - model-00002-of-00002.safetensors (1.5B parameters, ~6GB)
    
    glob.glob finds all files matching the pattern *.safetensors
    """
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    
    # Debug: Show found files
    # print(f"Found {len(safetensors_files)} safetensors files:")
    # for f in safetensors_files:
    #     print(f"  - {os.path.basename(f)}")

    # ===== STEP 3: LOAD ALL TENSORS =====
    """
    SafeTensors format:
    - Binary format optimized for ML model weights
    - Memory-mapped (efficient for large files)
    - Safe (no code execution risk like pickle)
    
    Process:
    1. Open each safetensors file
    2. Iterate through tensor names in file
    3. Load each tensor into memory
    4. Store in unified dictionary
    
    Tensor naming convention (PyTorch state_dict format):
    - "vision_tower.embeddings.patch_embedding.weight" → Conv2d weights
    - "language_model.model.layers.0.self_attn.q_proj.weight" → Layer 0 query projection
    - "language_model.model.embed_tokens.weight" → Token embeddings
    
    Example tensor shapes:
    - vision_tower.embeddings.patch_embedding.weight: [768, 3, 16, 16]
      → 768 output channels, 3 input channels (RGB), 16×16 kernel
    - language_model.model.embed_tokens.weight: [257152, 2048]
      → 257k vocab size, 2048 embedding dimension
    """
    tensors = {}
    
    for safetensors_file in safetensors_files:
        # Open safetensors file
        # framework="pt" means PyTorch format
        # device="cpu" loads tensors to CPU first (then moved to target device)
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            # Iterate through all tensors in this file
            for key in f.keys():
                # Load tensor and store in dictionary
                # key example: "vision_tower.encoder.layers.0.self_attn.q_proj.weight"
                # f.get_tensor(key) returns the actual tensor data
                tensors[key] = f.get_tensor(key)
    
    # At this point, tensors dictionary contains ALL model weights
    # print(f"Loaded {len(tensors)} tensors")
    # Total memory: Sum of all tensor sizes (can be several GB!)

    # ===== STEP 4: LOAD CONFIGURATION =====
    """
    config.json contains model architecture specifications.
    
    Example config.json structure:
    {
      "vision_config": {
        "hidden_size": 768,
        "image_size": 224,
        "patch_size": 16,
        "num_hidden_layers": 12,
        ...
      },
      "text_config": {
        "hidden_size": 2048,
        "num_hidden_layers": 18,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,  # GQA!
        ...
      },
      "vocab_size": 257152,
      "projection_dim": 2048,
      ...
    }
    
    This config is used to construct the model architecture.
    Weights must match this architecture or loading will fail!
    """
    with open(os.path.join(model_path, "config.json"), "r") as f:
        # Load JSON file
        model_config_file = json.load(f)
        
        # Create PaliGemmaConfig object from dictionary
        # ** unpacks dictionary as keyword arguments
        # PaliGemmaConfig(**{"vocab_size": 257152, ...})
        # becomes PaliGemmaConfig(vocab_size=257152, ...)
        config = PaliGemmaConfig(**model_config_file)

    # ===== STEP 5: INITIALIZE MODEL =====
    """
    Create model architecture with RANDOM weights initially.
    
    This creates all the layers:
    - Vision encoder (SigLIP): 12 transformer layers
    - Multimodal projector: Linear layer
    - Language model (Gemma): 18 transformer layers
    
    All weights are randomly initialized at this point!
    We'll load the pretrained weights in the next step.
    """
    model = PaliGemmaForConditionalGeneration(config).to(device)
    
    # Model is now on the target device (CPU/GPU)
    # But weights are still random - not useful yet!

    # ===== STEP 6: LOAD PRETRAINED WEIGHTS =====
    """
    Load the pretrained weights into our model.
    
    Process:
    1. For each parameter in model (e.g., "vision_tower.embeddings.patch_embedding.weight")
    2. Look for matching key in tensors dictionary
    3. Copy the pretrained tensor into the model parameter
    4. Verify shapes match
    
    strict=False means:
    - OK if tensors dict has extra keys (not used by our model)
    - OK if model has parameters not in tensors (keeps random init)
    - Useful for partial loading or when implementations differ slightly
    
    Example weight loading:
    ┌─────────────────────────────────────────────────────────┐
    │  Model Parameter (random):                              │
    │  model.vision_tower.embeddings.patch_embedding.weight   │
    │  Shape: [768, 3, 16, 16]                                │
    └────────────────────┬────────────────────────────────────┘
                         │ Load from tensors dict
                         ▼
    ┌─────────────────────────────────────────────────────────┐
    │  Pretrained Tensor:                                     │
    │  tensors["vision_tower.embeddings.patch_embedding..."]  │
    │  Shape: [768, 3, 16, 16] ✓ Match!                       │
    └─────────────────────────────────────────────────────────┘
                         │ Copy values
                         ▼
    ┌─────────────────────────────────────────────────────────┐
    │  Model Parameter (pretrained):                          │
    │  model.vision_tower.embeddings.patch_embedding.weight   │
    │  Now contains learned values from training!             │
    └─────────────────────────────────────────────────────────┘
    
    After this step, model has pretrained weights and is ready to use!
    """
    model.load_state_dict(tensors, strict=False)
    
    # Debug: Check which parameters were loaded
    # loaded = set(tensors.keys())
    # model_params = set(model.state_dict().keys())
    # missing = model_params - loaded
    # if missing:
    #     print(f"Parameters not loaded (will use random init): {missing}")

    # ===== STEP 7: TIE WEIGHTS =====
    """
    Weight tying: Share parameters between embedding layer and LM head.
    
    Concept:
    ┌──────────────────────────────────────────────────────┐
    │  Token Embedding Layer                               │
    │  Converts token IDs → vectors                        │
    │  Weight shape: [Vocab_Size, Hidden_Size]             │
    │                [257152, 2048]                        │
    └──────────────────────────────────────────────────────┘
                         ⇅ (Share weights!)
    ┌──────────────────────────────────────────────────────┐
    │  Language Model Head                                 │
    │  Converts vectors → token logits                     │
    │  Weight shape: [Hidden_Size, Vocab_Size]             │
    │                [2048, 257152] (transpose of above)   │
    └──────────────────────────────────────────────────────┘
    
    Why tie weights?
    1. Parameter efficiency: Save ~500M parameters!
       (257152 × 2048 = 526 million parameters)
    
    2. Improved generalization: Input and output use same semantic space
       - If "cat" embeds to [0.1, 0.5, ...], predicting "cat" uses same values
       - Ensures consistency between understanding and generation
    
    3. Standard practice: Used in BERT, GPT, T5, and most modern LMs
    
    Implementation:
    - embedding.weight and lm_head.weight point to SAME tensor
    - Changes to one automatically affect the other
    - In PyTorch: lm_head.weight = embedding.weight (parameter sharing)
    """
    model.tie_weights()
    
    # After weight tying:
    # assert model.language_model.lm_head.weight is model.language_model.model.embed_tokens.weight
    # ^ These are the SAME tensor in memory

    # ===== RETURN LOADED MODEL AND TOKENIZER =====
    """
    Model is now ready for inference:
    - All pretrained weights loaded ✓
    - On correct device (CPU/GPU) ✓
    - Weights tied ✓
    - Tokenizer ready ✓
    
    Usage:
    model, tokenizer = load_hf_model("google/paligemma-3b-pt-224", "cuda")
    # Now ready to process images and generate text!
    """
    return (model, tokenizer)


"""
Additional Notes:

1. **Model Size vs. File Size**:
   - 3B parameters ≈ 12 GB (float32)
   - Typically stored as float16/bfloat16 → 6 GB
   - SafeTensors files match this size

2. **Memory Requirements**:
   During loading:
   - Tensors loaded to CPU first (6+ GB)
   - Model initialized on target device (6+ GB)
   - load_state_dict copies tensors (peak: ~12 GB)
   - After loading, only model on device (6 GB)
   
   For inference on GPU:
   - Model: ~6 GB
   - Activations: 1-2 GB
   - KV cache: Grows with sequence length
   - Total: 8-10 GB minimum for 3B model

3. **Why strict=False?**:
   - HuggingFace may have extra keys (e.g., for specific tasks)
   - Our implementation may differ slightly in naming
   - Missing keys will use random initialization (usually not important)
   - Production code might want strict=True and handle mismatches explicitly

4. **Alternative Loading Methods**:
   Could also use HuggingFace's built-in loading:
   ```python
   from transformers import PaliGemmaForConditionalGeneration
   model = PaliGemmaForConditionalGeneration.from_pretrained(model_path)
   ```
   But we use custom loading to work with our own implementation!

5. **Model Variants**:
   - paligemma-3b-pt-224: 224×224 images, pretrained
   - paligemma-3b-mix-224: 224×224, fine-tuned on multiple tasks
   - paligemma-3b-pt-448: 448×448 images (higher resolution)
   - paligemma-3b-pt-896: 896×896 images (very high resolution)
"""
