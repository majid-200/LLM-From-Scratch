from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

"""
PaliGemma Processor - Prepares images and text for a multimodal language model

Overall Purpose:
This processor takes raw images and text prompts and converts them into the format
that PaliGemma (a vision-language model) expects. It handles:
1. Image preprocessing (resize, normalize, etc.)
2. Text tokenization
3. Special token insertion (image tokens, BOS, newlines)

High-Level Flow:
┌─────────────────┐          ┌─────────────────┐
│  Input Image    │          │  Input Text     │
│  (PIL Image)    │          │  "describe"     │
└────────┬────────┘          └────────┬────────┘
         │                            │
         ▼                            ▼
┌─────────────────┐          ┌─────────────────┐
│ Image Processing│          │ Add Image Tokens│
│ - Resize        │          │ <img><img>...   │
│ - Normalize     │          │ <bos>describe\n │
│ - Rescale       │          └────────┬────────┘
└────────┬────────┘                   │
         │                            ▼
         │                   ┌─────────────────┐
         │                   │   Tokenization  │
         │                   │  [101,234,...]  │
         │                   └────────┬────────┘
         │                            │
         └────────────┬───────────────┘
                      ▼
         ┌────────────────────────┐
         │  Combined Output Dict  │
         │  - pixel_values        │
         │  - input_ids           │
         │  - attention_mask      │
         └────────────────────────┘
"""

# Standard ImageNet normalization values
# These center the pixel values around 0 with std of 1
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]  # For R, G, B channels
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]   # For R, G, B channels

def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    """
    Prepends image tokens to the text prompt in PaliGemma's expected format.
    
    PaliGemma's Input Format:
    ┌────────────────────────────────────────────────────────────┐
    │  <image><image>...<image><bos>User's text prompt\n         │
    │  └─────┬─────┘            └─┬─┘└──────┬──────┘ └┬┘         │
    │     256 tokens          Start    User      Newline         │
    │   (one per patch)       token    input    (required!)      │
    └────────────────────────────────────────────────────────────┘
    
    Why this format?
    - Image tokens act as "placeholders" where vision features will be inserted
    - BOS (Beginning Of Sequence) marks the start of text
    - Newline \n is CRITICAL - model was trained with it, ensures consistency
    
    Example:
    Input:  prefix_prompt = "describe this image"
            bos_token = "<bos>"
            image_seq_len = 256
            image_token = "<image>"
    
    Output: "<image><image>...(256 times)...<image><bos>describe this image\n"
    
    Args:
        prefix_prompt: User's text input (e.g., "describe this image")
        bos_token: Beginning of sequence token (e.g., "<bos>")
        image_seq_len: Number of image tokens to prepend (typically 256 for 16x16 patches)
        image_token: Special token representing one image patch (e.g., "<image>")
        
    Returns:
        Formatted string with image tokens + BOS + prompt + newline
    """
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    #   The input text is tokenized normally.
    #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
    #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    #   The tokenized text is also prefixed with a fixed number of <image> tokens.
    # NOTE: from the paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
    #       ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    
    # String multiplication: "<image>" * 256 creates "<image><image><image>..."
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"

def rescale(
    image: np.ndarray, scale: float, dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Rescales pixel values by a multiplicative factor.
    
    Common use: Convert uint8 pixel values [0, 255] to float [0.0, 1.0]
    
    Visual example:
    ┌──────────────────┐       ┌──────────────────┐
    │  Original Image  │       │  Rescaled Image  │
    │                  │       │                  │
    │  Pixel values:   │ ×1/255│  Pixel values:   │
    │  [0, 127, 255]   │──────>│  [0.0, 0.5, 1.0] │
    │                  │       │                  │
    │  dtype: uint8    │       │  dtype: float32  │
    └──────────────────┘       └──────────────────┘
    
    Args:
        image: Input image as numpy array (typically uint8 with values 0-255)
        scale: Multiplicative factor (e.g., 1/255.0 = 0.00392...)
        dtype: Output data type (default: float32)
        
    Returns:
        Rescaled image with new dtype
    """
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image


def resize(
    image: Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
) -> np.ndarray:
    """
    Resizes an image to the specified dimensions.
    
    Visual example (resizing to 224x224):
    ┌─────────────────┐         ┌─────────────────┐
    │  Original       │         │  Resized        │
    │  1024 x 768     │         │  224 x 224      │
    │                 │         │                 │
    │   ████████      │ resize  │    ████         │
    │   ████████      │────────>│    ████         │
    │   ████████      │         │    ████         │
    │                 │         │                 │
    └─────────────────┘         └─────────────────┘
    
    Why resize?
    - Neural networks expect fixed-size inputs
    - PaliGemma uses 224x224 or other standard sizes
    - Resampling methods (like BICUBIC) smooth the resizing
    
    Args:
        image: PIL Image object
        size: Target (height, width) tuple
        resample: Interpolation method (e.g., BICUBIC for smooth resizing)
        reducing_gap: Optimization for downscaling large images
        
    Returns:
        Resized PIL Image
    """
    height, width = size
    # PIL's resize expects (width, height), not (height, width)!
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image


def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    """
    Normalizes image to have specified mean and standard deviation.
    
    Formula: normalized = (image - mean) / std
    
    Visual intuition:
    ┌──────────────────┐        ┌──────────────────┐
    │  Before          │        │  After           │
    │  Normalize       │        │  Normalize       │
    │                  │        │                  │
    │  Range: [0, 1]   │ (x-μ)/σ│  Range: [-1, 1]  │
    │  Mean: 0.5       │──────> │  Mean: 0.0       │
    │  Std: ~0.3       │        │  Std: 1.0        │
    └──────────────────┘        └──────────────────┘
    
    Why normalize?
    - Centers data around 0 (makes optimization easier)
    - Standardizes scale (helps gradient flow)
    - Matches training data distribution (CRITICAL for good performance)
    
    Example with ImageNet values:
    - Input pixel: [0.6, 0.3, 0.8] (RGB after rescaling to [0,1])
    - Mean: [0.5, 0.5, 0.5]
    - Std: [0.5, 0.5, 0.5]
    - Output: [(0.6-0.5)/0.5, (0.3-0.5)/0.5, (0.8-0.5)/0.5]
    -       = [0.2, -0.4, 0.6]
    
    Args:
        image: Input image as numpy array (typically shape [H, W, C])
        mean: Mean value(s) to subtract (one per channel or single value)
        std: Standard deviation(s) to divide by (one per channel or single value)
        
    Returns:
        Normalized image with mean≈0 and std≈1
    """
    # Convert mean and std to numpy arrays matching image dtype
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    # Apply normalization: (x - μ) / σ
    image = (image - mean) / std
    return image

def process_images(
    images: List[Image.Image],
    size: Dict[str, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    """
    Complete image preprocessing pipeline for PaliGemma.
    
    Pipeline visualization:
    ┌─────────────────┐
    │  PIL Image      │  Original image (any size, uint8)
    │  [H, W, 3]      │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  1. RESIZE      │  Resize to model's expected size (e.g., 224x224)
    │  [224, 224, 3]  │  Uses BICUBIC interpolation for quality
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  2. TO NUMPY    │  Convert PIL → numpy array
    │  dtype: uint8   │  Values still [0, 255]
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  3. RESCALE     │  Scale to [0, 1] range
    │  × (1/255)      │  [0, 255] → [0.0, 1.0]
    │  dtype: float32 │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  4. NORMALIZE   │  Standardize: (x - mean) / std
    │  μ=0.5, σ=0.5   │  [0, 1] → [-1, 1]
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  5. TRANSPOSE   │  Rearrange dimensions for PyTorch
    │  [224,224,3]    │  [H, W, C] → [C, H, W]
    │      ↓          │  Channels-first format
    │  [3,224,224]    │
    └─────────────────┘
    
    Args:
        images: List of PIL Image objects
        size: Target size as [height, width]
        resample: Resampling method (typically BICUBIC)
        rescale_factor: Scale factor (typically 1/255.0)
        image_mean: Mean for normalization (typically [0.5, 0.5, 0.5])
        image_std: Std for normalization (typically [0.5, 0.5, 0.5])
        
    Returns:
        List of preprocessed images as numpy arrays [C, H, W]
    """
    height, width = size[0], size[1]
    
    # Step 1: Resize all images to the target size
    # PIL Images → PIL Images (resized)
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]
    
    # Step 2: Convert each PIL image to a numpy array
    # PIL Images → numpy arrays [H, W, 3] with dtype uint8
    images = [np.array(image) for image in images]
    
    # Step 3: Rescale pixel values to be in the range [0, 1]
    # [0, 255] → [0.0, 1.0], dtype: uint8 → float32
    images = [rescale(image, scale=rescale_factor) for image in images]
    
    # Step 4: Normalize images to have specified mean and std
    # Typically: mean≈0, std≈1 for better training dynamics
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    
    # Step 5: Move channel dimension to the first position
    # PyTorch expects [Channel, Height, Width] format
    # numpy arrays [H, W, C] → [C, H, W]
    images = [image.transpose(2, 0, 1) for image in images]
    
    return images

class PaliGemmaProcessor:
    """
    Main processor class for PaliGemma vision-language model.
    
    Responsibilities:
    1. Set up special tokens (image tokens, location tokens, segmentation tokens)
    2. Process images through the preprocessing pipeline
    3. Process text prompts (tokenization + special token insertion)
    4. Combine everything into model-ready format
    
    Architecture context:
    ┌───────────────────────────────────────────────┐
    │            PaliGemma Model                    │
    │  ┌──────────────┐      ┌──────────────┐       │
    │  │ Vision       │      │  Language    │       │
    │  │ Encoder      │      │  Model       │       │
    │  │ (SigLIP)     │      │  (Gemma)     │       │
    │  └──────┬───────┘      └──────┬───────┘       │
    │         │                     │               │
    │         │  Image Features     │  Text Tokens  │
    │         │  [B,256,768]        │  [B,Seq]      │
    │         └──────────┬──────────┘               │
    │                    │                          │
    │           ┌────────▼────────┐                 │
    │           │  Cross-Attention │                │
    │           │  Fusion Layer    │                │
    │           └─────────────────┘                 │
    └───────────────────────────────────────────────┘
                        ▲
                        │
              This processor prepares inputs
    """

    IMAGE_TOKEN = "<image>"  # Special token representing one image patch

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        """
        Initialize the processor with tokenizer and model specifications.
        
        Args:
            tokenizer: HuggingFace tokenizer for text processing
            num_image_tokens: Number of image patches (e.g., 256 for 16x16 grid on 224x224 image)
            image_size: Target image size (e.g., 224)
        """
        super().__init__()

        self.image_seq_length = num_image_tokens  # Typically 256 = 16×16 patches
        self.image_size = image_size              # Typically 224 pixels

        # Tokenizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        # Add the special image token to the tokenizer vocabulary
        # This token will be replaced with actual vision features later
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        
        # Add location tokens for object detection tasks
        # Format: <loc0000>, <loc0001>, ..., <loc1023>
        # These represent normalized bounding box coordinates
        # Example: <loc0100><loc0200><loc0500><loc0600> = box from (10%, 20%) to (50%, 60%)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)  # 1024 tokens for fine-grained coordinates
        ]
        
        # Add segmentation tokens for pixel-level segmentation tasks
        # Format: <seg000>, <seg001>, ..., <seg127>
        # These can represent different object segments/masks
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)  # 128 different segment IDs
        ]
        
        tokenizer.add_tokens(EXTRA_TOKENS)
        
        # Get the token ID for our image token (needed for identifying image positions later)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        
        # We will manually control BOS and EOS tokens
        # This gives us precise control over the input format
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict:
        """
        Process images and text into model-ready format.
        
        Complete flow:
        ┌─────────────────────────────────────────────────────────┐
        │  INPUT                                                  │
        │  - images: [PIL.Image]                                  │
        │  - text: ["describe this image"]                        │
        └───────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────▼─────────────────────────────────────┐
        │  IMAGE PROCESSING                                        │
        │  1. Resize to 224×224                                   │
        │  2. Normalize                                           │
        │  3. Convert to tensor [B, 3, 224, 224]                  │
        └───────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────▼─────────────────────────────────────┐
        │  TEXT PROCESSING                                         │
        │  1. Add image tokens: <image>×256                       │
        │  2. Add BOS: <bos>                                      │
        │  3. Add user text: "describe this image"                │
        │  4. Add newline: \n                                     │
        │  Result: "<image><image>...<bos>describe this image\n"  │
        └───────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────▼─────────────────────────────────────┐
        │  TOKENIZATION                                           │
        │  Convert string to token IDs: [45023, 45023, ..., 198]  │
        │  Generate attention mask: [1, 1, 1, ..., 1]            │
        └───────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────▼─────────────────────────────────────┐
        │  OUTPUT DICTIONARY                                       │
        │  {                                                       │
        │    "pixel_values": Tensor[B, 3, 224, 224],             │
        │    "input_ids": Tensor[B, Seq_Len],                    │
        │    "attention_mask": Tensor[B, Seq_Len]                │
        │  }                                                       │
        └──────────────────────────────────────────────────────────┘
        
        Args:
            text: List of text prompts (currently must be length 1)
            images: List of PIL Images (currently must be length 1)
            padding: Padding strategy for tokenization (e.g., "longest")
            truncation: Whether to truncate long sequences
            
        Returns:
            Dictionary with:
            - pixel_values: Preprocessed images [B, C, H, W]
            - input_ids: Tokenized text with image tokens [B, Seq_Len]
            - attention_mask: Mask indicating real vs padded tokens [B, Seq_Len]
        """
        # Current implementation supports batch size of 1 only
        # This is a simplification; production code might handle larger batches
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."

        # ===== IMAGE PROCESSING =====
        # Apply the full preprocessing pipeline to all images
        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),  # Resize to square (e.g., 224×224)
            resample=Image.Resampling.BICUBIC,        # High-quality interpolation
            rescale_factor=1 / 255.0,                 # [0, 255] → [0, 1]
            image_mean=IMAGENET_STANDARD_MEAN,        # [0.5, 0.5, 0.5]
            image_std=IMAGENET_STANDARD_STD,          # [0.5, 0.5, 0.5]
        )
        
        # Stack list of arrays into a single batch
        # List[[C, H, W]] → [Batch_Size, C, H, W]
        pixel_values = np.stack(pixel_values, axis=0)
        
        # Convert numpy array to PyTorch tensor
        # numpy.ndarray → torch.Tensor
        pixel_values = torch.tensor(pixel_values)

        # ===== TEXT PROCESSING =====
        # Prepend image tokens to each prompt
        # "describe" → "<image>×256<bos>describe\n"
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,      # "<bos>"
                image_seq_len=self.image_seq_length,     # 256
                image_token=self.IMAGE_TOKEN,            # "<image>"
            )
            for prompt in text
        ]

        # Tokenize the formatted strings
        # String → Token IDs + Attention Mask
        # Returns dictionary with PyTorch tensors:
        # - input_ids: [Batch_Size, Sequence_Length]
        # - attention_mask: [Batch_Size, Sequence_Length]
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",    # Return PyTorch tensors
            padding=padding,         # Pad to longest sequence in batch
            truncation=truncation,   # Truncate if exceeds max length
        )

        # ===== COMBINE OUTPUTS =====
        # Merge image tensor with tokenizer outputs
        # The model will:
        # 1. Process pixel_values through vision encoder → [B, 256, 768]
        # 2. Replace <image> tokens in input_ids with those features
        # 3. Process combined sequence through language model
        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data
