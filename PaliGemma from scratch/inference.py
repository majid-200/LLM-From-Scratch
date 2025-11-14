from PIL import Image
import torch
import fire

from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model

"""
PaliGemma Inference Script - Generate text from images

Purpose: Run PaliGemma model to answer questions about images or describe them

High-Level Flow:
┌────────────────────────────────────────────────────────┐
│  1. Load Model & Processor                             │
│     - Load pretrained weights                          │
│     - Initialize processor with tokenizer              │
└────────────────┬───────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│  2. Prepare Inputs                                     │
│     - Load image from file                             │
│     - Process image + text prompt                      │
│     - Move tensors to device (CPU/GPU)                 │
└────────────────┬───────────────────────────────────────┘
                 │
                 ▼
┌───────────────────────────────────────────────────────┐
│  3. Autoregressive Generation Loop                    │
│     ┌────────────────────────────────────┐            │
│     │  Generate one token at a time:     │            │
│     │  - Run model forward pass          │            │
│     │  - Sample next token from logits   │            │
│     │  - Add to sequence                 │            │
│     │  - Update KV cache                 │            │
│     │  - Repeat until stop token or max  │            │
│     └────────────────────────────────────┘            │
└────────────────┬──────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│  4. Decode & Display                                   │
│     - Convert token IDs to text                        │
│     - Print result                                     │
└────────────────────────────────────────────────────────┘
"""


def move_inputs_to_device(model_inputs: dict, device: str):
    """
    Move all tensors in the input dictionary to the specified device.
    
    Why needed?
    - Model is on GPU/CPU
    - Input tensors must be on same device as model
    - Otherwise: RuntimeError about tensor device mismatch
    
    Args:
        model_inputs: Dictionary with 'input_ids', 'attention_mask', 'pixel_values'
        device: Target device ('cuda', 'cpu', or 'mps')
        
    Returns:
        Dictionary with all tensors moved to device
    """
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs


def get_model_inputs(
    processor: PaliGemmaProcessor, prompt: str, image_file_path: str, device: str
):
    """
    Load image and prepare all inputs for the model.
    
    Process:
    ┌─────────────────────────────────────────────────────┐
    │  Image file → PIL Image                             │
    │  "cat.jpg"  → Image object                          │
    └────────────────┬────────────────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────────────────────────┐
    │  Processor combines image + text                    │
    │  - Resize/normalize image                           │
    │  - Tokenize: "<img>×256<bos>describe\n"             │
    │  Returns: {                                         │
    │    'pixel_values': [1, 3, 224, 224],                │
    │    'input_ids': [1, seq_len],                       │
    │    'attention_mask': [1, seq_len]                   │
    │  }                                                  │
    └────────────────┬────────────────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────────────────────────┐
    │  Move to device (GPU/CPU)                           │
    └─────────────────────────────────────────────────────┘
    
    Args:
        processor: PaliGemmaProcessor for input preparation
        prompt: Text prompt (e.g., "describe this image")
        image_file_path: Path to image file
        device: Device to move tensors to
        
    Returns:
        Dictionary with processed inputs ready for model
    """
    # Load image from file
    image = Image.open(image_file_path)
    images = [image]  # Batch of 1 image
    prompts = [prompt]  # Batch of 1 prompt
    
    # Process through PaliGemmaProcessor
    # This handles all preprocessing: image normalization, tokenization, etc.
    model_inputs = processor(text=prompts, images=images)
    
    # Move to appropriate device
    model_inputs = move_inputs_to_device(model_inputs, device)
    
    return model_inputs


def test_inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    """
    Run inference to generate text based on image and prompt.
    
    This implements AUTOREGRESSIVE GENERATION:
    - Generate one token at a time
    - Each new token depends on all previous tokens
    - Continue until stop token or max length
    
    Visual representation of generation:
    ┌──────────────────────────────────────────────────────────┐
    │  Step 0 (Prefill): Process entire prompt                 │
    │  Input: [<img>×256, <bos>, "describe", "\n"]             │
    │  Output: Logits for next token                           │
    │  Sample: "A" (token_id: 32)                              │
    │  KV Cache: Store K,V for all 259 tokens                  │
    └──────────────────────────────────────────────────────────┘
                            ↓
    ┌──────────────────────────────────────────────────────────┐
    │  Step 1 (Generate): Add "A", generate next               │
    │  Input: "A" (only new token!)                            │
    │  KV Cache: Reuse cached K,V for 259 previous tokens      │
    │  Output: Logits for next token                           │
    │  Sample: "cat" (token_id: 2385)                          │
    │  KV Cache: Append K,V for "A" → now 260 tokens cached    │
    └──────────────────────────────────────────────────────────┘
                            ↓
    ┌──────────────────────────────────────────────────────────┐
    │  Step 2 (Generate): Add "cat", generate next             │
    │  Input: "cat" (only new token!)                          │
    │  KV Cache: Reuse cached K,V for 260 previous tokens      │
    │  Output: Logits for next token                           │
    │  Sample: "sitting" (token_id: 8934)                      │
    │  KV Cache: Append K,V for "cat" → now 261 tokens         │
    └──────────────────────────────────────────────────────────┘
                            ↓
    ... Continue until <EOS> or max_tokens ...
    
    Args:
        model: PaliGemma model
        processor: Processor with tokenizer
        device: Device model is on
        prompt: Text prompt (e.g., "describe")
        image_file_path: Path to image
        max_tokens_to_generate: Maximum number of tokens to generate
        temperature: Controls randomness (higher = more random)
        top_p: Nucleus sampling threshold
        do_sample: If True, sample from distribution; if False, use greedy
    """
    # Prepare all inputs
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]          # [1, seq_len] - tokenized prompt
    attention_mask = model_inputs["attention_mask"] # [1, seq_len] - mask (all 1s)
    pixel_values = model_inputs["pixel_values"]     # [1, 3, 224, 224] - image

    # Initialize KV cache for efficient generation
    # Will store keys and values from previous tokens
    kv_cache = KVCache()

    # Get the stop token ID (marks end of generation)
    stop_token = processor.tokenizer.eos_token_id
    
    # List to collect generated tokens
    generated_tokens = []

    # ===== AUTOREGRESSIVE GENERATION LOOP =====
    for _ in range(max_tokens_to_generate):
        # Step 1: Run model forward pass
        # First iteration (prefill): input_ids = [1, 259] (256 image + 3 text tokens)
        # Later iterations: input_ids = [1, 1] (just the new token)
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        
        # Step 2: Update KV cache with newly computed keys/values
        kv_cache = outputs["kv_cache"]
        
        # Step 3: Get logits for next token
        # outputs["logits"] shape: [Batch_Size, Seq_Len, Vocab_Size]
        # We only care about the LAST position (the next token to generate)
        # [:, -1, :] extracts last position: [Batch_Size, Vocab_Size]
        next_token_logits = outputs["logits"][:, -1, :]  # [1, vocab_size]
        
        # Step 4: Sample the next token
        if do_sample:
            # Sampling mode: Use temperature and top-p for diversity
            
            # Apply temperature scaling:
            # - temperature < 1: Makes distribution sharper (less random)
            # - temperature > 1: Makes distribution flatter (more random)
            # - temperature = 1: No change
            # Then apply softmax to get probabilities
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            
            # Top-p (nucleus) sampling: Sample from smallest set of tokens
            # whose cumulative probability exceeds p
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            # Greedy mode: Always pick the most likely token
            # argmax finds the token with highest logit
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # Verify shape is correct
        assert next_token.size() == (1, 1)  # [Batch=1, Seq=1]
        
        # Remove batch dimension for easier handling
        next_token = next_token.squeeze(0)  # [1] - just the token ID
        
        # Step 5: Add to our list of generated tokens
        generated_tokens.append(next_token)
        
        # Step 6: Check if we should stop
        if next_token.item() == stop_token:
            print("(Stopped: EOS token generated)")
            break
        
        # Step 7: Prepare input for next iteration
        # The next input is ONLY the token we just generated
        # We don't need to pass the whole sequence again thanks to KV cache!
        input_ids = next_token.unsqueeze(-1)  # [1, 1] - batch dimension back
        
        # Step 8: Update attention mask to include the new token
        # Add a 1 to the mask (new token should be attended to)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )
        # attention_mask grows: [1, 259] → [1, 260] → [1, 261] → ...

    # ===== POST-PROCESSING =====
    
    # Concatenate all generated tokens into a single tensor
    # List of [1] tensors → [num_generated] tensor
    generated_tokens = torch.cat(generated_tokens, dim=-1)
    
    # Decode token IDs back to text
    # skip_special_tokens=True removes <EOS>, <PAD>, etc.
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Print the complete response (prompt + generated text)
    print(prompt + decoded)


def _sample_top_p(probs: torch.Tensor, p: float):
    """
    Nucleus (top-p) sampling: Sample from the smallest set of tokens whose
    cumulative probability exceeds p.
    
    Why use top-p instead of top-k?
    - Top-k always samples from exactly k tokens (rigid)
    - Top-p adapts: more tokens when distribution is flat, fewer when peaked
    
    Visual example with p=0.9:
    ┌────────────────────────────────────────────────────┐
    │  Original probability distribution:                │
    │  Token  | Prob  | Cumulative                       │
    │  -----  | ----  | ----------                       │
    │  "cat"  | 0.50  | 0.50  ← Include                  │
    │  "dog"  | 0.25  | 0.75  ← Include                  │
    │  "bird" | 0.15  | 0.90  ← Include (reaches p=0.9)  │
    │  "fish" | 0.05  | 0.95  ← Exclude (over threshold) │
    │  "lion" | 0.03  | 0.98  ← Exclude                  │
    │  ...    | ...   | ...   ← Exclude all rest         │
    └────────────────────────────────────────────────────┘
                          ↓
    ┌────────────────────────────────────────────────────┐
    │  After masking and renormalization:                │
    │  Token  | New Prob                                 │
    │  -----  | --------                                 │
    │  "cat"  | 0.556 (0.50 / 0.90)                      │
    │  "dog"  | 0.278 (0.25 / 0.90)                      │
    │  "bird" | 0.167 (0.15 / 0.90)                      │
    │  "fish" | 0.000 (masked out)                       │
    │  "lion" | 0.000 (masked out)                       │
    │  ...    | 0.000                                    │
    └────────────────────────────────────────────────────┘
                          ↓
              Sample from this renormalized distribution
    
    Benefits:
    - Filters out unlikely tokens (reduces nonsense)
    - Keeps diversity (multiple tokens possible)
    - Adapts to confidence (more options when uncertain)
    
    Args:
        probs: Probability distribution [Batch_Size, Vocab_Size]
        p: Cumulative probability threshold (typically 0.9 or 0.95)
        
    Returns:
        next_token: Sampled token ID [Batch_Size, 1]
    """
    # Step 1: Sort probabilities in descending order
    # probs_sort: sorted probabilities [B, vocab_size]
    # probs_idx: original indices of sorted probs [B, vocab_size]
    # Example: If "cat" (id=100) has highest prob, probs_idx[0] = 100
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    
    # Step 2: Compute cumulative sum of sorted probabilities
    # [0.5, 0.25, 0.15, 0.05, ...] → [0.5, 0.75, 0.90, 0.95, ...]
    # This tells us "how much probability have we accumulated so far?"
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    
    # Step 3: Create mask for tokens to exclude
    # Subtract probs_sort to shift cumsum by 1 position
    # This ensures we INCLUDE the token that crosses the threshold
    # Example with p=0.9:
    # - Token with cumsum=0.75: 0.75 - 0.25 = 0.50 < 0.9 → Keep
    # - Token with cumsum=0.90: 0.90 - 0.15 = 0.75 < 0.9 → Keep (crosses threshold)
    # - Token with cumsum=0.95: 0.95 - 0.05 = 0.90 > 0.9 → Mask
    mask = probs_sum - probs_sort > p
    
    # Step 4: Zero out probabilities of excluded tokens
    probs_sort[mask] = 0.0
    
    # Step 5: Renormalize so probabilities sum to 1
    # Divide each probability by the sum of remaining probabilities
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    
    # Step 6: Sample from the renormalized distribution
    # multinomial samples indices according to probability
    # num_samples=1: sample one token
    # Returns index in the SORTED array
    next_token = torch.multinomial(probs_sort, num_samples=1)
    
    # Step 7: Map back to original vocabulary index
    # gather retrieves the original token ID from probs_idx
    # Example: If we sampled position 2 in sorted array,
    #          and probs_idx[2] = 4567, then next_token = 4567
    next_token = torch.gather(probs_idx, -1, next_token)
    
    return next_token


def main(
    model_path: str = None,
    prompt: str = None,
    image_file_path: str = None,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
    only_cpu: bool = False,
):
    """
    Main entry point for PaliGemma inference.
    
    Usage examples:
    
    1. Describe an image (greedy decoding):
       python inference.py \\
           --model_path="google/paligemma-3b-pt-224" \\
           --prompt="describe" \\
           --image_file_path="cat.jpg" \\
           --do_sample=False
    
    2. Answer question about image (with sampling):
       python inference.py \\
           --model_path="google/paligemma-3b-pt-224" \\
           --prompt="what color is the cat?" \\
           --image_file_path="cat.jpg" \\
           --do_sample=True \\
           --temperature=0.7 \\
           --top_p=0.9
    
    3. Object detection (outputs bounding boxes as <loc####> tokens):
       python inference.py \\
           --model_path="google/paligemma-3b-pt-224" \\
           --prompt="detect cat" \\
           --image_file_path="cat.jpg"
    
    Args:
        model_path: HuggingFace model path or local directory
        prompt: Text prompt (e.g., "describe", "what is this?")
        image_file_path: Path to image file
        max_tokens_to_generate: Maximum tokens to generate (default: 100)
        temperature: Sampling temperature (default: 0.8)
                    - Lower (0.1-0.5): More focused, deterministic
                    - Higher (0.8-1.5): More creative, random
        top_p: Nucleus sampling threshold (default: 0.9)
                    - 0.9 = sample from top 90% probability mass
        do_sample: Use sampling (True) or greedy decoding (False)
                    - False: Always pick most likely token (deterministic)
                    - True: Sample from distribution (varied outputs)
        only_cpu: Force CPU usage even if GPU available
    """
    # ===== DEVICE SELECTION =====
    device = "cpu"

    if not only_cpu:
        # Check for available accelerators in order of preference
        if torch.cuda.is_available():
            device = "cuda"  # NVIDIA GPU
        elif torch.backends.mps.is_available():
            device = "mps"   # Apple Silicon GPU

    print("Device in use: ", device)

    # ===== MODEL LOADING =====
    print(f"Loading model from {model_path}")
    model, tokenizer = load_hf_model(model_path, device)
    
    # Move model to device and set to evaluation mode
    # .eval() disables dropout and sets batchnorm to inference mode
    model = model.to(device).eval()

    # ===== PROCESSOR INITIALIZATION =====
    # Extract config values needed for processor
    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    
    # Create processor that handles image preprocessing and tokenization
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    # ===== INFERENCE =====
    print("Running inference")
    with torch.no_grad():  # Disable gradient computation (saves memory, faster)
        test_inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
        )


if __name__ == "__main__":
    # Use Python Fire to automatically create CLI from main() function
    # Converts function arguments to command-line flags
    # Example: --model_path="path" --temperature=0.7
    fire.Fire(main)


"""
Key Concepts Summary:

1. **Autoregressive Generation**:
   - Generate one token at a time
   - Each token depends on all previous tokens
   - Continue until stop condition

2. **KV Cache**:
   - Stores keys and values from previous tokens
   - Avoids recomputing attention for old tokens
   - Makes generation ~10-100x faster

3. **Sampling Strategies**:
   - **Greedy**: Always pick most likely token (deterministic)
   - **Temperature**: Controls randomness of distribution
   - **Top-p (Nucleus)**: Sample from smallest set that exceeds probability p

4. **Two-Phase Generation**:
   - **Prefill**: Process entire prompt at once (slow, one-time)
   - **Decode**: Generate one token at a time (fast with KV cache)

5. **Why It Works**:
   - Model processes image as 256 visual tokens
   - Language model treats them like text tokens
   - Generates text that describes or answers about the image
"""
