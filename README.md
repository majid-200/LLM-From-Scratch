# Building LLMs from Scratch: A Hands-On Guide to GPT and Llama 3 Architectures

This repository documents my journey into building Large Language Models (LLMs) from scratch, following two separate, in-depth tutorials. The goal is to demystify the core components of modern transformer architectures by implementing them step-by-step in PyTorch.

The repository is divided into two main sections:
1.  **A GPT-like Model**: We build a complete, character-level Generative Pre-trained Transformer from the ground up, starting with a simple bigram model and progressively adding complexity.
2.  **Llama 3 Architectural Deep Dive**: We explore the key innovations in a state-of-the-art model, Llama 3, by implementing its advanced attention and feed-forward mechanisms in focused notebooks.

## Repository Structure

```
/
├── gpt/
│   ├── bigram.py               # A simple baseline language model.
│   ├── GPT.py                  # Full implementation of a GPT-like transformer.
│   ├── GPT-Scratch.ipynb       # A detailed, step-by-step notebook for building the GPT model.
│   ├── BPE_Scratch.py          # Script for Byte-Pair Encoding tokenizer.
│   ├── BPE_Scratch.ipynb       # Notebook explaining and implementing BPE from scratch.
│   └── TinyShakespear.txt      # The training dataset.
│
├── llama3/
│   ├── Llama_3_Attention.ipynb # Notebook implementing Llama 3's advanced attention mechanism.
│   └── Llama_3_feed_forward.ipynb # Notebook implementing Llama 3's SwiGLU feed-forward network.
│
└── README.md                   # You are here!
```

---

## Part 1: Building a GPT-like Model

This section focuses on constructing a decoder-only transformer model, similar in spirit to OpenAI's GPT-2, trained on the works of Shakespeare.

### Key Concepts Covered
-   Tokenization (Character-level and Subword-level)
-   Bigram Language Models
-   Transformer Architecture
-   Self-Attention Mechanism
-   Multi-Head Attention
-   Positional Embeddings
-   Layer Normalization and Residual Connections
-   Autoregressive Text Generation

### File Breakdown

#### 📜 `bigram.py`
This script serves as the "Hello World" of language modeling. It implements a simple Bigram model where the prediction for the next character depends only on the immediately preceding character. It's a crucial baseline that helps establish the data loading pipeline and evaluation framework before moving to a more complex architecture.

#### 🤖 `GPT.py` & `GPT-Scratch.ipynb`
These files are the core of this section.
-   The `GPT-Scratch.ipynb` notebook provides a narrative, cell-by-cell walkthrough of the entire process. It explains the "why" behind each component, from the mathematical trick in self-attention to the final model assembly.
-   The `GPT.py` script is a clean, runnable implementation of the final model.

The model built here includes all the fundamental components of a transformer block:
-   **Multi-Head Self-Attention**: Allows tokens to communicate with each other and weigh the importance of different tokens in the context.
-   **Feed-Forward Network**: A simple MLP applied to each token position independently, adding computational depth.
-   **Residual Connections & Layer Normalization**: Critical for stabilizing training in deep networks by preventing vanishing/exploding gradients.

#### 🧩 `BPE_Scratch.ipynb` & `BPE_Scratch.py`
While our GPT model uses simple character-level tokenization, modern LLMs use more sophisticated methods. This notebook explores **Byte-Pair Encoding (BPE)**, a subword tokenization algorithm. It starts with a vocabulary of individual characters and iteratively merges the most frequent adjacent pairs of tokens. This allows the model to handle rare words and create a more efficient, semantically meaningful vocabulary. The notebook visualizes this merging process step-by-step.

---

## Part 2: Deconstructing Llama 3's Architecture

This section moves beyond the foundational GPT architecture to explore the specific, high-performance components used in Meta's Llama 3 model. These notebooks isolate and implement these key innovations.

### Key Concepts Covered
-   **RMSNorm**: A simpler and more efficient alternative to LayerNorm.
-   **SwiGLU (Gated Linear Unit)**: An advanced feed-forward network that often provides better performance than standard ReLU-based networks.
-   **Rotary Positional Embeddings (RoPE)**: A sophisticated method for encoding positional information by rotating the query and key vectors, allowing the model to better understand relative positions.
-   **Grouped-Query Attention (GQA)**: An optimization of the attention mechanism that groups query heads to share a single key and value head, offering a balance between the performance of Multi-Head Attention and the efficiency of Multi-Query Attention.
-   **QK Normalization**: An additional normalization step applied to query and key vectors before the attention calculation to improve training stability.

### File Breakdown

#### 🧠 `Llama_3_Attention.ipynb`
This notebook is a deep dive into the heart of the Llama 3 transformer block: its attention mechanism. It meticulously reconstructs the entire attention forward pass, explaining and implementing:
1.  **Q, K, V Projections**: Standard linear transformations of the input.
2.  **Applying RoPE**: Integrating relative positional information directly into the queries and keys.
3.  **QK Norm**: Applying L2 normalization to the Q and K vectors.
4.  **Grouped-Query Attention (GQA)**: Efficiently handling keys and values by repeating them to match the number of query heads.
5.  **Scaled Dot-Product Attention**: The final calculation of attention scores and output.

#### 🚀 `Llama_3_feed_forward.ipynb`
This notebook focuses on the other major component of the Llama 3 transformer block: the feed-forward network. It implements the **SwiGLU** variant, which involves three linear projections (`gate`, `up`, and `down`) instead of the standard two. It also demonstrates the use of **RMSNorm** for pre-normalization, a key feature of the Llama architecture that contributes to its efficiency and stability.

---

## How to Use This Repository

### Prerequisites
-   Python 3.8+
-   PyTorch
-   Jupyter Notebook or JupyterLab (for `.ipynb` files)

### Setup and Running
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install torch torchvision torchaudio notebook
    ```

3.  **Download the dataset:**
    The `TinyShakespear.txt` dataset is used for the GPT model. If not already present, you can download it.
    ```bash
    wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O gpt/TinyShakespear.txt
    ```

4.  **Run the Python scripts:**
    To train the GPT model, navigate to the `gpt` directory and run the script:
    ```bash
    cd gpt
    python GPT.py
    ```

5.  **Explore the notebooks:**
    Launch Jupyter and navigate through the notebooks to see the step-by-step implementations.
    ```bash
    jupyter notebook
    ```

## Acknowledgements
This project is heavily based on the excellent, educational content from online tutorials that break down these complex topics into manageable pieces. The implementations here are my personal re-creations aimed at solidifying my own understanding of these powerful models.
