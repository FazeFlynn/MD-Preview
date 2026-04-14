# nanoGPT — Full Codebase Explanation

## What Is This Project?

nanoGPT is a minimal, from-scratch implementation of the **GPT (Generative Pre-trained Transformer)** language model by Andrej Karpathy. It can:

1. **Train a GPT model from scratch** on any text dataset
2. **Fine-tune** OpenAI's pre-trained GPT-2 models on custom text
3. **Generate text** (sample) from a trained model

The entire core is just two files: `model.py` (~300 lines) defines the GPT architecture, and `train.py` (~300 lines) runs the training loop. Everything else is support code.

---

## High-Level Workflow

```
[1] Prepare Data          [2] Train Model          [3] Generate Text
    prepare.py        -->     train.py         -->     sample.py

  Raw text                 Reads train.bin/val.bin     Loads checkpoint
      |                    Builds/loads GPT model      Generates tokens
  Tokenize to integers     Trains with backprop        Decodes to text
      |                    Saves checkpoints
  Save as train.bin            |
  and val.bin              out/ckpt.pt
```

---

## File-by-File Breakdown

### `model.py` — The GPT Model (The Brain)

This file contains the **entire GPT-2 architecture** in pure PyTorch. Here's how the pieces fit together:

#### Building Blocks (bottom-up):

| Class | What It Does |
|-------|-------------|
| `LayerNorm` | Normalizes activations within a layer. Custom version that supports optional bias. |
| `CausalSelfAttention` | The core attention mechanism. Each token looks at all **previous** tokens (not future ones — that's the "causal" part) to decide what's relevant. Uses Flash Attention when available for speed. |
| `MLP` | A two-layer feed-forward network (expand 4x → GELU activation → project back). Processes each token independently after attention has mixed information across tokens. |
| `Block` | One Transformer block = LayerNorm → Attention → LayerNorm → MLP, with residual (skip) connections around each. |
| `GPTConfig` | A dataclass holding model hyperparameters: number of layers, heads, embedding size, vocab size, etc. |
| `GPT` | The full model. Stacks N `Block`s together with token + position embeddings at the input and a classification head at the output. |

#### How a Forward Pass Works:

```
Input token IDs [batch, sequence_length]
        |
   Token Embedding (lookup table: token → vector)
   + Position Embedding (lookup table: position → vector)
        |
   Dropout
        |
   Block 0: LayerNorm → Self-Attention → LayerNorm → MLP
   Block 1: LayerNorm → Self-Attention → LayerNorm → MLP
   ...
   Block N: LayerNorm → Self-Attention → LayerNorm → MLP
        |
   Final LayerNorm
        |
   Linear Head (project back to vocabulary size)
        |
   Output logits [batch, sequence_length, vocab_size]
   (probability distribution over next token)
```

#### Key Methods on `GPT`:

- **`forward(idx, targets)`** — Run input tokens through the model. If targets are given, also computes cross-entropy loss.
- **`from_pretrained(model_type)`** — Load official OpenAI GPT-2 weights from HuggingFace and map them into this model's format.
- **`configure_optimizers(...)`** — Creates an AdamW optimizer with weight decay for 2D parameters (weights) and no decay for 1D parameters (biases, layer norms).
- **`generate(idx, max_new_tokens)`** — Autoregressive text generation: predict next token, append it, repeat.
- **`crop_block_size(block_size)`** — Shrink the model's context window (useful when loading a pretrained model but wanting a smaller context).
- **`estimate_mfu(...)`** — Estimates Model FLOPs Utilization (what % of your GPU's theoretical peak you're achieving).

---

### `train.py` — The Training Loop (The Coach)

This is the main script you run to train or fine-tune a model. It handles everything:

#### Configuration (lines 1-70):
All hyperparameters are defined as plain Python variables at the top:
- **I/O**: `out_dir`, `eval_interval`, `always_save_checkpoint`
- **Model**: `n_layer`, `n_head`, `n_embd`, `dropout`, `bias`
- **Optimization**: `learning_rate`, `max_iters`, `weight_decay`, `grad_clip`
- **LR Schedule**: cosine decay with linear warmup
- **System**: `device`, `dtype`, `compile` (PyTorch 2.0 compilation)

These get overridden by config files and/or command-line args via `configurator.py`.

#### Initialization:
1. **DDP Setup** — Detects if running with `torchrun` for multi-GPU training. If so, sets up distributed data parallel.
2. **Data Loading** — `get_batch()` reads random chunks from `train.bin`/`val.bin` using memory-mapped files (efficient for huge datasets).
3. **Model Init** — Three modes:
   - `'scratch'`: Random initialization, train from zero
   - `'resume'`: Load from a saved checkpoint and continue training
   - `'gpt2'`/`'gpt2-medium'`/etc.: Load pretrained OpenAI weights for fine-tuning
4. **Optimizer** — AdamW with separate weight-decay groups
5. **Compilation** — Optionally uses `torch.compile()` for faster execution

#### Training Loop:
```
For each iteration:
  1. Set learning rate (cosine schedule with warmup)
  2. Every eval_interval steps:
     - Estimate train/val loss (average over eval_iters batches)
     - Save checkpoint if val loss improved
     - Log to wandb (optional)
  3. Forward pass:
     - Run micro-batches for gradient accumulation
     - Each micro-batch: forward → loss → backward
  4. Gradient clipping
  5. Optimizer step
  6. Log loss, time, MFU
  7. Repeat until max_iters
```

#### Key Concepts in the Training Loop:
- **Gradient Accumulation**: Simulates larger batch sizes by accumulating gradients over multiple micro-batches before stepping the optimizer.
- **Mixed Precision (AMP)**: Uses `float16` or `bfloat16` for faster training with `GradScaler` to prevent underflow.
- **DDP Gradient Sync**: Only syncs gradients across GPUs on the last micro-step for efficiency.

---

### `sample.py` — Text Generation (The Output)

Loads a trained model and generates text:

1. Load model from checkpoint (`out/ckpt.pt`) or from pretrained GPT-2
2. Set up the tokenizer:
   - If a `meta.pkl` exists (character-level), use its encoder/decoder
   - Otherwise, use GPT-2's BPE tokenizer (tiktoken)
3. Encode the starting prompt into token IDs
4. Call `model.generate()` autoregressively in a loop
5. Decode and print the results

Key parameters: `temperature` (randomness), `top_k` (limit sampling pool), `num_samples`, `max_new_tokens`.

---

### `configurator.py` — Configuration System (The Settings)

A clever "poor man's configurator" — it's not imported as a module; instead, it's `exec()`-ed inside `train.py` so it can directly override global variables.

Two ways to override defaults:
1. **Config file**: `python train.py config/train_gpt2.py` — executes the Python file, which sets variables
2. **CLI args**: `python train.py --batch_size=32 --learning_rate=1e-4` — parses and overrides individual values

---

### `bench.py` — Benchmarking (The Speedometer)

A stripped-down version of `train.py` for measuring training speed:
- Runs a few training iterations (10 burn-in + 20 measured)
- Reports time per iteration and MFU (Model FLOPs Utilization)
- Optional PyTorch profiler support for detailed analysis
- Can run with real data or random synthetic data

---

### `config/` — Preset Configurations

| File | Purpose |
|------|---------|
| `train_gpt2.py` | Train GPT-2 124M from scratch on OpenWebText (8×A100, ~4 days) |
| `train_shakespeare_char.py` | Train a tiny character-level GPT on Shakespeare (single GPU, ~3 min) |
| `finetune_shakespeare.py` | Fine-tune GPT-2 XL on Shakespeare (single GPU, minutes) |
| `eval_gpt2.py` | Evaluate pretrained GPT-2 124M on OpenWebText |
| `eval_gpt2_medium.py` | Evaluate pretrained GPT-2 350M |
| `eval_gpt2_large.py` | Evaluate pretrained GPT-2 774M |
| `eval_gpt2_xl.py` | Evaluate pretrained GPT-2 1.5B |

These are just Python files that set variables — they get exec'd by the configurator.

---

### `data/` — Dataset Preparation

Each subfolder has a `prepare.py` script that downloads and preprocesses text:

| Folder | Tokenization | Data Size | What It Does |
|--------|-------------|-----------|-------------|
| `openwebtext/` | GPT-2 BPE (tiktoken) | ~17GB train, ~8.5MB val | Downloads the OpenWebText dataset from HuggingFace, tokenizes ~8M documents, writes `train.bin` + `val.bin` |
| `shakespeare/` | GPT-2 BPE (tiktoken) | ~300K tokens | Downloads Tiny Shakespeare, encodes with GPT-2 tokenizer. Used for **fine-tuning** pretrained GPT-2. |
| `shakespeare_char/` | Character-level | ~1M characters | Downloads Tiny Shakespeare, maps each character to an integer. Used for **training from scratch**. Also saves `meta.pkl` with the character→int mapping. |

All output the same format: `train.bin` and `val.bin` files containing uint16 token IDs that `train.py` can directly memory-map.

---

## How to Understand the Code (Reading Order)

If you want to understand nanoGPT, read the files in this order:

### Step 1: Understand the Model (`model.py`)
Start here. Read bottom-up:
1. `GPTConfig` — what hyperparameters define a GPT
2. `LayerNorm` — simple normalization
3. `CausalSelfAttention` — the heart of the Transformer. Understand Q/K/V projections, multi-head splitting, causal masking, and how attention scores are computed.
4. `MLP` — straightforward feed-forward network
5. `Block` — puts attention + MLP together with residual connections
6. `GPT.__init__` — how the full model is assembled from blocks
7. `GPT.forward` — how data flows through: embeddings → blocks → output head → loss
8. `GPT.generate` — token-by-token autoregressive generation

### Step 2: Understand Data Prep (`data/shakespeare_char/prepare.py`)
The simplest example. See how raw text becomes binary token files.

### Step 3: Understand Training (`train.py`)
Read the main training loop. Key things to understand:
- How `get_batch()` loads random chunks from binary files
- The gradient accumulation loop (inner `for micro_step`)
- Mixed precision context (`ctx`)
- The learning rate schedule (`get_lr`)
- Checkpointing logic

### Step 4: Understand Sampling (`sample.py`)
See how a trained model generates text token by token.

### Step 5: Understand Configuration (`configurator.py`)
Short file. See how config files and CLI args override defaults.

---

## Key Concepts You Need to Know

### Tokenization
Text must be converted to numbers. Two approaches used here:
- **Character-level**: Each character is a token (vocab size = ~65 for Shakespeare)
- **BPE (Byte Pair Encoding)**: Subword tokenization used by GPT-2 (vocab size = 50,257)

### Autoregressive Language Modeling
The model is trained to predict the **next token** given all previous tokens. During generation, it predicts one token at a time and feeds it back as input.

### Causal Masking
In self-attention, each token can only attend to tokens **before** it (not after). This is what makes the model generate text left-to-right.

### Residual Connections
Each block adds its output to its input (`x = x + block(x)`). This helps gradients flow through deep networks.

### Weight Tying
The token embedding matrix and the final output projection share the same weights. This reduces parameter count and improves performance.

### Mixed Precision Training
Using `float16` or `bfloat16` instead of `float32` for most operations. Faster and uses less memory, with `GradScaler` preventing numerical issues (only needed for `float16`).

### Gradient Accumulation
When you can't fit a large batch in GPU memory, you process smaller micro-batches and accumulate gradients before updating weights. Mathematically equivalent to a larger batch.

### Distributed Data Parallel (DDP)
For multi-GPU training. Each GPU processes different data, gradients are averaged across GPUs before the optimizer step.

---

## Common Workflows

### Train a small model from scratch (quickstart):
```bash
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py
python sample.py --out_dir=out-shakespeare-char
```

### Train on YOUR OWN custom text (books, Wikipedia, articles):
```bash
# 1. Put your .txt files in data/custom/input/
# 2. Prepare the data:
python data/custom/prepare.py
# 3. Train:
python train.py config/train_custom.py
# 4. Generate:
python sample.py --out_dir=out-custom
```

### RAG-enhanced generation (search the web + generate):
```bash
# Single query:
python rag_sample.py --init_from=gpt2-xl --start="What is quantum computing?"

# Interactive mode:
python rag_sample.py --out_dir=out-custom

# With more search depth:
python rag_sample.py --init_from=gpt2 --num_search_results=10 --max_pages=8
```

### Fine-tune GPT-2 on custom text:
```bash
python data/shakespeare/prepare.py
python train.py config/finetune_shakespeare.py
python sample.py --out_dir=out-shakespeare
```

### Reproduce GPT-2 124M:
```bash
python data/openwebtext/prepare.py
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
python sample.py
```

### Evaluate pretrained GPT-2:
```bash
python train.py config/eval_gpt2.py
```

---

## New Features Added

### 1. Custom Data Training Pipeline (`data/custom/prepare.py`)

Train on **any text data** — books, Wikipedia articles, your own writing, anything:

```
data/custom/
  input/           ← Drop your .txt / .md files here
  prepare.py       ← Run this to tokenize everything
  train.bin        ← Created automatically
  val.bin          ← Created automatically
```

**Features:**
- Reads all `.txt`, `.md`, `.text`, `.csv`, `.html` files recursively from a directory
- Supports both **BPE** (GPT-2 tokenizer) and **character-level** encoding
- Automatic train/val split at document boundaries
- Handles encoding issues (UTF-8 with Latin-1 fallback)
- Configurable via CLI args: `--encoding=char`, `--val_fraction=0.1`, `--input_dir=path/`, etc.

**Examples:**
```bash
# BPE tokenization (default, for fine-tuning GPT-2):
python data/custom/prepare.py

# Character-level (for training from scratch):
python data/custom/prepare.py --encoding=char

# From a single file:
python data/custom/prepare.py --input_file=mybook.txt

# From a custom directory:
python data/custom/prepare.py --input_dir=C:/my_wikipedia_dumps/
```

### 2. RAG-Enhanced Generation (`rag_sample.py` + `search_utils.py`)

**Retrieval-Augmented Generation**: the model searches the web before answering.

```
User Prompt
    |
    ├── [1] Extract Topics  →  "quantum computing", "quantum applications"
    |
    ├── [2] Search Web      →  DuckDuckGo search for each topic
    |
    ├── [3] Fetch Pages     →  Top 5-10 links, extract text content
    |
    ├── [4] Build Context   →  Assemble retrieved text as context
    |
    └── [5] Generate        →  Context + Question → Model → Answer
```

**How it works:**
1. `search_utils.py` extracts keywords from your prompt (removes stopwords, identifies topics)
2. Searches DuckDuckGo for each topic (no API key needed)
3. Fetches the top pages and extracts clean text (strips HTML, scripts, nav, etc.)
4. Prepends the gathered context to your prompt
5. The model generates a response informed by the web content

**Files:**
- `search_utils.py` — Topic extraction, web search, page scraping, context assembly
- `rag_sample.py` — Ties it all together: loads model + runs RAG pipeline + generates

### 3. Training Config for Custom Data (`config/train_custom.py`)

Pre-configured settings for training on custom datasets with different hardware:
- **~8GB GPU** (default): 8-layer model, 512 context, batch 8
- **~4GB GPU**: 6-layer model, 256 context (commented preset)
- **~16GB+ GPU**: 12-layer model, 1024 context (commented preset)
- **CPU/MacBook**: 4-layer model, 256 context (commented preset)

### Required Additional Packages

```bash
pip install duckduckgo-search beautifulsoup4 requests
```

These are only needed for the RAG feature. Training on custom data only needs the original dependencies.
