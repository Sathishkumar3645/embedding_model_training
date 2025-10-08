readme_content = """
# Multilingual Embedding Model for RAG

A sophisticated multilingual embedding model designed for Retrieval-Augmented Generation (RAG) applications. This model can handle queries in multiple languages and retrieve relevant content regardless of the language mismatch between query and corpus.

## Features

- **Multilingual Support**: Works with 12+ languages including English, German, French, Spanish, Italian, Portuguese, Dutch, Polish, Russian, Chinese, Japanese, and Arabic
- **Cross-platform Compatibility**: Runs on Mac (M1/M2), Windows, and Linux
- **High-quality Embeddings**: 768-dimensional embeddings optimized for semantic similarity
- **RAG-Optimized**: Specifically designed for retrieval tasks
- **Production Ready**: Includes inference optimization and model export capabilities

## Quick Start

### 1. Installation

```bash
# Clone or download the files
# Install dependencies
python install_dependencies.py

# Or manually install:
pip install -r requirements.txt
```

### 2. Training

```bash
# Simple training
python run_training.py

# Or with custom parameters
python multilingual_embedding_model.py --action train
```

### 3. Testing

```bash
# Test the trained model
python test_model.py

# Or run demo
python multilingual_embedding_model.py --action demo
```

## Usage Examples

### Basic Inference

```python
from multilingual_embedding_model import MultilingualEmbeddingInference

# Load trained model
inference = MultilingualEmbeddingInference('model_checkpoints/best_multilingual_embedding_model.pt')

# Encode texts
texts = ["Hello world", "Hola mundo", "Bonjour le monde"]
embeddings = inference.encode(texts)

# Compute similarity
similarity = inference.similarity("What is AI?", "Was ist KI?")
print(f"Similarity: {similarity:.4f}")

# Find similar texts
corpus = ["AI is amazing", "I love cooking", "Machine learning is powerful"]
results = inference.find_most_similar("What is artificial intelligence?", corpus, top_k=2)
```

### Cross-lingual Retrieval

```python
# English query, multilingual corpus
query = "What is machine learning?"
corpus = [
    "Machine learning is a subset of AI",  # English
    "L'apprentissage automatique est une branche de l'IA",  # French
    "Maschinelles Lernen ist ein Teilgebiet der KI",  # German
    "I love cooking pasta"  # Unrelated
]

results = inference.find_most_similar(query, corpus, top_k=3)
# Will correctly rank ML-related content higher regardless of language
```

## Model Architecture

- **Base Model**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- **Embedding Dimension**: 768
- **Max Sequence Length**: 512 tokens
- **Pooling Strategy**: Mean pooling with attention masking
- **Loss Function**: Combined triplet loss and contrastive loss
- **Normalization**: L2 normalization for consistent similarity computation

## Dataset

The model is trained on:
- **AllNLI**: Combination of SNLI and MultiNLI datasets
- **Custom Multilingual Data**: Domain-specific examples
- **MS MARCO**: Optional passage retrieval dataset

## Performance

- **Cross-lingual Retrieval**: Excellent performance across language pairs
- **Semantic Similarity**: High correlation with human judgments
- **Speed**: Optimized for both training and inference
- **Memory**: Efficient memory usage with gradient accumulation

## System Requirements

- **Python**: 3.8+
- **Memory**: 8GB+ RAM recommended
- **GPU**: Optional but recommended (supports CUDA and Apple Silicon MPS)
- **Storage**: 5GB+ for model and data

## Platform Support

### macOS (M1/M2)
- Native Apple Silicon support with MPS
- Optimized batch sizes for Mac hardware
- Automatic fallback to CPU if needed

### Windows
- CUDA support for NVIDIA GPUs
- CPU fallback for systems without GPU
- Optimized for Windows environment

### Linux
- Full CUDA support
- Distributed training capabilities
- Docker support (optional)

## Advanced Usage

### Custom Training Data

```python
# Create custom dataset
custom_data = [
    {
        "query": "Your query",
        "positive": "Relevant text",
        "negative": "Irrelevant text"
    }
]

# Train with custom data
python multilingual_embedding_model.py --action custom --data_path custom_data.json
```

### Model Export

```python
# Export for production
from multilingual_embedding_model import export_model_for_production

export_model_