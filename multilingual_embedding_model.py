import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import json
from tqdm import tqdm
import os
import platform
import requests
import gzip
import csv
from dataclasses import dataclass
import logging
from sklearn.metrics.pairwise import cosine_similarity
import random
from datasets import load_dataset, Dataset as HFDataset
import pickle
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for multilingual embedding model"""
    base_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedding_dim: int = 768  # Good dimension for multilingual tasks
    max_length: int = 512
    batch_size: int = 16  # Optimized for Mac M1/M2 and Windows
    learning_rate: float = 2e-5
    num_epochs: int = 4
    warmup_steps: int = 1000
    temperature: float = 0.05
    margin: float = 0.3
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    save_steps: int = 1000
    eval_steps: int = 500
    
    # Dataset configuration
    dataset_name: str = "sentence-transformers/all-nli"
    max_train_samples: int = 50000
    max_val_samples: int = 5000
    
    # Multilingual settings
    languages: List[str] = None  # Will be set to common languages
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ['en', 'de', 'fr', 'es', 'it', 'pt', 'nl', 'pl', 'ru', 'zh', 'ja', 'ar']

class MultilingualDatasetProcessor:
    """Process datasets for multilingual embedding training"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.data_dir = "embedding_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def download_and_process_allnli(self):
        """Download and process AllNLI dataset"""
        logger.info("Loading AllNLI dataset...")
        
        try:
            # Load the dataset
            dataset = load_dataset(self.config.dataset_name)
            
            # Process training data
            train_data = self._process_nli_split(dataset['train'], 'train')
            val_data = self._process_nli_split(dataset['dev'], 'validation')
            
            # Save processed data
            self._save_processed_data(train_data, 'train_data.json')
            self._save_processed_data(val_data, 'val_data.json')
            
            logger.info(f"Processed {len(train_data)} training samples and {len(val_data)} validation samples")
            return True
            
        except Exception as e:
            logger.error(f"Error processing AllNLI dataset: {e}")
            return False
    
    def download_and_process_msmarco(self):
        """Download and process MS MARCO dataset"""
        logger.info("Loading MS MARCO dataset...")
        
        try:
            # Load MS MARCO passage ranking dataset
            dataset = load_dataset("ms_marco", "v1.1")
            
            # Process the dataset
            train_data = self._process_msmarco_split(dataset['train'])
            val_data = self._process_msmarco_split(dataset['validation'])
            
            # Save processed data
            self._save_processed_data(train_data, 'train_data.json')
            self._save_processed_data(val_data, 'val_data.json')
            
            logger.info(f"Processed {len(train_data)} training samples and {len(val_data)} validation samples")
            return True
            
        except Exception as e:
            logger.error(f"Error processing MS MARCO dataset: {e}")
            return False
    
    def _process_nli_split(self, split_data, split_name):
        """Process NLI data split"""
        processed_data = []
        max_samples = self.config.max_train_samples if split_name == 'train' else self.config.max_val_samples
        
        # Sample data if too large
        if len(split_data) > max_samples:
            indices = random.sample(range(len(split_data)), max_samples)
            split_data = split_data.select(indices)
        
        for item in tqdm(split_data, desc=f"Processing {split_name} data"):
            if item['label'] == 0:  # Entailment - positive pair
                processed_data.append({
                    'query': item['premise'],
                    'positive': item['hypothesis'],
                    'negative': self._get_random_negative(split_data, item['premise'])
                })
            elif item['label'] == 2:  # Contradiction - can be used as negative
                # Find a neutral or entailment pair for positive
                positive_text = self._get_random_positive(split_data, item['premise'])
                if positive_text:
                    processed_data.append({
                        'query': item['premise'],
                        'positive': positive_text,
                        'negative': item['hypothesis']
                    })
        
        return processed_data
    
    def _process_msmarco_split(self, split_data):
        """Process MS MARCO data split"""
        processed_data = []
        
        for item in tqdm(split_data, desc="Processing MS MARCO data"):
            if 'passages' in item and 'is_selected' in item:
                query = item['query']
                positive_passages = [p['passage_text'] for p, selected in zip(item['passages'], item['is_selected']) if selected]
                negative_passages = [p['passage_text'] for p, selected in zip(item['passages'], item['is_selected']) if not selected]
                
                # Create training pairs
                for positive in positive_passages:
                    if negative_passages:
                        negative = random.choice(negative_passages)
                        processed_data.append({
                            'query': query,
                            'positive': positive,
                            'negative': negative
                        })
        
        return processed_data
    
    def _get_random_negative(self, dataset, query):
        """Get a random negative example"""
        max_attempts = 10
        for _ in range(max_attempts):
            random_item = random.choice(dataset)
            if random_item['premise'] != query:
                return random_item['hypothesis']
        return "This is a random negative example."
    
    def _get_random_positive(self, dataset, query):
        """Get a random positive example"""
        for item in dataset:
            if item['premise'] == query and item['label'] == 0:
                return item['hypothesis']
        return None
    
    def _save_processed_data(self, data, filename):
        """Save processed data to JSON file"""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(data)} samples to {filepath}")

class MultilingualEmbeddingDataset(Dataset):
    """Dataset for multilingual embedding training"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_path)
        
    def load_data(self, data_path: str) -> List[Dict]:
        """Load training data from JSON file"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} samples from {data_path}")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize query, positive, and negative examples
        query = self.tokenize_text(item['query'])
        positive = self.tokenize_text(item['positive'])
        negative = self.tokenize_text(item['negative'])
        
        return {
            'query': query,
            'positive': positive,
            'negative': negative
        }
    
    def tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text with proper padding and truncation"""
        if not isinstance(text, str):
            text = str(text)
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze()
        }

class MultilingualEmbeddingModel(nn.Module):
    """Multilingual embedding model with advanced pooling"""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__()
        self.config = config
        self.base_model = AutoModel.from_pretrained(config.base_model)
        
        # Get the hidden size from base model
        self.hidden_size = self.base_model.config.hidden_size
        
        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, config.embedding_dim),
            nn.Tanh(),
            nn.Linear(config.embedding_dim, config.embedding_dim)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection layer weights"""
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)
    
    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling with attention mask"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def cls_pooling(self, model_output):
        """CLS token pooling"""
        return model_output[0][:, 0]
    
    def max_pooling(self, model_output, attention_mask):
        """Max pooling with attention mask"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9
        return torch.max(token_embeddings, 1)[0]
    
    def forward(self, input_ids, attention_mask, pooling_strategy='mean'):
        """Forward pass through the model"""
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Apply different pooling strategies
        if pooling_strategy == 'mean':
            pooled_output = self.mean_pooling(outputs, attention_mask)
        elif pooling_strategy == 'cls':
            pooled_output = self.cls_pooling(outputs)
        elif pooling_strategy == 'max':
            pooled_output = self.max_pooling(outputs, attention_mask)
        else:
            pooled_output = self.mean_pooling(outputs, attention_mask)
        
        # Apply projection
        embeddings = self.projection(pooled_output)
        
        # Apply dropout and layer norm
        embeddings = self.dropout(embeddings)
        embeddings = self.layer_norm(embeddings)
        
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings

class MultiScaleLoss(nn.Module):
    """Combined loss function for better training"""
    
    def __init__(self, margin: float = 0.3, temperature: float = 0.05, alpha: float = 0.5):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.alpha = alpha
    
    def triplet_loss(self, anchor, positive, negative):
        """Triplet loss"""
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
    
    def contrastive_loss(self, anchor, positive, negative):
        """Contrastive loss with temperature scaling"""
        pos_sim = F.cosine_similarity(anchor, positive, dim=1) / self.temperature
        neg_sim = F.cosine_similarity(anchor, negative, dim=1) / self.temperature
        
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def forward(self, anchor, positive, negative):
        """Compute combined loss"""
        triplet_loss = self.triplet_loss(anchor, positive, negative)
        contrastive_loss = self.contrastive_loss(anchor, positive, negative)
        
        return self.alpha * triplet_loss + (1 - self.alpha) * contrastive_loss

class MultilingualEmbeddingTrainer:
    """Trainer for multilingual embedding models"""
    
    def __init__(self, model: MultilingualEmbeddingModel, config: EmbeddingConfig):
        self.model = model
        self.config = config
        
        # Setup device (Mac M1/M2 and Windows compatible)
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')  # Mac M1/M2
            logger.info("Using MPS (Apple Silicon) device")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')  # Windows/Linux with CUDA
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU device")
        
        self.model.to(self.device)
        
        # Initialize loss
        self.criterion = MultiScaleLoss(
            margin=config.margin,
            temperature=config.temperature,
            alpha=0.5
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize metrics
        self.train_losses = []
        self.val_losses = []
        self.global_step = 0
        
    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}')
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                query = {k: v.to(self.device) for k, v in batch['query'].items()}
                positive = {k: v.to(self.device) for k, v in batch['positive'].items()}
                negative = {k: v.to(self.device) for k, v in batch['negative'].items()}
                
                # Forward pass
                query_emb = self.model(**query)
                positive_emb = self.model(**positive)
                negative_emb = self.model(**negative)
                
                # Compute loss
                loss = self.criterion(query_emb, positive_emb, negative_emb)
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                total_loss += loss.item() * self.config.gradient_accumulation_steps
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * self.config.gradient_accumulation_steps:.4f}',
                    'avg_loss': f'{total_loss / num_batches:.4f}'
                })
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def validate(self, dataloader: DataLoader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validating'):
                try:
                    query = {k: v.to(self.device) for k, v in batch['query'].items()}
                    positive = {k: v.to(self.device) for k, v in batch['positive'].items()}
                    negative = {k: v.to(self.device) for k, v in batch['negative'].items()}
                    
                    query_emb = self.model(**query)
                    positive_emb = self.model(**positive)
                    negative_emb = self.model(**negative)
                    
                    loss = self.criterion(query_emb, positive_emb, negative_emb)
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.error(f"Error in validation batch: {e}")
                    continue
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def train(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None):
        """Full training loop"""
        # Initialize scheduler
        total_steps = len(train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        best_val_loss = float('inf')
        
        logger.info(f"Starting training for {self.config.num_epochs} epochs...")
        logger.info(f"Total training steps: {total_steps}")
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss = self.train_epoch(train_dataloader, epoch)
            self.train_losses.append(train_loss)
            
            logger.info(f'Epoch {epoch+1}/{self.config.num_epochs} - Train Loss: {train_loss:.4f}')
            
            # Validate
            if val_dataloader:
                val_loss = self.validate(val_dataloader)
                self.val_losses.append(val_loss)
                logger.info(f'Epoch {epoch+1}/{self.config.num_epochs} - Val Loss: {val_loss:.4f}')
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(f'best_multilingual_embedding_model.pt')
                    logger.info(f'New best model saved with val loss: {val_loss:.4f}')
            
            # Update scheduler
            scheduler.step()
        
        # Save final model
        self.save_model('final_multilingual_embedding_model.pt')
        logger.info('Training completed!')
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        save_dir = "model_checkpoints"
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, path)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'global_step': self.global_step
        }, full_path)
        
        logger.info(f'Model saved to {full_path}')
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        save_dir = "model_checkpoints"
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, path)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'global_step': self.global_step
        }, full_path)

class MultilingualEmbeddingInference:
    """Inference class for multilingual embedding model"""
    
    def __init__(self, model_path: str, config: Optional[EmbeddingConfig] = None):
        # Setup device
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = config or checkpoint['config']
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        
        # Load model
        self.model = MultilingualEmbeddingModel(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Using device: {self.device}")
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32, 
               show_progress: bool = True, normalize: bool = True) -> np.ndarray:
        """Encode texts to embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding")
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            try:
                encoded = self.tokenizer(
                    batch_texts,
                    max_length=self.config.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Generate embeddings
                with torch.no_grad():
                    embeddings = self.model(input_ids, attention_mask)
                    all_embeddings.append(embeddings.cpu().numpy())
                    
            except Exception as e:
                logger.error(f"Error encoding batch: {e}")
                # Return zero embeddings for failed batch
                zero_embeddings = np.zeros((len(batch_texts), self.config.embedding_dim))
                all_embeddings.append(zero_embeddings)
        
        result = np.vstack(all_embeddings)
        
        if normalize:
            result = result / (np.linalg.norm(result, axis=1, keepdims=True) + 1e-8)
        
        return result
    
    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts"""
        embeddings = self.encode([text1, text2], show_progress=False)
        return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
    
    def find_most_similar(self, query: str, texts: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar texts to query"""
        query_embedding = self.encode([query], show_progress=False)
        text_embeddings = self.encode(texts, show_progress=True)
        
        similarities = cosine_similarity(query_embedding, text_embeddings)[0]
        
        # Get top k most similar
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((texts[idx], float(similarities[idx])))
        
        return results

def setup_environment():
    """Setup environment for cross-platform compatibility"""
    system = platform.system()
    logger.info(f"Running on {system}")
    
    if system == "Darwin":  # macOS
        logger.info("Mac detected - optimizing for Apple Silicon")
        if torch.backends.mps.is_available():
            logger.info("MPS backend available")
        else:
            logger.info("MPS backend not available, using CPU")
    elif system == "Windows":
        logger.info("Windows detected")
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
        else:
            logger.info("CUDA not available, using CPU")
    
    # Set environment variables for better performance
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

def main():
    """Main training and inference function"""
    # Setup environment
    setup_environment()
    
    # Configuration
    config = EmbeddingConfig(
        base_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        embedding_dim=768,
        max_length=512,
        batch_size=8,  # Reduced for better compatibility
        learning_rate=2e-5,
        num_epochs=3,
        warmup_steps=500,
        temperature=0.05,
        margin=0.3,
        gradient_accumulation_steps=4,
        max_train_samples=20000,
        max_val_samples=2000
    )
    
    logger.info("Starting multilingual embedding model training...")
    
    # Initialize dataset processor
    processor = MultilingualDatasetProcessor(config)
    
    # Download and process dataset
    logger.info("Processing dataset...")
    if not processor.download_and_process_allnli():
        logger.error("Failed to process dataset")
        return
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    
    # Create datasets
    train_dataset = MultilingualEmbeddingDataset(
        os.path.join(processor.data_dir, 'train_data.json'),
        tokenizer,
        config.max_length
    )
    
    val_dataset = MultilingualEmbeddingDataset(
        os.path.join(processor.data_dir, 'val_data.json'),
        tokenizer,
        config.max_length
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Mac compatibility
        pin_memory=False
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Initialize model
    model = MultilingualEmbeddingModel(config)
    
    # Initialize trainer
    trainer = MultilingualEmbeddingTrainer(model, config)
    
    # Train model
    trainer.train(train_dataloader, val_dataloader)
    
    # Test inference
    logger.info("Testing inference...")
    try:
        inference = MultilingualEmbeddingInference(
            'model_checkpoints/best_multilingual_embedding_model.pt',
            config
        )
        
        # Test multilingual similarity
        test_cases = [
            ("What is machine learning?", "Machine learning is a subset of AI"),
            ("Was ist maschinelles Lernen?", "Machine learning is a subset of AI"),  # German
            ("Qu'est-ce que l'apprentissage automatique?", "Machine learning is a subset of AI"),  # French
            ("¿Qué es el aprendizaje automático?", "Machine learning is a subset of AI"),  # Spanish
        ]
        
        for query, text in test_cases:
            similarity = inference.similarity(query, text)
            logger.info(f"Similarity between '{query}' and '{text}': {similarity:.4f}")
        
        # Test embedding generation
        test_texts = [
            "Hello world",
            "Hola mundo",
            "Bonjour le monde",
            "Hallo Welt",
            "Ciao mondo"
        ]
        
        embeddings = inference.encode(test_texts)
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        
        # Test cross-lingual retrieval
        corpus = [
            "Machine learning is a method of data analysis that automates analytical model building.",
            "Maschinelles Lernen ist eine Methode der Datenanalyse, die den Aufbau analytischer Modelle automatisiert.",
            "L'apprentissage automatique est une méthode d'analyse de données qui automatise la construction de modèles analytiques.",
            "El aprendizaje automático es un método de análisis de datos que automatiza la construcción de modelos analíticos.",
            "I love cooking pasta with tomatoes.",
            "Sports are great for physical fitness.",
            "Music helps me relax after work."
        ]
        
        # Query in German, should find ML-related sentences
        german_query = "Was ist maschinelles Lernen?"
        results = inference.find_most_similar(german_query, corpus, top_k=3)
        
        logger.info(f"\nTop 3 results for German query: '{german_query}'")
        for i, (text, score) in enumerate(results):
            logger.info(f"{i+1}. Score: {score:.4f} - Text: {text[:100]}...")
        
        # Query in English, should find ML-related sentences
        english_query = "What is machine learning?"
        results = inference.find_most_similar(english_query, corpus, top_k=3)
        
        logger.info(f"\nTop 3 results for English query: '{english_query}'")
        for i, (text, score) in enumerate(results):
            logger.info(f"{i+1}. Score: {score:.4f} - Text: {text[:100]}...")
        
    except Exception as e:
        logger.error(f"Error in inference testing: {e}")
        logger.info("You can still use the trained model for inference later.")

def create_custom_dataset():
    """Create a custom multilingual dataset for specific domains"""
    custom_data = []
    
    # Technology domain - multilingual
    tech_queries = [
        "What is artificial intelligence?",
        "Was ist künstliche Intelligenz?",
        "Qu'est-ce que l'intelligence artificielle?",
        "¿Qué es la inteligencia artificial?",
        "What is deep learning?",
        "Was ist Deep Learning?",
        "Qu'est-ce que l'apprentissage profond?",
        "¿Qué es el aprendizaje profundo?"
    ]
    
    tech_positives = [
        "Artificial intelligence is the simulation of human intelligence processes by machines.",
        "Künstliche Intelligenz ist die Simulation menschlicher Intelligenzprozesse durch Maschinen.",
        "L'intelligence artificielle est la simulation des processus d'intelligence humaine par des machines.",
        "La inteligencia artificial es la simulación de procesos de inteligencia humana por máquinas.",
        "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
        "Deep Learning ist eine Teilmenge des maschinellen Lernens, die neuronale Netze mit mehreren Schichten verwendet.",
        "L'apprentissage profond est un sous-ensemble de l'apprentissage automatique qui utilise des réseaux de neurones à plusieurs couches.",
        "El aprendizaje profundo es un subconjunto del aprendizaje automático que utiliza redes neuronales con múltiples capas."
    ]
    
    tech_negatives = [
        "Cooking is the art of preparing food by combining ingredients.",
        "Kochen ist die Kunst, Essen durch das Kombinieren von Zutaten zuzubereiten.",
        "La cuisine est l'art de préparer des aliments en combinant des ingrédients.",
        "Cocinar es el arte de preparar comida combinando ingredientes.",
        "Sports involve physical activity and competition between individuals or teams.",
        "Sport beinhaltet körperliche Aktivität und Wettbewerb zwischen Individuen oder Teams.",
        "Le sport implique une activité physique et une compétition entre individus ou équipes.",
        "Los deportes involucran actividad física y competencia entre individuos o equipos."
    ]
    
    # Create training pairs
    for i in range(len(tech_queries)):
        custom_data.append({
            'query': tech_queries[i],
            'positive': tech_positives[i],
            'negative': tech_negatives[i % len(tech_negatives)]
        })
    
    # Save custom dataset
    os.makedirs("custom_data", exist_ok=True)
    with open("custom_data/custom_multilingual_data.json", 'w', encoding='utf-8') as f:
        json.dump(custom_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Created custom dataset with {len(custom_data)} samples")
    return "custom_data/custom_multilingual_data.json"

def evaluate_model_performance(model_path: str, test_data_path: str):
    """Evaluate model performance on test data"""
    logger.info("Evaluating model performance...")
    
    # Load model
    config = EmbeddingConfig()
    inference = MultilingualEmbeddingInference(model_path, config)
    
    # Load test data
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Evaluate
    correct_predictions = 0
    total_predictions = 0
    
    for item in tqdm(test_data[:100], desc="Evaluating"):  # Test on first 100 samples
        query = item['query']
        positive = item['positive']
        negative = item['negative']
        
        # Compute similarities
        pos_sim = inference.similarity(query, positive)
        neg_sim = inference.similarity(query, negative)
        
        # Check if positive is more similar than negative
        if pos_sim > neg_sim:
            correct_predictions += 1
        
        total_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    logger.info(f"Model accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
    
    return accuracy

def fine_tune_on_custom_data(base_model_path: str, custom_data_path: str):
    """Fine-tune model on custom data"""
    logger.info("Fine-tuning model on custom data...")
    
    # Load base model configuration
    checkpoint = torch.load(base_model_path, map_location='cpu')
    config = checkpoint['config']
    
    # Update config for fine-tuning
    config.learning_rate = 1e-5  # Lower learning rate for fine-tuning
    config.num_epochs = 2
    config.batch_size = 4
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    
    # Create dataset
    dataset = MultilingualEmbeddingDataset(custom_data_path, tokenizer, config.max_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    
    # Initialize model
    model = MultilingualEmbeddingModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize trainer
    trainer = MultilingualEmbeddingTrainer(model, config)
    
    # Fine-tune
    trainer.train(dataloader)
    
    logger.info("Fine-tuning completed!")

def export_model_for_production(model_path: str, export_path: str):
    """Export model for production use"""
    logger.info("Exporting model for production...")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    
    # Create model
    model = MultilingualEmbeddingModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Export as TorchScript
    example_input = torch.randint(0, 1000, (1, config.max_length))
    example_mask = torch.ones(1, config.max_length)
    
    try:
        traced_model = torch.jit.trace(model, (example_input, example_mask))
        torch.jit.save(traced_model, export_path)
        logger.info(f"Model exported to {export_path}")
    except Exception as e:
        logger.error(f"Error exporting model: {e}")
        # Save as regular PyTorch model
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config
        }, export_path)
        logger.info(f"Model saved as regular PyTorch model to {export_path}")

def demo_multilingual_search():
    """Demonstrate multilingual search capabilities"""
    logger.info("Demonstrating multilingual search...")
    
    # Sample multilingual corpus
    corpus = [
        # English
        "Machine learning algorithms can automatically improve through experience.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret and understand visual information.",
        "Deep learning uses neural networks with multiple layers to learn complex patterns.",
        
        # German
        "Maschinelles Lernen-Algorithmen können sich automatisch durch Erfahrung verbessern.",
        "Natürliche Sprachverarbeitung ermöglicht es Computern, menschliche Sprache zu verstehen.",
        "Computer Vision ermöglicht es Maschinen, visuelle Informationen zu interpretieren und zu verstehen.",
        "Deep Learning verwendet neuronale Netze mit mehreren Schichten, um komplexe Muster zu lernen.",
        
        # French
        "Les algorithmes d'apprentissage automatique peuvent s'améliorer automatiquement grâce à l'expérience.",
        "Le traitement du langage naturel permet aux ordinateurs de comprendre le langage humain.",
        "La vision par ordinateur permet aux machines d'interpréter et de comprendre les informations visuelles.",
        "L'apprentissage profond utilise des réseaux de neurones à plusieurs couches pour apprendre des modèles complexes.",
        
        # Spanish
        "Los algoritmos de aprendizaje automático pueden mejorar automáticamente a través de la experiencia.",
        "El procesamiento del lenguaje natural permite a las computadoras entender el lenguaje humano.",
        "La visión por computadora permite a las máquinas interpretar y entender información visual.",
        "El aprendizaje profundo utiliza redes neuronales con múltiples capas para aprender patrones complejos.",
        
        # Unrelated content
        "I love cooking Italian pasta with fresh tomatoes and basil.",
        "The weather today is sunny and perfect for outdoor activities.",
        "Music therapy has been shown to reduce stress and improve mental health.",
        "Gardening is a relaxing hobby that connects people with nature."
    ]
    
    # Test queries in different languages
    test_queries = [
        "What is machine learning?",  # English
        "Was ist maschinelles Lernen?",  # German
        "Qu'est-ce que l'apprentissage automatique?",  # French
        "¿Qué es el aprendizaje automático?",  # Spanish
        "How does computer vision work?",  # English
        "Wie funktioniert Computer Vision?",  # German
    ]
    
    try:
        # Load model
        model_path = 'model_checkpoints/best_multilingual_embedding_model.pt'
        if not os.path.exists(model_path):
            logger.warning("Model not found. Please train the model first.")
            return
        
        inference = MultilingualEmbeddingInference(model_path)
        
        # Test each query
        for query in test_queries:
            logger.info(f"\nQuery: '{query}'")
            results = inference.find_most_similar(query, corpus, top_k=3)
            
            for i, (text, score) in enumerate(results):
                logger.info(f"  {i+1}. Score: {score:.4f} - {text[:80]}...")
                
    except Exception as e:
        logger.error(f"Error in demo: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multilingual Embedding Model Training and Inference")
    parser.add_argument('--action', choices=['train', 'inference', 'demo', 'evaluate', 'custom'], 
                       default='train', help='Action to perform')
    parser.add_argument('--model_path', type=str, default='model_checkpoints/best_multilingual_embedding_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, help='Path to custom data')
    parser.add_argument('--export_path', type=str, help='Path to export model')
    
    args = parser.parse_args()
    
    if args.action == 'train':
        main()
    elif args.action == 'inference':
        demo_multilingual_search()
    elif args.action == 'demo':
        demo_multilingual_search()
    elif args.action == 'evaluate':
        if args.data_path:
            evaluate_model_performance(args.model_path, args.data_path)
        else:
            logger.error("Please provide --data_path for evaluation")
    elif args.action == 'custom':
        custom_data_path = create_custom_dataset()
        if args.model_path and os.path.exists(args.model_path):
            fine_tune_on_custom_data(args.model_path, custom_data_path)
        else:
            logger.info("Custom dataset created. Train base model first, then fine-tune.")
    
    logger.info("Script completed!")
