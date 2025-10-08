import torch
import numpy as np
from multilingual_embedding_model import MultilingualEmbeddingInference, EmbeddingConfig
import os

def test_multilingual_model():
    """Test the trained multilingual model"""
    
    model_path = 'model_checkpoints/best_multilingual_embedding_model.pt'
    
    if not os.path.exists(model_path):
        print("❌ Model not found. Please train the model first.")
        print("Run: python run_training.py")
        return
    
    print("🧪 Testing Multilingual Embedding Model")
    print("=" * 40)
    
    try:
        # Load model
        config = EmbeddingConfig()
        inference = MultilingualEmbeddingInference(model_path, config)
        
        # Test cases
        test_cases = [
            {
                "query": "What is machine learning?",
                "language": "English",
                "expected_matches": [
                    "Machine learning is a subset of artificial intelligence",
                    "ML algorithms learn patterns from data",
                    "Supervised learning uses labeled data"
                ]
            },
            {
                "query": "Was ist maschinelles Lernen?",
                "language": "German",
                "expected_matches": [
                    "Machine learning is a subset of artificial intelligence",
                    "ML algorithms learn patterns from data",
                    "Supervised learning uses labeled data"
                ]
            },
            {
                "query": "Qu'est-ce que l'apprentissage automatique?",
                "language": "French",
                "expected_matches": [
                    "Machine learning is a subset of artificial intelligence",
                    "ML algorithms learn patterns from data",
                    "Supervised learning uses labeled data"
                ]
            }
        ]
        
        # Test corpus
        corpus = [
            "Machine learning is a subset of artificial intelligence",
            "ML algorithms learn patterns from data",
            "Supervised learning uses labeled data",
            "I love cooking pasta with tomatoes",
            "The weather is sunny today",
            "Music helps me relax",
            "Sports are great for fitness"
        ]
        
        print("Testing cross-lingual retrieval...")
        
        for test_case in test_cases:
            query = test_case["query"]
            language = test_case["language"]
            
            print(f"\n🔍 Query ({language}): {query}")
            
            # Find most similar
            results = inference.find_most_similar(query, corpus, top_k=3)
            
            print("Top 3 results:")
            for i, (text, score) in enumerate(results):
                print(f"  {i+1}. Score: {score:.4f} - {text}")
            
            # Check if ML-related content is ranked higher
            ml_scores = []
            other_scores = []
            
            for text, score in results:
                if any(keyword in text.lower() for keyword in ['machine', 'learning', 'ml', 'supervised', 'algorithm']):
                    ml_scores.append(score)
                else:
                    other_scores.append(score)
            
            if ml_scores and other_scores:
                avg_ml_score = np.mean(ml_scores)
                avg_other_score = np.mean(other_scores)
                
                if avg_ml_score > avg_other_score:
                    print(f"  ✅ ML content ranked higher (avg: {avg_ml_score:.4f} vs {avg_other_score:.4f})")
                else:
                    print(f"  ⚠️  ML content not ranked higher (avg: {avg_ml_score:.4f} vs {avg_other_score:.4f})")
        
        # Test embedding generation
        print(f"\n📊 Testing embedding generation...")
        sample_texts = [
            "Hello world",
            "Hola mundo", 
            "Bonjour le monde",
            "Hallo Welt"
        ]
        
        embeddings = inference.encode(sample_texts)
        print(f"Generated embeddings shape: {embeddings.shape}")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        
        # Test similarity between similar phrases
        similarities = []
        for i in range(len(sample_texts)):
            for j in range(i+1, len(sample_texts)):
                sim = inference.similarity(sample_texts[i], sample_texts[j])
                similarities.append(sim)
                print(f"Similarity '{sample_texts[i]}' vs '{sample_texts[j]}': {sim:.4f}")
        
        avg_similarity = np.mean(similarities)
        print(f"Average cross-lingual similarity: {avg_similarity:.4f}")
        
        if avg_similarity > 0.7:
            print("✅ Good cross-lingual alignment!")
        elif avg_similarity > 0.5:
            print("⚠️  Moderate cross-lingual alignment")
        else:
            print("❌ Poor cross-lingual alignment")
        
        print("\n🎉 Model testing completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multilingual_model()