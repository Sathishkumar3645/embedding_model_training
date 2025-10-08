import sys
import os

def run_training():
    """Run the complete training pipeline"""
    
    print("🚀 Starting Multilingual Embedding Model Training")
    print("=" * 50)
    
    # Import the main module
    try:
        from multilingual_embedding_model import main, setup_environment
        
        # Setup environment
        setup_environment()
        
        # Run training
        main()
        
        print("\n✅ Training completed successfully!")
        print("Model saved to: model_checkpoints/best_multilingual_embedding_model.pt")
        print("\nTo test the model, run:")
        print("python multilingual_embedding_model.py --action demo")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed.")
        print("Run: python install_dependencies.py")
    except Exception as e:
        print(f"❌ Training error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_training()