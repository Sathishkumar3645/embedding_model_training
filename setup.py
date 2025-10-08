from setuptools import setup, find_packages

setup(
    name="multilingual-embedding-model",
    version="1.0.0",
    description="Multilingual Embedding Model for RAG Applications",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

# install_dependencies.py
import subprocess
import sys
import platform

def install_dependencies():
    """Install dependencies based on the operating system"""
    system = platform.system()
    python_version = sys.version_info
    
    print(f"Detected {system} with Python {python_version.major}.{python_version.minor}")
    
    # Base requirements
    base_requirements = [
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0",
    ]
    
    # Install base requirements
    for req in base_requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✓ Installed {req}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {req}")
    
    # Install PyTorch based on system
    if system == "Darwin":  # macOS
        print("Installing PyTorch for macOS (with MPS support)...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ])
            print("✓ PyTorch installed for macOS")
        except subprocess.CalledProcessError:
            print("✗ Failed to install PyTorch for macOS")
    
    elif system == "Windows":
        print("Installing PyTorch for Windows...")
        try:
            # Try CUDA first
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ])
            print("✓ PyTorch installed for Windows (CUDA)")
        except subprocess.CalledProcessError:
            try:
                # Fall back to CPU
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cpu"
                ])
                print("✓ PyTorch installed for Windows (CPU)")
            except subprocess.CalledProcessError:
                print("✗ Failed to install PyTorch for Windows")
    
    else:  # Linux
        print("Installing PyTorch for Linux...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ])
            print("✓ PyTorch installed for Linux")
        except subprocess.CalledProcessError:
            print("✗ Failed to install PyTorch for Linux")
    
    print("\nInstallation completed!")
    print("Run: python multilingual_embedding_model.py --action train")

if __name__ == "__main__":
    install_dependencies()