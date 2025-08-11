import os
import requests
from pathlib import Path
import zipfile
import shutil

def download_file(url, filename):
    """Download a file from URL with progress bar"""
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f:
        if total_size == 0:
            f.write(response.content)
        else:
            downloaded = 0
            total_size = int(total_size)
            for data in response.iter_content(chunk_size=8192):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total_size)
                print(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded}/{total_size} bytes", end='')
    print(f"\nDownloaded {filename} successfully!")

def download_embeddings_model():
    """Download the sentence-transformers model locally"""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = os.path.join(cache_dir, "models--sentence-transformers--all-MiniLM-L6-v2")
    
    # Create directories if they don't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Download the model files
    base_url = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"
    files_to_download = [
        "config.json",
        "pytorch_model.bin",
        "sentence_bert_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt"
    ]
    
    for file in files_to_download:
        url = f"{base_url}/{file}"
        local_path = os.path.join(model_dir, file)
        if not os.path.exists(local_path):
            download_file(url, local_path)
        else:
            print(f"{file} already exists, skipping...")

def download_llama_model():
    """Provide instructions for downloading the Llama model"""
    print("\n" + "="*60)
    print("LLAMA MODEL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("The app requires the Llama-2-7B-Chat model in GGUF format.")
    print("You have two options:")
    print("\nOption 1: Download from Hugging Face (requires login)")
    print("1. Go to: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF")
    print("2. Download the file: llama-2-7b-chat.Q4_0.gguf")
    print("3. Place it in the 'models/' directory")
    
    print("\nOption 2: Download from direct link (no login required)")
    print("1. Create a 'models' directory if it doesn't exist")
    print("2. Download from: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf")
    print("3. Save it as 'models/llama-2-7b-chat.Q4_0.gguf'")
    
    print("\nThe model file is approximately 4GB.")
    print("="*60)

def create_models_directory():
    """Create the models directory if it doesn't exist"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print(f"Models directory created/verified: {models_dir.absolute()}")

def main():
    print("Financial Chatbot Model Downloader")
    print("="*40)
    
    # Create models directory
    create_models_directory()
    
    # Download embeddings model
    print("\n1. Downloading embeddings model...")
    download_embeddings_model()
    
    # Provide instructions for Llama model
    print("\n2. Llama model download instructions...")
    download_llama_model()
    
    print("\n" + "="*40)
    print("DOWNLOAD COMPLETE!")
    print("="*40)
    print("Next steps:")
    print("1. Download the Llama model following the instructions above")
    print("2. Run: streamlit run app.py")
    print("="*40)

if __name__ == "__main__":
    main() 