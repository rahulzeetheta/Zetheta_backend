import os
import requests
from pathlib import Path
import sys

def download_file_with_progress(url, filename):
    """Download a file from URL with progress bar"""
    print(f"Downloading {filename}...")
    print(f"URL: {url}")
    print("This file is approximately 4GB, please be patient...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    if total_size == 0:
        print("Warning: Could not determine file size")
    
    with open(filename, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}MB / {mb_total:.1f}MB)", end='', flush=True)
                else:
                    mb_downloaded = downloaded / (1024 * 1024)
                    print(f"\rDownloaded: {mb_downloaded:.1f}MB", end='', flush=True)
    
    print(f"\nDownloaded {filename} successfully!")
    print(f"File size: {os.path.getsize(filename) / (1024*1024*1024):.2f}GB")

def download_llama_model():
    """Download the Llama-2-7B-Chat model in GGUF format"""
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print(f"Models directory: {models_dir.absolute()}")
    
    # Model file details
    model_filename = "llama-2-7b-chat.Q4_0.gguf"
    model_path = models_dir / model_filename
    
    # Check if file already exists
    if model_path.exists():
        file_size = model_path.stat().st_size / (1024 * 1024 * 1024)
        print(f"Model file already exists: {model_path}")
        print(f"File size: {file_size:.2f}GB")
        response = input("Do you want to re-download? (y/N): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return
    
    # Direct download URL (no login required)
    url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf"
    
    try:
        download_file_with_progress(url, str(model_path))
        print(f"\n✅ Model downloaded successfully to: {model_path}")
        print("You can now run: streamlit run app.py")
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nAlternative download methods:")
        print("1. Visit: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF")
        print("2. Download the file: llama-2-7b-chat.Q4_0.gguf")
        print("3. Place it in the 'models/' directory")

def main():
    print("Llama-2-7B-Chat Model Downloader")
    print("="*50)
    print("This script will download the Llama model (~4GB)")
    print("Make sure you have enough disk space and a stable internet connection.")
    print("="*50)
    
    # Check available disk space
    models_dir = Path("models")
    if models_dir.exists():
        import shutil
        total, used, free = shutil.disk_usage(models_dir)
        free_gb = free / (1024**3)
        print(f"Available disk space: {free_gb:.1f}GB")
        if free_gb < 5:
            print("⚠️  Warning: Less than 5GB available. Download may fail.")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("Download cancelled.")
                return
    
    download_llama_model()

if __name__ == "__main__":
    main() 