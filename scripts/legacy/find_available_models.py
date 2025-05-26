#!/usr/bin/env python3
"""
利用可能なGGUFモデルを検索
"""

from huggingface_hub import list_repo_files

def list_available_models():
    """利用可能なQwenモデルを一覧表示"""
    
    repos = [
        "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "Qwen/Qwen2.5-7B-Instruct-GGUF", 
        "microsoft/DialoGPT-medium",
        "microsoft/DialoGPT-large"
    ]
    
    for repo in repos:
        try:
            print(f"\n📁 {repo}:")
            files = list_repo_files(repo)
            
            # GGUFファイルのみ表示
            gguf_files = [f for f in files if f.endswith('.gguf')]
            
            if gguf_files:
                for file in gguf_files[:5]:  # 最初の5個だけ表示
                    print(f"  ✅ {file}")
                if len(gguf_files) > 5:
                    print(f"  ... and {len(gguf_files) - 5} more files")
            else:
                print("  ❌ No GGUF files found")
                
        except Exception as e:
            print(f"  ❌ Error accessing {repo}: {e}")

def download_working_model():
    """実際に利用可能なモデルをダウンロード"""
    from huggingface_hub import hf_hub_download
    import os
    
    model_dir = "/Users/mymac/manxo/models/llm"
    os.makedirs(model_dir, exist_ok=True)
    
    # 代替モデルを試す
    alternatives = [
        {
            "repo": "microsoft/DialoGPT-medium", 
            "file": "pytorch_model.bin",
            "size": "1.4GB",
            "description": "DialogGPT Medium (対話特化)"
        },
        {
            "repo": "TheBloke/Llama-2-7B-Chat-GGUF",
            "file": "llama-2-7b-chat.Q4_K_M.gguf", 
            "size": "4.1GB",
            "description": "Llama2 7B Chat Q4"
        },
        {
            "repo": "TheBloke/CodeLlama-7B-Instruct-GGUF",
            "file": "codellama-7b-instruct.Q4_K_M.gguf",
            "size": "4.2GB", 
            "description": "CodeLlama 7B Instruct Q4"
        }
    ]
    
    for alt in alternatives:
        try:
            print(f"\n🔄 {alt['description']} を試行中...")
            print(f"サイズ: {alt['size']}")
            
            model_path = hf_hub_download(
                repo_id=alt["repo"],
                filename=alt["file"],
                local_dir=model_dir
            )
            
            print(f"✅ ダウンロード成功: {model_path}")
            return model_path
            
        except Exception as e:
            print(f"❌ {alt['description']} 失敗: {e}")
            continue
    
    return None

if __name__ == "__main__":
    print("🔍 利用可能なモデルを検索中...")
    list_available_models()
    
    print("\n🚀 実用的なモデルをダウンロード中...")
    model_path = download_working_model()
    
    if model_path:
        print(f"\n🎉 モデル準備完了: {model_path}")
    else:
        print("\n❌ 全ての代替案が失敗しました")