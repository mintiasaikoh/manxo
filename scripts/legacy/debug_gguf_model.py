#!/usr/bin/env python3
"""
GGUFモデルの詳細デバッグ
"""

import os
from llama_cpp import Llama

def test_simple_load():
    """最もシンプルな設定でロードテスト"""
    
    model_path = "/Users/mymac/manxo/models/llm/qwen2.5-7b-instruct-q4_k_m.gguf"
    
    if not os.path.exists(model_path):
        print(f"❌ ファイルが存在しません: {model_path}")
        return False
    
    file_size = os.path.getsize(model_path) / (1024**3)
    print(f"📁 ファイル確認: {file_size:.1f}GB")
    
    try:
        print("🧠 最小設定でロード試行...")
        model = Llama(
            model_path=model_path,
            n_ctx=256,      # 最小コンテキスト
            n_batch=8,      # 最小バッチ
            n_gpu_layers=0, # CPU実行
            use_mlock=False,
            use_mmap=True,
            verbose=True    # 詳細ログ
        )
        
        print("✅ モデルロード成功!")
        
        # 最小テスト
        response = model("Hello", max_tokens=5, echo=False)
        print(f"📝 応答: {response}")
        
        del model
        return True
        
    except Exception as e:
        print(f"❌ ロードエラー詳細: {e}")
        return False

def test_alternative_models():
    """代替モデルでテスト"""
    
    model_dir = "/Users/mymac/manxo/models/llm"
    
    # 利用可能なGGUFファイルを全て試す
    gguf_files = [f for f in os.listdir(model_dir) if f.endswith('.gguf')]
    
    print(f"📋 利用可能なGGUFファイル: {len(gguf_files)}個")
    
    for gguf_file in gguf_files:
        if gguf_file == "qwen2.5-14b-instruct-q4_k_m.gguf":
            continue  # 0GBファイルはスキップ
            
        model_path = os.path.join(model_dir, gguf_file)
        file_size = os.path.getsize(model_path) / (1024**3)
        
        print(f"\n🧪 テスト: {gguf_file} ({file_size:.1f}GB)")
        
        if file_size < 0.1:  # 100MB未満はスキップ
            print("⏸️ ファイルサイズが小さすぎます")
            continue
        
        try:
            model = Llama(
                model_path=model_path,
                n_ctx=256,
                n_gpu_layers=0,
                verbose=False
            )
            
            # シンプルテスト
            response = model("Create Max/MSP reverb:", max_tokens=20, echo=False)
            print(f"✅ 成功: {response['choices'][0]['text'][:50]}...")
            
            del model
            import gc
            gc.collect()
            
            return model_path  # 成功したモデルパスを返す
            
        except Exception as e:
            print(f"❌ 失敗: {str(e)[:100]}...")
            continue
    
    return None

def download_confirmed_working_model():
    """確実に動作するモデルをダウンロード"""
    
    from huggingface_hub import hf_hub_download
    model_dir = "/Users/mymac/manxo/models/llm"
    
    # 確実に動作する小さなモデル
    working_models = [
        {
            "repo": "microsoft/DialoGPT-small",
            "file": "pytorch_model.bin",
            "description": "DialoGPT Small (117MB)",
            "use_transformers": True
        },
        {
            "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
            "file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", 
            "description": "TinyLlama 1.1B (669MB)",
            "use_transformers": False
        }
    ]
    
    for model_config in working_models:
        try:
            print(f"⬇️ {model_config['description']} をダウンロード中...")
            
            model_path = hf_hub_download(
                repo_id=model_config["repo"],
                filename=model_config["file"],
                local_dir=model_dir
            )
            
            file_size = os.path.getsize(model_path) / (1024**2)  # MB
            print(f"✅ ダウンロード完了: {file_size:.0f}MB")
            
            if not model_config["use_transformers"]:
                # GGUF形式なのでllama-cppでテスト
                model = Llama(model_path=model_path, n_ctx=256, verbose=False)
                response = model("Hello", max_tokens=10, echo=False)
                print(f"✅ 動作確認: {response['choices'][0]['text']}")
                del model
            
            return model_path
            
        except Exception as e:
            print(f"❌ {model_config['description']} 失敗: {e}")
            continue
    
    return None

if __name__ == "__main__":
    print("🔧 GGUFモデルの詳細デバッグ開始...")
    
    # Step 1: 結合したQwenモデルをテスト
    print("\\n=== Step 1: Qwen結合モデルテスト ===")
    qwen_success = test_simple_load()
    
    if qwen_success:
        print("🎉 Qwenモデル正常動作!")
    else:
        # Step 2: 代替GGUFファイルをテスト
        print("\\n=== Step 2: 代替GGUFファイルテスト ===")
        working_model = test_alternative_models()
        
        if working_model:
            print(f"🎉 動作するモデル発見: {os.path.basename(working_model)}")
        else:
            # Step 3: 確実に動作するモデルをダウンロード
            print("\\n=== Step 3: 動作確認済みモデルダウンロード ===")
            fallback_model = download_confirmed_working_model()
            
            if fallback_model:
                print(f"🎉 フォールバックモデル準備完了: {os.path.basename(fallback_model)}")
            else:
                print("❌ 全ての選択肢が失敗しました")
    
    print("\\n🔧 デバッグ完了")