#!/usr/bin/env python3
"""
正確なファイル名でQwenモデルをダウンロード
"""

from huggingface_hub import hf_hub_download, list_repo_files
import os
import subprocess

def download_qwen_with_cli():
    """huggingface-cliを使ってダウンロード"""
    
    model_dir = "/Users/mymac/manxo/models/llm"
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        print("🚀 huggingface-cliでQwen2.5-7B Q4をダウンロード中...")
        
        # huggingface-cliを使用
        cmd = [
            "huggingface-cli", "download", 
            "Qwen/Qwen2.5-7B-Instruct-GGUF",
            "--include", "*q4_k_m*.gguf",
            "--local-dir", model_dir
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print("✅ huggingface-cliダウンロード成功!")
            
            # ダウンロードされたファイルを確認
            gguf_files = [f for f in os.listdir(model_dir) if f.endswith('.gguf')]
            if gguf_files:
                model_path = os.path.join(model_dir, gguf_files[0])
                file_size = os.path.getsize(model_path) / (1024**3)
                print(f"📁 ファイル: {gguf_files[0]} ({file_size:.1f}GB)")
                return model_path
        else:
            print(f"❌ huggingface-cli失敗: {result.stderr}")
            
    except Exception as e:
        print(f"❌ huggingface-cli エラー: {e}")
    
    return None

def download_qwen_python_api():
    """Python APIでダウンロード（フォールバック）"""
    
    model_dir = "/Users/mymac/manxo/models/llm"
    
    try:
        print("🔄 Python APIでダウンロード中...")
        
        # 利用可能なファイルを確認
        files = list_repo_files("Qwen/Qwen2.5-7B-Instruct-GGUF")
        q4_files = [f for f in files if 'q4_k_m' in f and f.endswith('.gguf')]
        
        print(f"📋 利用可能なQ4ファイル: {q4_files}")
        
        if q4_files:
            target_file = q4_files[0]  # 最初のファイルを選択
            print(f"⬇️ ダウンロード: {target_file}")
            
            model_path = hf_hub_download(
                repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
                filename=target_file,
                local_dir=model_dir
            )
            
            file_size = os.path.getsize(model_path) / (1024**3)
            print(f"✅ ダウンロード完了: {file_size:.1f}GB")
            return model_path
            
    except Exception as e:
        print(f"❌ Python API エラー: {e}")
    
    return None

def download_bartowski_alternative():
    """bartowskiの代替リポジトリからダウンロード"""
    
    model_dir = "/Users/mymac/manxo/models/llm"
    
    try:
        print("🔄 Bartowski版をダウンロード中...")
        
        model_path = hf_hub_download(
            repo_id="bartowski/Qwen2.5-7B-Instruct-GGUF",
            filename="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            local_dir=model_dir
        )
        
        file_size = os.path.getsize(model_path) / (1024**3)
        print(f"✅ Bartowski版ダウンロード完了: {file_size:.1f}GB")
        return model_path
        
    except Exception as e:
        print(f"❌ Bartowski版エラー: {e}")
    
    return None

def test_downloaded_model(model_path):
    """ダウンロードしたモデルをテスト"""
    
    if not model_path or not os.path.exists(model_path):
        return False
    
    try:
        from llama_cpp import Llama
        
        print(f"🧠 モデルテスト: {os.path.basename(model_path)}")
        model = Llama(
            model_path=model_path,
            n_ctx=512,
            n_gpu_layers=0,
            verbose=False
        )
        
        # Max/MSP特化テスト
        response = model(
            "Max/MSPでリバーブを作るオブジェクトは？",
            max_tokens=30,
            echo=False
        )
        
        print("✅ モデル動作確認!")
        print(f"📝 応答例: {response['choices'][0]['text'][:100]}")
        
        del model
        import gc
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ モデルテスト失敗: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Qwen2.5-7B Q4量子化モデルのダウンロード開始...")
    
    model_path = None
    
    # 方法1: huggingface-cli
    model_path = download_qwen_with_cli()
    
    # 方法2: Python API（フォールバック）
    if not model_path:
        model_path = download_qwen_python_api()
    
    # 方法3: Bartowski版（最終手段）
    if not model_path:
        model_path = download_bartowski_alternative()
    
    # テスト実行
    if model_path and test_downloaded_model(model_path):
        print("\n🎉 Qwen2.5-7B セットアップ完了!")
        print(f"📍 モデルパス: {model_path}")
        print("🔧 次のステップ: パッチ生成システムの統合")
    else:
        print("\n❌ 全ての方法が失敗しました")
        print("💡 推奨: インターネット接続とHuggingFaceへのアクセスを確認してください")