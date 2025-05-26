#!/usr/bin/env python3
"""
13B LLMモデルをダウンロード
"""

import os
from huggingface_hub import hf_hub_download

def download_qwen_model():
    """Qwen2.5-14B INT4量子化モデルをダウンロード"""
    
    model_dir = "/Users/mymac/manxo/models/llm"
    os.makedirs(model_dir, exist_ok=True)
    
    print("Qwen2.5-14B INT4量子化モデルをダウンロード中...")
    print("ファイルサイズ: 約8.5GB")
    print("予想時間: 10-20分")
    
    try:
        # Qwen2.5-14B-Instruct GGUF形式をダウンロード
        model_path = hf_hub_download(
            repo_id="Qwen/Qwen2.5-14B-Instruct-GGUF",
            filename="qwen2.5-14b-instruct-q4_k_m.gguf",
            local_dir=model_dir,
            local_dir_use_symlinks=False  # シンボリックリンクを使わない
        )
        
        print(f"✅ ダウンロード完了: {model_path}")
        
        # ファイルサイズを確認
        file_size = os.path.getsize(model_path) / (1024**3)  # GB
        print(f"📊 ファイルサイズ: {file_size:.2f} GB")
        
        return model_path
        
    except Exception as e:
        print(f"❌ ダウンロードエラー: {e}")
        
        # フォールバック: 別の量子化モデルを試す
        print("\n🔄 フォールバック: より小さいモデルを試行...")
        try:
            model_path = hf_hub_download(
                repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF", 
                filename="qwen2.5-7b-instruct-q4_k_m.gguf",
                local_dir=model_dir,
                local_dir_use_symlinks=False
            )
            print(f"✅ 7Bモデルのダウンロード完了: {model_path}")
            return model_path
            
        except Exception as e2:
            print(f"❌ フォールバックも失敗: {e2}")
            return None

def test_model(model_path):
    """ダウンロードしたモデルをテスト"""
    if not model_path or not os.path.exists(model_path):
        print("❌ モデルファイルが見つかりません")
        return False
    
    try:
        from llama_cpp import Llama
        
        print("🧠 モデルをロード中...")
        model = Llama(
            model_path=model_path,
            n_ctx=512,  # 小さなコンテキストでテスト
            n_gpu_layers=0,  # CPU使用
            verbose=False
        )
        
        print("💬 テスト生成中...")
        response = model(
            "Max/MSPでリバーブエフェクトを作るには？",
            max_tokens=100,
            stop=["。", "\n\n"],
            echo=False
        )
        
        print("✅ モデルテスト成功!")
        print(f"📝 応答: {response['choices'][0]['text']}")
        
        # メモリ解放
        del model
        import gc
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ モデルテストエラー: {e}")
        return False

if __name__ == "__main__":
    print("🚀 13B LLMモデルのセットアップを開始...")
    
    # モデルをダウンロード
    model_path = download_qwen_model()
    
    if model_path:
        # モデルをテスト
        if test_model(model_path):
            print("\n🎉 セットアップ完了!")
            print(f"モデルパス: {model_path}")
        else:
            print("\n⚠️ ダウンロードは成功しましたが、モデルテストに失敗しました")
    else:
        print("\n❌ セットアップ失敗")