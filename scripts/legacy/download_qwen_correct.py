#!/usr/bin/env python3
"""
正しいQwenモデルをダウンロード
"""

from huggingface_hub import hf_hub_download
import os

def download_qwen_q4():
    """Qwen 7B Q4量子化モデルをダウンロード"""
    
    model_dir = "/Users/mymac/manxo/models/llm"
    os.makedirs(model_dir, exist_ok=True)
    
    print("🚀 Qwen2.5-7B Q4量子化モデルをダウンロード中...")
    print("ファイルサイズ: 約4.4GB")
    
    try:
        model_path = hf_hub_download(
            repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
            filename="qwen2.5-7b-instruct-q4_k_m.gguf",
            local_dir=model_dir
        )
        
        print(f"✅ ダウンロード完了: {model_path}")
        
        # ファイルサイズ確認
        file_size = os.path.getsize(model_path) / (1024**3)
        print(f"📊 ファイルサイズ: {file_size:.2f} GB")
        
        return model_path
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None

def test_qwen_model(model_path):
    """Qwenモデルをテスト"""
    if not model_path or not os.path.exists(model_path):
        return False
    
    try:
        from llama_cpp import Llama
        
        print("🧠 Qwenモデルをロード中...")
        model = Llama(
            model_path=model_path,
            n_ctx=1024,
            n_gpu_layers=0,
            verbose=False
        )
        
        print("💬 Max/MSP特化テスト中...")
        
        # Max/MSP特化プロンプト
        prompt = """あなたはMax/MSPエキスパートです。以下の要求を分析してください。

要求: "ステレオリバーブエフェクトを作って"

以下の形式で回答してください:
{
  "category": "effect",
  "subcategory": "reverb", 
  "objects": ["adc~", "freeverb~", "dac~"],
  "parameters": {"roomsize": 0.8, "damp": 0.5},
  "complexity": 2
}"""
        
        response = model(
            prompt,
            max_tokens=200,
            stop=["}"],
            echo=False,
            temperature=0.1
        )
        
        result = response['choices'][0]['text'] + "}"
        print("✅ Qwenテスト成功!")
        print(f"📝 JSON応答: {result}")
        
        # メモリ解放
        del model
        import gc
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ Qwenテストエラー: {e}")
        return False

if __name__ == "__main__":
    # Qwen 7B Q4をダウンロード
    model_path = download_qwen_q4()
    
    if model_path:
        # テスト実行
        if test_qwen_model(model_path):
            print("\n🎉 Qwen2.5-7B セットアップ完了!")
            print("Max/MSPパッチ生成の準備ができました。")
        else:
            print("\n⚠️ ダウンロード成功、テスト失敗")
    else:
        print("\n❌ Qwenダウンロード失敗")