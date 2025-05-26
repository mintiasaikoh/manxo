#!/usr/bin/env python3
"""
分割されたGGUFファイルを結合
"""

import os
import glob

def merge_split_gguf():
    """分割されたGGUFファイルを結合"""
    
    model_dir = "/Users/mymac/manxo/models/llm"
    
    # 分割ファイルを検索
    pattern = os.path.join(model_dir, "qwen2.5-7b-instruct-q4_k_m-*-of-*.gguf")
    split_files = sorted(glob.glob(pattern))
    
    print(f"📁 分割ファイル発見: {len(split_files)}個")
    for f in split_files:
        size_gb = os.path.getsize(f) / (1024**3)
        print(f"  - {os.path.basename(f)}: {size_gb:.1f}GB")
    
    if len(split_files) < 2:
        print("❌ 分割ファイルが不完全です")
        return None
    
    # 結合後のファイル名
    merged_file = os.path.join(model_dir, "qwen2.5-7b-instruct-q4_k_m.gguf")
    
    print(f"🔗 ファイルを結合中: {os.path.basename(merged_file)}")
    
    try:
        with open(merged_file, 'wb') as outfile:
            for split_file in split_files:
                print(f"  📥 追加中: {os.path.basename(split_file)}")
                with open(split_file, 'rb') as infile:
                    # 64MBずつ読み込んで結合
                    while True:
                        chunk = infile.read(64 * 1024 * 1024)
                        if not chunk:
                            break
                        outfile.write(chunk)
        
        # 結合ファイルのサイズ確認
        merged_size = os.path.getsize(merged_file) / (1024**3)
        print(f"✅ 結合完了: {merged_size:.1f}GB")
        
        return merged_file
        
    except Exception as e:
        print(f"❌ 結合エラー: {e}")
        return None

def test_merged_model(model_path):
    """結合したモデルをテスト"""
    
    if not model_path or not os.path.exists(model_path):
        return False
    
    try:
        from llama_cpp import Llama
        
        print(f"🧠 結合モデルをテスト: {os.path.basename(model_path)}")
        
        model = Llama(
            model_path=model_path,
            n_ctx=1024,
            n_gpu_layers=0,
            verbose=False
        )
        
        print("💬 Max/MSP特化テスト実行中...")
        
        # テストプロンプト
        test_prompt = """あなたはMax/MSPエキスパートです。以下の要求を分析してJSON形式で回答してください。

要求: "ステレオリバーブエフェクトを作って"

JSON形式で回答:
{
  "category": "effect",
  "subcategory": "reverb",
  "objects": ["adc~", "freeverb~", "dac~"],
  "parameters": {"roomsize": 0.8},
  "complexity": 2
}

要求: "ディストーションエフェクトを作って"

JSON:"""
        
        response = model(
            test_prompt,
            max_tokens=150,
            stop=["\n\n"],
            echo=False,
            temperature=0.1
        )
        
        result = response['choices'][0]['text']
        print("✅ モデルテスト成功!")
        print(f"📝 JSON応答:")
        print(result)
        
        # メモリ解放
        del model
        import gc
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ モデルテスト失敗: {e}")
        return False

def cleanup_split_files():
    """分割ファイルを削除（オプション）"""
    
    model_dir = "/Users/mymac/manxo/models/llm"
    pattern = os.path.join(model_dir, "qwen2.5-7b-instruct-q4_k_m-*-of-*.gguf")
    split_files = glob.glob(pattern)
    
    print(f"🗑️ 分割ファイル{len(split_files)}個を削除しますか？ (y/N): ", end="")
    response = input().lower()
    
    if response == 'y':
        for f in split_files:
            os.remove(f)
            print(f"  🗑️ 削除: {os.path.basename(f)}")
        print("✅ 分割ファイル削除完了")
    else:
        print("⏸️ 分割ファイルは保持されます")

if __name__ == "__main__":
    print("🔗 分割GGUFファイルの結合開始...")
    
    # ファイルを結合
    merged_file = merge_split_gguf()
    
    if merged_file:
        # テスト実行
        if test_merged_model(merged_file):
            print("\n🎉 Qwen2.5-7B 結合・テスト完了!")
            print(f"📍 最終モデル: {merged_file}")
            
            # 分割ファイル削除オプション
            cleanup_split_files()
            
        else:
            print("\n⚠️ 結合成功、テスト失敗")
    else:
        print("\n❌ ファイル結合失敗")