#!/usr/bin/env python3
"""
実用的なLLMモデルを確実にセットアップ
"""

from huggingface_hub import hf_hub_download, list_repo_files
import os

def find_and_download_qwen():
    """利用可能なQwenモデルを見つけてダウンロード"""
    
    model_dir = "/Users/mymac/manxo/models/llm"
    os.makedirs(model_dir, exist_ok=True)
    
    # Qwen 7Bのファイルを探す
    try:
        print("🔍 Qwen2.5-7B の利用可能ファイルを確認中...")
        files = list_repo_files("Qwen/Qwen2.5-7B-Instruct-GGUF")
        
        # Q4量子化ファイルを探す
        q4_files = [f for f in files if 'q4' in f.lower() and f.endswith('.gguf')]
        print(f"📋 利用可能なQ4ファイル: {q4_files[:3]}")
        
        if q4_files:
            # 最初のQ4ファイルをダウンロード
            target_file = q4_files[0]
            print(f"⬇️ ダウンロード中: {target_file}")
            
            model_path = hf_hub_download(
                repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
                filename=target_file,
                local_dir=model_dir
            )
            
            file_size = os.path.getsize(model_path) / (1024**3)
            print(f"✅ Qwen ダウンロード完了: {file_size:.1f}GB")
            return model_path
            
    except Exception as e:
        print(f"❌ Qwen ダウンロード失敗: {e}")
    
    return None

def download_alternative_model():
    """代替モデルをダウンロード"""
    
    model_dir = "/Users/mymac/manxo/models/llm" 
    
    # 確実にダウンロードできるモデル
    alternatives = [
        {
            "repo": "TheBloke/Llama-2-7B-Chat-GGUF",
            "file": "llama-2-7b-chat.Q4_K_M.gguf",
            "description": "Llama2 7B Chat (4.1GB)"
        },
        {
            "repo": "TheBloke/CodeLlama-7B-Instruct-GGUF", 
            "file": "codellama-7b-instruct.Q4_K_M.gguf",
            "description": "CodeLlama 7B (4.2GB) - プログラミング特化"
        }
    ]
    
    for alt in alternatives:
        try:
            print(f"⬇️ {alt['description']} をダウンロード中...")
            
            model_path = hf_hub_download(
                repo_id=alt["repo"],
                filename=alt["file"],
                local_dir=model_dir
            )
            
            file_size = os.path.getsize(model_path) / (1024**3)
            print(f"✅ {alt['description']} 完了: {file_size:.1f}GB")
            return model_path
            
        except Exception as e:
            print(f"❌ {alt['description']} 失敗: {e}")
            continue
    
    return None

def test_any_model(model_path):
    """任意のGGUFモデルをテスト"""
    if not model_path or not os.path.exists(model_path):
        return False
    
    try:
        from llama_cpp import Llama
        
        print(f"🧠 モデルをロード中: {os.path.basename(model_path)}")
        model = Llama(
            model_path=model_path,
            n_ctx=512,  # 小さなコンテキスト
            n_gpu_layers=0,
            verbose=False
        )
        
        # シンプルなテスト
        response = model(
            "Create a Max/MSP reverb effect using:",
            max_tokens=50,
            echo=False,
            temperature=0.3
        )
        
        print("✅ モデル動作確認成功!")
        print(f"📝 テスト応答: {response['choices'][0]['text'][:100]}...")
        
        del model
        import gc
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ モデルテスト失敗: {e}")
        return False

def create_llm_interface():
    """LLMインターフェースクラスを作成"""
    
    interface_code = '''#!/usr/bin/env python3
"""
Max/MSP パッチ生成用 LLM インターフェース
"""

import os
import json
from llama_cpp import Llama

class MaxPatchLLM:
    def __init__(self, model_path=None):
        if model_path is None:
            # 利用可能なモデルを自動検出
            model_dir = "/Users/mymac/manxo/models/llm"
            gguf_files = [f for f in os.listdir(model_dir) if f.endswith('.gguf')]
            if gguf_files:
                model_path = os.path.join(model_dir, gguf_files[0])
            else:
                raise FileNotFoundError("GGUFモデルが見つかりません")
        
        self.model_path = model_path
        self.model = None
    
    def load_model(self):
        """必要時にモデルをロード"""
        if self.model is None:
            print(f"🧠 LLMロード中: {os.path.basename(self.model_path)}")
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=2048,
                n_gpu_layers=0,
                verbose=False
            )
    
    def unload_model(self):
        """メモリ節約のためモデル解放"""
        if self.model:
            del self.model
            self.model = None
            import gc
            gc.collect()
    
    def parse_intent(self, user_input):
        """ユーザー入力を解析してMax/MSPの意図を抽出"""
        self.load_model()
        
        prompt = f"""あなたはMax/MSPエキスパートです。以下の要求を分析してJSON形式で回答してください。

要求: "{user_input}"

回答形式:
{{
  "category": "effect|synth|utility|sequencer",
  "subcategory": "具体的なタイプ",
  "objects": ["必要なMax/MSPオブジェクト"],
  "parameters": {{"パラメーター名": "値"}},
  "complexity": 1-5
}}

JSON:"""
        
        try:
            response = self.model(
                prompt,
                max_tokens=300,
                stop=["\\n\\n", "```"],
                echo=False,
                temperature=0.1
            )
            
            result = response['choices'][0]['text'].strip()
            
            # JSONを抽出
            if '{' in result and '}' in result:
                start = result.find('{')
                end = result.rfind('}') + 1
                json_str = result[start:end]
                return json.loads(json_str)
            else:
                # フォールバック: 基本的な解析
                return self._fallback_parse(user_input)
                
        except Exception as e:
            print(f"⚠️ LLM解析エラー: {e}")
            return self._fallback_parse(user_input)
    
    def _fallback_parse(self, user_input):
        """LLM失敗時のフォールバック解析"""
        keywords = {
            'reverb': {'category': 'effect', 'subcategory': 'reverb', 'objects': ['adc~', 'freeverb~', 'dac~']},
            'delay': {'category': 'effect', 'subcategory': 'delay', 'objects': ['adc~', 'tapin~', 'tapout~', 'dac~']},
            'filter': {'category': 'effect', 'subcategory': 'filter', 'objects': ['adc~', 'lores~', 'dac~']},
            'oscillator': {'category': 'synth', 'subcategory': 'oscillator', 'objects': ['phasor~', 'dac~']},
            'synth': {'category': 'synth', 'subcategory': 'basic', 'objects': ['phasor~', '*~', 'dac~']}
        }
        
        user_lower = user_input.lower()
        for keyword, config in keywords.items():
            if keyword in user_lower:
                config['complexity'] = 2
                config['parameters'] = {}
                return config
        
        # デフォルト
        return {
            'category': 'effect',
            'subcategory': 'basic',
            'objects': ['adc~', 'dac~'],
            'parameters': {},
            'complexity': 1
        }

if __name__ == "__main__":
    # テスト実行
    llm = MaxPatchLLM()
    
    test_inputs = [
        "ステレオリバーブエフェクトを作って",
        "FM合成でベル音を作りたい",
        "ローパスフィルターエフェクト"
    ]
    
    for test_input in test_inputs:
        print(f"\\n📝 テスト: {test_input}")
        result = llm.parse_intent(test_input)
        print(f"📊 結果: {result}")
    
    llm.unload_model()
    print("\\n✅ LLMインターフェーステスト完了!")
'''
    
    with open("/Users/mymac/manxo/scripts/max_patch_llm.py", "w") as f:
        f.write(interface_code)
    
    print("📝 LLMインターフェース作成完了: scripts/max_patch_llm.py")

if __name__ == "__main__":
    print("🚀 実用的LLMセットアップ開始...")
    
    # Step 1: Qwenを試す
    model_path = find_and_download_qwen()
    
    # Step 2: 失敗したら代替モデル
    if not model_path:
        print("🔄 代替モデルにフォールバック...")
        model_path = download_alternative_model()
    
    # Step 3: テスト実行
    if model_path and test_any_model(model_path):
        print("\\n🎉 LLMセットアップ成功!")
        
        # Step 4: インターフェース作成
        create_llm_interface()
        
        print(f"📍 モデルパス: {model_path}")
        print("📝 使用方法:")
        print("  python scripts/max_patch_llm.py")
        
    else:
        print("\\n❌ 全てのモデルセットアップが失敗しました")