#!/usr/bin/env python3
"""
最適化されたMax/MSP LLMインターフェース（高速版）
"""

import os
import json
import re
from llama_cpp import Llama

class OptimizedMaxPatchLLM:
    def __init__(self):
        # 動作確認済みモデルパス
        self.model_path = "/Users/mymac/manxo/models/llm/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
        self.model = None
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {self.model_path}")
    
    def load_model_optimized(self):
        """最適化された設定でモデルロード"""
        if self.model is None:
            print(f"🚀 最適化設定でLLMロード中...")
            
            self.model = Llama(
                model_path=self.model_path,
                
                # 🔧 高速化設定
                n_ctx=512,          # ← 小さなコンテキスト（高速）
                n_batch=8,          # ← 小さなバッチサイズ
                n_threads=4,        # ← スレッド数指定
                n_gpu_layers=0,     # ← とりあえずCPU（安定性優先）
                
                # 📈 メモリ最適化
                use_mmap=True,      # ← メモリマッピング有効
                use_mlock=False,    # ← メモリロック無効（柔軟性優先）
                
                # 🔇 ログ設定
                verbose=False,
                
                # ⚡ 追加最適化
                f16_kv=True,        # ← KVキャッシュをFP16で（メモリ節約）
                logits_all=False,   # ← 全ロジット不要（高速化）
            )
            print("✅ モデルロード完了")
    
    def generate_fast(self, prompt, max_tokens=150):
        """高速生成（最小設定）"""
        self.load_model_optimized()
        
        print(f"💬 高速生成開始...")
        start_time = __import__('time').time()
        
        try:
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                
                # 🚀 高速化パラメーター
                temperature=0.1,    # ← 低温度（決定的）
                top_p=0.9,         # ← nucleus sampling
                repeat_penalty=1.1, # ← 繰り返し防止
                
                # ⏹️ 停止条件
                stop=["\\n\\n", "```", "Response:", "Human:"],
                
                # 🔇 エコー無効
                echo=False,
                
                # ⚡ ストリーミング無効（バッチ処理）
                stream=False
            )
            
            duration = __import__('time').time() - start_time
            print(f"⏱️ 生成時間: {duration:.1f}秒")
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            print(f"❌ 生成エラー: {e}")
            return ""
    
    def parse_intent_fast(self, user_input):
        """高速意図解析"""
        
        # 📝 短縮されたプロンプト（高速化）
        prompt = f"""Max/MSP expert. JSON only.

Request: "{user_input}"

JSON:
{{
  "category": "effect|synth|utility",
  "objects": ["obj1", "obj2"],
  "params": {{"key": "value"}}
}}"""
        
        print(f"🧠 意図解析: {user_input}")
        
        result_text = self.generate_fast(prompt, max_tokens=100)
        
        if result_text:
            print(f"🤖 LLM応答: {result_text}")
            
            # JSON抽出
            json_data = self._extract_json_fast(result_text)
            if json_data:
                return json_data
        
        # フォールバック
        print("⚠️ LLM解析失敗、フォールバック使用")
        return self._fallback_parse(user_input)
    
    def _extract_json_fast(self, text):
        """高速JSON抽出"""
        try:
            # 最初のJSONブロックを探す
            start = text.find('{')
            if start == -1:
                return None
                
            # 対応する}を探す
            brace_count = 0
            end = start
            for i, char in enumerate(text[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            
            if brace_count == 0:
                json_str = text[start:end]
                return json.loads(json_str)
                
        except Exception as e:
            print(f"JSON抽出エラー: {e}")
        
        return None
    
    def _fallback_parse(self, user_input):
        """フォールバック解析（高速ルールベース）"""
        
        user_lower = user_input.lower()
        
        # 🎯 シンプルなキーワードマッチング
        if any(word in user_lower for word in ['reverb', 'リバーブ']):
            return {
                'category': 'effect',
                'objects': ['adc~', 'freeverb~', 'dac~'],
                'params': {'roomsize': '0.8'}
            }
        elif any(word in user_lower for word in ['delay', 'ディレイ']):
            return {
                'category': 'effect', 
                'objects': ['adc~', 'tapin~', 'tapout~', 'dac~'],
                'params': {'time': '500'}
            }
        elif any(word in user_lower for word in ['filter', 'フィルター']):
            return {
                'category': 'effect',
                'objects': ['adc~', 'lores~', 'dac~'],
                'params': {'freq': '1000'}
            }
        elif any(word in user_lower for word in ['osc', 'oscillator', 'オシレーター']):
            return {
                'category': 'synth',
                'objects': ['phasor~', 'dac~'],
                'params': {'freq': '440'}
            }
        elif any(word in user_lower for word in ['fm', 'bell', 'ベル']):
            return {
                'category': 'synth',
                'objects': ['phasor~', '*~', '+~', 'cos~', 'dac~'],
                'params': {'carrier': '440', 'mod': '220'}
            }
        
        # デフォルト
        return {
            'category': 'utility',
            'objects': ['adc~', 'dac~'],
            'params': {}
        }
    
    def unload_model(self):
        """メモリ解放"""
        if self.model:
            del self.model
            self.model = None
            import gc
            gc.collect()
            print("🗑️ モデル解放完了")

def speed_test():
    """速度テスト"""
    
    llm = OptimizedMaxPatchLLM()
    
    test_cases = [
        "リバーブエフェクト",
        "FM synthesis bell",
        "lowpass filter", 
        "delay effect"
    ]
    
    print("⚡ 高速化テスト開始\\n")
    
    total_time = __import__('time').time()
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"=== テスト {i}/{len(test_cases)} ===")
        
        start = __import__('time').time()
        result = llm.parse_intent_fast(test_input)
        duration = __import__('time').time() - start
        
        print(f"📊 結果: {result}")
        print(f"⏱️ 処理時間: {duration:.1f}秒\\n")
    
    total_duration = __import__('time').time() - total_time
    print(f"🏁 総時間: {total_duration:.1f}秒")
    print(f"📈 平均: {total_duration/len(test_cases):.1f}秒/テスト")
    
    llm.unload_model()

if __name__ == "__main__":
    speed_test()