#!/usr/bin/env python3
"""
動作確認済みMax/MSP パッチ生成用 LLM インターフェース
"""

import os
import json
import re
from llama_cpp import Llama

class MaxPatchLLM:
    def __init__(self):
        # 動作確認済みモデルパス
        self.model_path = "/Users/mymac/manxo/models/llm/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
        self.model = None
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {self.model_path}")
    
    def load_model(self):
        """必要時にモデルをロード"""
        if self.model is None:
            print(f"🧠 LLMロード中: {os.path.basename(self.model_path)}")
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=2048,     # 充分なコンテキスト
                n_gpu_layers=0, # CPU実行（安定性重視）
                verbose=False,
                use_mlock=False,
                use_mmap=True
            )
            print("✅ モデルロード完了")
    
    def unload_model(self):
        """メモリ節約のためモデル解放"""
        if self.model:
            del self.model
            self.model = None
            import gc
            gc.collect()
            print("🗑️ モデル解放完了")
    
    def parse_intent(self, user_input):
        """ユーザー入力を解析してMax/MSPの意図を抽出"""
        self.load_model()
        
        # Max/MSP特化プロンプト
        prompt = f"""You are a Max/MSP expert. Analyze the following request and respond in JSON format.

Request: "{user_input}"

Respond with JSON in this exact format:
{{
  "category": "effect|synth|utility|sequencer",
  "subcategory": "specific type like reverb, delay, oscillator",
  "objects": ["list", "of", "maxmsp", "objects"],
  "parameters": {{"param": "value"}},
  "complexity": 1-5
}}

JSON:"""
        
        try:
            response = self.model(
                prompt,
                max_tokens=300,
                stop=["\\n\\n", "\\n```", "Response:"],
                echo=False,
                temperature=0.1
            )
            
            result = response['choices'][0]['text'].strip()
            print(f"🤖 LLM応答: {result}")
            
            # JSONを抽出・パース
            json_data = self._extract_json(result)
            if json_data:
                return json_data
            else:
                print("⚠️ JSON解析失敗、フォールバック使用")
                return self._fallback_parse(user_input)
                
        except Exception as e:
            print(f"⚠️ LLM解析エラー: {e}")
            return self._fallback_parse(user_input)
    
    def _extract_json(self, text):
        """テキストからJSONを抽出"""
        try:
            # JSONブロックを探す
            if '{' in text and '}' in text:
                start = text.find('{')
                end = text.rfind('}') + 1
                json_str = text[start:end]
                
                # 不正な文字を修正
                json_str = re.sub(r'["""]', '"', json_str)  # 全角引用符を半角に
                json_str = re.sub(r'，', ',', json_str)      # 全角カンマを半角に
                
                return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            print(f"JSON解析エラー: {e}")
        except Exception as e:
            print(f"JSON抽出エラー: {e}")
        
        return None
    
    def _fallback_parse(self, user_input):
        """LLM失敗時のルールベース解析"""
        
        user_lower = user_input.lower()
        
        # キーワードベースの分類
        patterns = {
            # エフェクト系
            'reverb': {
                'category': 'effect',
                'subcategory': 'reverb',
                'objects': ['adc~', 'freeverb~', 'dac~'],
                'parameters': {'roomsize': '0.8', 'damp': '0.5'},
                'complexity': 2
            },
            'delay': {
                'category': 'effect', 
                'subcategory': 'delay',
                'objects': ['adc~', 'tapin~', 'tapout~', '*~', '+~', 'dac~'],
                'parameters': {'delay_time': '500', 'feedback': '0.3'},
                'complexity': 3
            },
            'filter': {
                'category': 'effect',
                'subcategory': 'filter', 
                'objects': ['adc~', 'lores~', 'dac~'],
                'parameters': {'frequency': '1000', 'resonance': '1.0'},
                'complexity': 2
            },
            'distortion': {
                'category': 'effect',
                'subcategory': 'distortion',
                'objects': ['adc~', 'overdrive~', 'dac~'],
                'parameters': {'drive': '0.7'},
                'complexity': 2
            },
            
            # シンセサイザー系  
            'oscillator': {
                'category': 'synth',
                'subcategory': 'oscillator',
                'objects': ['phasor~', 'dac~'],
                'parameters': {'frequency': '440'},
                'complexity': 1
            },
            'fm': {
                'category': 'synth',
                'subcategory': 'fm_synthesis',
                'objects': ['phasor~', '*~', '+~', 'cos~', 'dac~'],
                'parameters': {'carrier_freq': '440', 'mod_freq': '220', 'mod_depth': '100'},
                'complexity': 4
            },
            'synth': {
                'category': 'synth',
                'subcategory': 'subtractive',
                'objects': ['phasor~', 'lores~', '*~', 'dac~'],
                'parameters': {'frequency': '440', 'filter_freq': '1000'},
                'complexity': 3
            },
            
            # ユーティリティ系
            'mixer': {
                'category': 'utility',
                'subcategory': 'mixer',
                'objects': ['*~', '+~'],
                'parameters': {'channels': '4'},
                'complexity': 2
            },
            
            # シーケンサー系
            'sequencer': {
                'category': 'sequencer',
                'subcategory': 'step_sequencer', 
                'objects': ['metro', 'counter', 'sel', 'message'],
                'parameters': {'bpm': '120', 'steps': '8'},
                'complexity': 3
            }
        }
        
        # パターンマッチング
        for keyword, config in patterns.items():
            if keyword in user_lower:
                print(f"🎯 キーワードマッチ: {keyword}")
                return config
        
        # デフォルト設定
        print("🔧 デフォルト設定を使用")
        return {
            'category': 'effect',
            'subcategory': 'basic',
            'objects': ['adc~', 'dac~'],
            'parameters': {},
            'complexity': 1
        }

def test_llm_interface():
    """LLMインターフェースをテスト"""
    
    llm = MaxPatchLLM()
    
    test_cases = [
        "ステレオリバーブエフェクトを作って",
        "FM合成でベル音を作りたい", 
        "ローパスフィルターエフェクト",
        "8ステップシーケンサー",
        "ディストーションエフェクト",
        "4チャンネルミキサー"
    ]
    
    print("🧪 Max/MSP LLMインターフェーステスト開始\\n")
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"=== テスト {i}/{len(test_cases)} ===")
        print(f"📝 入力: {test_input}")
        
        try:
            result = llm.parse_intent(test_input)
            print(f"📊 結果:")
            print(f"  カテゴリ: {result['category']} > {result['subcategory']}")
            print(f"  オブジェクト: {result['objects']}")
            print(f"  パラメーター: {result['parameters']}")
            print(f"  複雑度: {result['complexity']}/5")
            print("✅ 成功\\n")
            
        except Exception as e:
            print(f"❌ エラー: {e}\\n")
    
    # メモリ解放
    llm.unload_model()
    print("🎉 全テスト完了!")

if __name__ == "__main__":
    test_llm_interface()