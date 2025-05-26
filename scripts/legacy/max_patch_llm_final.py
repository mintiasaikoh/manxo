#!/usr/bin/env python3
"""
創造的Max/MSP LLMパッチジェネレーター（最終版）
"""

import os
import json
import re
from llama_cpp import Llama

class CreativeMaxPatchLLM:
    def __init__(self):
        self.model_path = "/Users/mymac/manxo/models/llm/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
        self.model = None
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {self.model_path}")
    
    def load_model(self):
        """最適化設定でモデルロード"""
        if self.model is None:
            print(f"🧠 CreativeLLM ロード中...")
            
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=512,          # 高速化
                n_batch=16,         # 少し大きめ
                n_threads=4,
                n_gpu_layers=0,
                use_mmap=True,
                use_mlock=False,
                verbose=False,
                f16_kv=True,
            )
            print("✅ CreativeLLM 準備完了")
    
    def creative_interpret(self, user_input):
        """創造的解釈でパッチ生成"""
        
        self.load_model()
        
        # 🎨 創造的プロンプト
        prompt = f"""You are a creative Max/MSP artist. Create an innovative patch for: "{user_input}"

Think creatively about sounds, textures, and musical expression.

Respond in JSON format only:
{{
  "concept": "creative interpretation",
  "category": "effect|synth|generative",
  "objects": ["maxmsp", "objects", "list"],
  "connections": [
    {{"from": "obj1", "to": "obj2", "outlet": 0, "inlet": 0}}
  ],
  "parameters": {{"param": "value"}},
  "description": "artistic description"
}}

JSON:"""
        
        print(f"🎨 創造的解釈: {user_input}")
        
        try:
            response = self.model(
                prompt,
                max_tokens=250,
                temperature=0.3,    # 創造性を少し上げる
                top_p=0.95,
                stop=["\\n\\n", "Human:", "Response:"],
                echo=False
            )
            
            result_text = response['choices'][0]['text'].strip()
            print(f"🤖 LLM応答: {result_text[:200]}...")
            
            # JSON抽出を試行
            json_data = self._robust_json_extract(result_text)
            
            if json_data and 'objects' in json_data:
                return self._enhance_creative_result(json_data, user_input)
            else:
                print("⚠️ JSON解析失敗、創造的フォールバック")
                return self._creative_fallback(user_input)
                
        except Exception as e:
            print(f"❌ LLM エラー: {e}")
            return self._creative_fallback(user_input)
    
    def _robust_json_extract(self, text):
        """堅牢なJSON抽出"""
        
        # パターン1: 完全なJSONブロック
        json_pattern = r'\\{[^{}]*(?:\\{[^{}]*\\}[^{}]*)*\\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                # 文字修正
                clean_match = re.sub(r'["""]', '"', match)
                clean_match = re.sub(r'，', ',', clean_match)
                return json.loads(clean_match)
            except:
                continue
        
        # パターン2: 手動構築
        if '"objects"' in text and '[' in text:
            try:
                # オブジェクトリストを抽出
                obj_match = re.search(r'"objects"\\s*:\\s*\\[([^\\]]+)\\]', text)
                if obj_match:
                    objects_str = obj_match.group(1)
                    # オブジェクトを解析
                    objects = [obj.strip(' "').replace('"', '') for obj in objects_str.split(',')]
                    
                    return {
                        'concept': 'extracted from LLM',
                        'category': 'effect',
                        'objects': objects,
                        'connections': [],
                        'parameters': {},
                        'description': 'Generated creatively'
                    }
            except:
                pass
        
        return None
    
    def _enhance_creative_result(self, base_result, user_input):
        """創造的結果を強化"""
        
        # 基本構造を保持しつつ、実用的な接続を追加
        objects = base_result.get('objects', [])
        
        if len(objects) < 2:
            # 最低限のオブジェクトを保証
            if 'synth' in base_result.get('category', ''):
                objects = ['phasor~'] + objects + ['dac~']
            else:
                objects = ['adc~'] + objects + ['dac~']
        
        # 自動接続生成（シンプルチェーン）
        connections = []
        for i in range(len(objects) - 1):
            connections.append({
                'from': objects[i],
                'to': objects[i + 1],
                'outlet': 0,
                'inlet': 0
            })
        
        # レイアウト座標を追加
        layout = {}
        for i, obj in enumerate(objects):
            layout[obj] = {
                'x': 50 + i * 120,
                'y': 50 + (i % 3) * 60,
                'width': 100,
                'height': 22
            }
        
        enhanced = base_result.copy()
        enhanced.update({
            'objects': objects,
            'connections': connections,
            'layout': layout,
            'enhanced': True
        })
        
        return enhanced
    
    def _creative_fallback(self, user_input):
        """創造的フォールバック（ルールベース + 創造性）"""
        
        user_lower = user_input.lower()
        
        # 🎨 創造的パターン
        creative_patterns = {
            # 感情・雰囲気系
            'ambient|雰囲気|静か': {
                'concept': 'Ambient soundscape',
                'category': 'generative',
                'objects': ['noise~', 'lores~', 'freeverb~', '*~', 'dac~'],
                'description': 'Dreamy ambient textures'
            },
            'aggressive|激しい|distortion': {
                'concept': 'Aggressive processing',
                'category': 'effect',
                'objects': ['adc~', 'overdrive~', 'tanh~', 'clip~', 'dac~'],
                'description': 'Intense sonic destruction'
            },
            'bell|鐘|chime': {
                'concept': 'Metallic resonance',
                'category': 'synth',
                'objects': ['phasor~', '*~', '+~', 'cos~', 'comb~', 'dac~'],
                'description': 'Shimmering bell tones'
            },
            'rain|雨|water': {
                'concept': 'Liquid textures',
                'category': 'generative',
                'objects': ['noise~', 'bp~', 'delay~', '*~', 'dac~'],
                'description': 'Gentle water sounds'
            },
            'space|宇宙|cosmic': {
                'concept': 'Cosmic exploration',
                'category': 'effect',
                'objects': ['adc~', 'freeverb~', 'delay~', 'pitch~', 'dac~'],
                'description': 'Ethereal space sounds'
            }
        }
        
        # パターンマッチング
        for pattern, config in creative_patterns.items():
            if any(word in user_lower for word in pattern.split('|')):
                print(f"🎨 創造的パターン: {config['concept']}")
                return self._enhance_creative_result(config, user_input)
        
        # デフォルト創造的構成
        return {
            'concept': f'Creative interpretation of: {user_input}',
            'category': 'experimental',
            'objects': ['adc~', 'slide~', 'reson~', 'dac~'],
            'connections': [
                {'from': 'adc~', 'to': 'slide~', 'outlet': 0, 'inlet': 0},
                {'from': 'slide~', 'to': 'reson~', 'outlet': 0, 'inlet': 0},
                {'from': 'reson~', 'to': 'dac~', 'outlet': 0, 'inlet': 0}
            ],
            'parameters': {'slide_time': '100', 'resonance': '0.8'},
            'description': 'Experimental sound processing'
        }
    
    def unload_model(self):
        """メモリ解放"""
        if self.model:
            del self.model
            self.model = None
            import gc
            gc.collect()

def creative_test():
    """創造的生成テスト"""
    
    llm = CreativeMaxPatchLLM()
    
    creative_requests = [
        "雨の音を表現したアンビエント",
        "宇宙の響きを作りたい",
        "激しいディストーション",
        "幻想的なベルの音",
        "Create a dreamy reverb soundscape"
    ]
    
    print("🎨 創造的パッチ生成テスト\\n")
    
    for i, request in enumerate(creative_requests, 1):
        print(f"=== 創造的テスト {i}/{len(creative_requests)} ===")
        print(f"🎵 要求: {request}")
        
        start_time = __import__('time').time()
        result = llm.creative_interpret(request)
        duration = __import__('time').time() - start_time
        
        print(f"🎨 コンセプト: {result.get('concept', 'N/A')}")
        print(f"📊 カテゴリ: {result.get('category', 'N/A')}")
        print(f"🔧 オブジェクト: {result.get('objects', [])}")
        print(f"📝 説明: {result.get('description', 'N/A')}")
        print(f"⏱️ 処理時間: {duration:.1f}秒\\n")
    
    llm.unload_model()
    print("🎉 創造的テスト完了!")

if __name__ == "__main__":
    creative_test()