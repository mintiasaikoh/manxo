#!/usr/bin/env python3
"""
newobjの解析でNoneになる問題を調査
"""

import json
import os

def debug_newobj_parsing():
    """newobjの解析問題をデバッグ"""
    test_file = '/Users/mymac/manxo/Max_Projects/all_amxd/08bf2ced_Red Panda Particle 2.amxd'
    
    try:
        with open(test_file, 'rb') as f:
            content = f.read()
        
        # JSONの開始位置を探す
        json_start = content.find(b'"patcher"')
        if json_start != -1:
            # パッチャーの前にある{を探す
            for i in range(json_start, -1, -1):
                if content[i:i+1] == b'{':
                    json_start = i
                    break
            
            json_bytes = content[json_start:]
            
            # JSONの終了位置を見つける
            brace_count = 0
            end_pos = 0
            for i, byte in enumerate(json_bytes):
                if byte == ord('{'):
                    brace_count += 1
                elif byte == ord('}'):
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            
            if end_pos > 0:
                json_data = json_bytes[:end_pos].decode('utf-8', errors='ignore')
                patch_data = json.loads(json_data)
                
                print('=== newobj解析問題の調査 ===')
                
                # newobjボックスを探してtextフィールドを調査
                patcher = patch_data.get('patcher', {})
                boxes = patcher.get('boxes', [])
                
                newobj_examples = []
                none_examples = []
                
                for box in boxes:
                    box_data = box.get('box', {})
                    if box_data.get('maxclass') == 'newobj':
                        text = box_data.get('text', '')
                        obj_id = box_data.get('id', '')
                        
                        # オブジェクト名の抽出をテスト
                        if text:
                            parts = text.split()
                            object_name = parts[0] if parts else None
                            if object_name:
                                newobj_examples.append({
                                    'id': obj_id,
                                    'text': text,
                                    'extracted': object_name
                                })
                            else:
                                none_examples.append({
                                    'id': obj_id,
                                    'text': text,
                                    'reason': 'empty_after_split'
                                })
                        else:
                            none_examples.append({
                                'id': obj_id,
                                'text': text,
                                'reason': 'empty_text'
                            })
                
                print('正常に解析されたnewobjの例（最初の10個）:')
                for i, example in enumerate(newobj_examples[:10]):
                    print(f'  ID: {example["id"]}, text: "{example["text"]}", extracted: "{example["extracted"]}"')
                
                print(f'\nNoneになったnewobjの例（最初の10個）:')
                for i, example in enumerate(none_examples[:10]):
                    print(f'  ID: {example["id"]}, text: "{example["text"]}", reason: {example["reason"]}')
                
                print(f'\n統計:')
                print(f'  正常解析: {len(newobj_examples)}個')
                print(f'  None: {len(none_examples)}個')
                print(f'  None率: {len(none_examples) / (len(newobj_examples) + len(none_examples)) * 100:.1f}%')

    except Exception as e:
        print(f'エラー: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_newobj_parsing()