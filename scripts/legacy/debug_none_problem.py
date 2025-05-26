#!/usr/bin/env python3
"""
Noneになる具体的な原因を調査
"""

import json
import os

def analyze_none_problem():
    """Noneになる原因を詳しく調査"""
    problem_files = [
        '/Users/mymac/manxo/Max_Projects/all_amxd/LP S7epper1.8.amxd',
        '/Users/mymac/manxo/Max_Projects/all_amxd/ASD.amxd'
    ]
    
    for file_path in problem_files:
        if not os.path.exists(file_path):
            print(f"ファイルが見つかりません: {file_path}")
            continue
            
        print(f"\n=== {os.path.basename(file_path)} の分析 ===")
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # JSONの開始位置を探す
            json_start = content.find(b'"patcher"')
            if json_start == -1:
                print("patcherが見つかりません")
                continue
                
            # パッチャーの前にある{を探す
            for i in range(json_start, -1, -1):
                if content[i:i+1] == b'{':
                    json_start = i
                    break
            
            json_bytes = content[json_start:]
            
            # JSONの終了位置を見つける
            brace_count = 0
            end_pos = 0
            in_string = False
            escape_next = False
            
            for i, byte in enumerate(json_bytes):
                char = chr(byte) if byte < 128 else '?'
                
                if escape_next:
                    escape_next = False
                    continue
                    
                if char == '\\':
                    escape_next = True
                    continue
                    
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                    
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
            
            if end_pos > 0:
                json_data = json_bytes[:end_pos].decode('utf-8', errors='ignore')
                patch_data = json.loads(json_data)
                
                # newobjボックスを詳しく調査
                patcher = patch_data.get('patcher', {})
                boxes = patcher.get('boxes', [])
                
                newobj_analysis = {
                    'valid': [],
                    'empty_text': [],
                    'whitespace_only': [],
                    'special_chars': [],
                    'other_issues': []
                }
                
                for box in boxes:
                    box_data = box.get('box', {})
                    if box_data.get('maxclass') == 'newobj':
                        text = box_data.get('text', '')
                        obj_id = box_data.get('id', '')
                        
                        if not text:
                            newobj_analysis['empty_text'].append({
                                'id': obj_id,
                                'text': repr(text)
                            })
                        elif text.strip() == '':
                            newobj_analysis['whitespace_only'].append({
                                'id': obj_id,
                                'text': repr(text)
                            })
                        elif not text.strip().split():
                            newobj_analysis['other_issues'].append({
                                'id': obj_id,
                                'text': repr(text),
                                'issue': 'no_tokens_after_split'
                            })
                        else:
                            parts = text.strip().split()
                            object_name = parts[0]
                            if object_name:
                                newobj_analysis['valid'].append({
                                    'id': obj_id,
                                    'text': text,
                                    'extracted': object_name
                                })
                            else:
                                newobj_analysis['other_issues'].append({
                                    'id': obj_id,
                                    'text': repr(text),
                                    'issue': 'empty_first_token'
                                })
                
                print(f"newobjボックス分析結果:")
                print(f"  正常: {len(newobj_analysis['valid'])}個")
                print(f"  空のtext: {len(newobj_analysis['empty_text'])}個")
                print(f"  空白のみ: {len(newobj_analysis['whitespace_only'])}個")
                print(f"  その他の問題: {len(newobj_analysis['other_issues'])}個")
                
                # 問題のある例を表示
                if newobj_analysis['empty_text']:
                    print(f"\n空のtextの例（最初の5個）:")
                    for item in newobj_analysis['empty_text'][:5]:
                        print(f"  ID: {item['id']}, text: {item['text']}")
                
                if newobj_analysis['whitespace_only']:
                    print(f"\n空白のみのtextの例（最初の5個）:")
                    for item in newobj_analysis['whitespace_only'][:5]:
                        print(f"  ID: {item['id']}, text: {item['text']}")
                
                if newobj_analysis['other_issues']:
                    print(f"\nその他の問題の例（最初の5個）:")
                    for item in newobj_analysis['other_issues'][:5]:
                        print(f"  ID: {item['id']}, text: {item['text']}, issue: {item['issue']}")
                        
        except Exception as e:
            print(f"エラー: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    analyze_none_problem()