#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
amxdファイルのデバイスタイプ判定とパース能力をテスト
Max for Liveデバイスの正しい処理を確認
"""

import json
import os
from pathlib import Path
from analyze_patch_connections import PatchConnectionAnalyzer

def extract_amxd_json(amxd_file_path):
    """amxdファイルからJSONデータを抽出"""
    try:
        with open(amxd_file_path, 'rb') as f:
            content = f.read()
        
        # JSONの開始位置を探す
        json_start = content.find(b'{"patcher"')
        if json_start == -1:
            json_start = content.find(b'{\n\t"patcher"')
        
        if json_start == -1:
            print(f"JSONデータが見つかりません: {amxd_file_path}")
            return None
        
        # JSONデータを抽出
        json_data = content[json_start:].decode('utf-8', errors='ignore')
        
        # 不完全なJSONを修正（最後の}まで）
        brace_count = 0
        end_pos = 0
        for i, char in enumerate(json_data):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break
        
        if end_pos > 0:
            json_data = json_data[:end_pos]
        
        return json.loads(json_data)
        
    except Exception as e:
        print(f"amxd解析エラー {amxd_file_path}: {e}")
        return None

def get_device_type(amxd_data):
    """amxdデータからデバイスタイプを判定"""
    if not amxd_data:
        return "unknown"
    
    patcher = amxd_data.get('patcher', {})
    
    # Live.deviceタグをチェック
    device_hint = patcher.get('devicehint', '')
    if device_hint:
        if 'audio_effect' in device_hint.lower():
            return "audio_effect"
        elif 'midi_effect' in device_hint.lower():
            return "midi_effect"
        elif 'instrument' in device_hint.lower():
            return "instrument"
    
    # ボックスを分析してデバイスタイプを推定
    boxes = patcher.get('boxes', [])
    
    has_audio_in = False
    has_audio_out = False
    has_midi_in = False
    has_midi_out = False
    has_plugin_in = False
    has_plugin_out = False
    
    for box in boxes:
        box_data = box.get('box', {})
        maxclass = box_data.get('maxclass', '')
        text = box_data.get('text', '')
        
        if maxclass == 'newobj':
            if text.startswith('in ') or text == 'in':
                has_midi_in = True
            elif text.startswith('out ') or text == 'out':
                has_midi_out = True
            elif text.startswith('in~ ') or text == 'in~':
                has_audio_in = True
            elif text.startswith('out~ ') or text == 'out~':
                has_audio_out = True
            elif text.startswith('plugin~ ') or text == 'plugin~':
                has_plugin_in = True
                has_plugin_out = True
    
    # デバイスタイプを判定
    if has_plugin_in and has_plugin_out:
        return "vst_wrapper"
    elif has_audio_in and has_audio_out:
        return "audio_effect"
    elif has_midi_in and has_audio_out:
        return "instrument"
    elif has_midi_in and has_midi_out:
        return "midi_effect"
    elif has_audio_out:
        return "generator"
    else:
        return "utility"

def test_amxd_files():
    """複数のamxdファイルをテスト"""
    
    test_files = [
        "Max_Projects/all_amxd/github_OSC Send.amxd",
        "Max_Projects/all_amxd/ableton_example.amxd_35a1951e.amxd", 
        "Max_Projects/all_amxd/Drumfoldr1.3.amxd"
    ]
    
    print("=== amxdファイル分析テスト ===")
    
    analyzer = PatchConnectionAnalyzer('scripts/db_settings.ini')
    analyzer.load_object_id_cache()
    
    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"ファイルが見つかりません: {file_path}")
            continue
            
        print(f"\n--- {os.path.basename(file_path)} ---")
        
        # 1. JSONデータ抽出テスト
        amxd_data = extract_amxd_json(file_path)
        if amxd_data:
            print("✓ JSONデータ抽出成功")
            
            # 2. デバイスタイプ判定
            device_type = get_device_type(amxd_data)
            print(f"デバイスタイプ: {device_type}")
            
            # 3. ボックス数とライン数
            patcher = amxd_data.get('patcher', {})
            boxes = patcher.get('boxes', [])
            lines = patcher.get('lines', [])
            print(f"ボックス数: {len(boxes)}")
            print(f"接続数: {len(lines)}")
            
            # 4. 既存の分析スクリプトでのテスト
            try:
                analysis = analyzer.analyze_patch_file(file_path)
                if analysis:
                    print(f"✓ 分析成功: {analysis['total_objects']}個のオブジェクト, {analysis['total_connections']}個の接続")
                    
                    # データベース格納テスト
                    stored_count = analyzer.store_connections_to_db(analysis)
                    print(f"✓ データベース格納: {stored_count}個の接続")
                    
                else:
                    print("✗ 分析失敗")
            except Exception as e:
                print(f"✗ 分析エラー: {e}")
        else:
            print("✗ JSONデータ抽出失敗")

def analyze_amxd_distribution():
    """amxdファイルの分布を分析"""
    print("\n=== amxdファイル分布分析 ===")
    
    amxd_files = list(Path("Max_Projects").rglob("*.amxd"))
    print(f"総amxdファイル数: {len(amxd_files)}")
    
    device_types = {}
    successful_parses = 0
    
    for file_path in amxd_files[:20]:  # 最初の20個をテスト
        amxd_data = extract_amxd_json(str(file_path))
        if amxd_data:
            successful_parses += 1
            device_type = get_device_type(amxd_data)
            device_types[device_type] = device_types.get(device_type, 0) + 1
    
    print(f"解析成功率: {successful_parses}/20 ({successful_parses/20*100:.1f}%)")
    print("\nデバイスタイプ分布:")
    for device_type, count in sorted(device_types.items()):
        print(f"  {device_type}: {count}個")

if __name__ == "__main__":
    test_amxd_files()
    analyze_amxd_distribution()