#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Max/MSPバイナリヘッダー解析のテスト
"""

from analyze_patch_connections import PatchConnectionAnalyzer

def main():
    """バイナリヘッダー解析テスト"""
    config_path = "./db_settings.ini"
    analyzer = PatchConnectionAnalyzer(config_path)
    
    test_files = [
        "/Users/mymac/manxo/Max_Projects/all_amxd/ASD.amxd",
        "/Users/mymac/manxo/Max_Projects/all_amxd/Arduino.amxd",
        "/Users/mymac/manxo/Max_Projects/all_amxd/Filter.amxd"
    ]
    
    for amxd_file in test_files:
        print(f"\n=== バイナリヘッダー解析: {amxd_file} ===")
        
        try:
            # バイナリ読み込み
            with open(amxd_file, 'rb') as f:
                content = f.read()
            
            # ヘッダー解析
            header_info = analyzer.parse_amxd_binary_header(content)
            
            if header_info:
                print(f"フォーマット: {header_info.get('format')}")
                print(f"AMPFサイズ: {header_info.get('ampf_size')}")
                print(f"メタヘッダー: {header_info.get('meta_header')}")
                print(f"メタサイズ: {header_info.get('meta_size')}")
                print(f"バージョン: {header_info.get('version')}")
                print(f"パッチヘッダー: {header_info.get('patch_header')}")
                print(f"パッチサイズ: {header_info.get('patch_size')}")
                print(f"コレクションヘッダー: {header_info.get('collection_header', 'なし')}")
                print(f"JSONデータ開始位置: {header_info.get('json_start_pos')}")
                
                # JSONデータの抽出テスト
                json_start = header_info.get('json_start_pos', 0)
                json_preview = content[json_start:json_start+100].decode('utf-8', errors='ignore')
                print(f"JSONプレビュー: {json_preview[:50]}...")
                
            else:
                print("ヘッダー解析に失敗")
                
        except Exception as e:
            print(f"エラー: {e}")

if __name__ == "__main__":
    main()