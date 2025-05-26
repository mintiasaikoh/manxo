#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
amxdファイルの詳細な分析テスト
デバイスタイプ判定と接続情報の抽出を確認
"""

import json
from analyze_patch_connections import PatchConnectionAnalyzer

def main():
    """テスト用メイン関数"""
    config_path = "./db_settings.ini"
    analyzer = PatchConnectionAnalyzer(config_path)
    
    # テスト対象のamxdファイル
    test_files = [
        "/Users/mymac/manxo/Max_Projects/all_amxd/ASD.amxd",
        "/Users/mymac/manxo/Max_Projects/all_amxd/Arduino.amxd",
        "/Users/mymac/manxo/Max_Projects/all_amxd/Filter.amxd"
    ]
    
    analyzer.load_object_id_cache()
    
    for amxd_file in test_files:
        print(f"\n=== 分析: {amxd_file} ===")
        
        try:
            result = analyzer.analyze_patch_file(amxd_file)
            
            if result:
                print(f"ファイル名: {result['patch_name']}")
                print(f"総オブジェクト数: {result['total_objects']}")
                print(f"総接続数: {result['total_connections']}")
                
                # デバイスタイプの確認
                if 'device_type' in result:
                    print(f"デバイスタイプ: {result['device_type']}")
                else:
                    print("デバイスタイプ: 未検出")
                
                # オブジェクトの詳細（一部のみ表示）
                print(f"\nオブジェクト例（最初の5個）:")
                for i, (obj_id, obj_info) in enumerate(result['objects'].items()):
                    if i >= 5:
                        break
                    print(f"  {obj_id}: {obj_info['maxclass']} - {obj_info.get('object_name', 'N/A')}")
                
                # 接続例（最初の3個のみ表示）
                print(f"\n接続例（最初の3個）:")
                for i, conn in enumerate(result['connections']):
                    if i >= 3:
                        break
                    print(f"  {conn['source_object_name']}[{conn['source_port']}] -> {conn['dest_object_name']}[{conn['dest_port']}]")
                
            else:
                print("分析失敗")
                
        except Exception as e:
            print(f"エラー: {e}")

if __name__ == "__main__":
    main()