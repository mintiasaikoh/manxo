#!/usr/bin/env python3
"""
maxhelpファイルからMax/MSPオブジェクト情報を収集
XMLリファレンスに含まれていないオブジェクトを補完
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# プロジェクトルートのパスを追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.db_connector import DatabaseConnector

class MaxhelpObjectCollector:
    """maxhelpファイルからオブジェクト情報を収集するクラス"""
    
    def __init__(self, config_path: str):
        """初期化"""
        self.db = DatabaseConnector(config_path)
        self.db.connect()
        
        # 既存のオブジェクト名を取得
        cursor = self.db.connection.cursor()
        cursor.execute('SELECT object_name FROM objects')
        self.existing_objects = set(row[0] for row in cursor.fetchall())
        cursor.close()
        
    def extract_objects_from_help_dir(self, help_dir: str) -> List[Dict[str, Any]]:
        """
        helpディレクトリからオブジェクト情報を抽出
        
        Args:
            help_dir: helpディレクトリのパス
            
        Returns:
            新しいオブジェクト情報のリスト
        """
        new_objects = []
        
        for root, dirs, files in os.walk(help_dir):
            for file in files:
                if file.endswith('.maxhelp'):
                    # ファイル名からオブジェクト名を抽出
                    object_name = file.replace('.maxhelp', '')
                    
                    # 既存オブジェクトでない場合のみ処理
                    if object_name not in self.existing_objects:
                        # ディレクトリ名からカテゴリを推定
                        relative_path = os.path.relpath(root, help_dir)
                        category = self.guess_category_from_path(relative_path, object_name)
                        
                        # オブジェクトタイプを推定
                        object_type = self.guess_object_type(object_name)
                        
                        # ポート数を推定（基本的な推定のみ）
                        inlets, outlets = self.guess_port_count(object_name)
                        
                        obj_info = {
                            'object_name': object_name,
                            'object_type': object_type,
                            'category': category,
                            'num_inlets': inlets,
                            'num_outlets': outlets,
                            'inlet_types': json.dumps([]),
                            'outlet_types': json.dumps([]),
                            'description': f'Help file object: {object_name}'
                        }
                        
                        new_objects.append(obj_info)
                        
        return new_objects
    
    def guess_category_from_path(self, relative_path: str, object_name: str) -> str:
        """パスとオブジェクト名からカテゴリを推定"""
        path_parts = relative_path.lower().split(os.sep)
        
        # パスベースの推定
        if 'max' in path_parts:
            if object_name.endswith('~'):
                return 'MSP'
            elif object_name.startswith('jit.'):
                return 'Jitter'
            elif object_name.startswith('live.'):
                return 'Max for Live'
            else:
                return 'Max'
        elif 'msp' in path_parts:
            return 'MSP'
        elif 'jitter' in path_parts or 'jit' in path_parts:
            return 'Jitter'
        elif 'm4l' in path_parts or 'live' in path_parts:
            return 'Max for Live'
        
        # オブジェクト名ベースの推定
        if object_name.endswith('~'):
            return 'MSP'
        elif object_name.startswith('jit.'):
            return 'Jitter'
        elif object_name.startswith('live.'):
            return 'Max for Live'
        elif object_name.startswith('mc.'):
            return 'MC'
        elif object_name in ['equals', 'greaterthan', 'lessthan', 'greaterthaneq', 'lessthaneq']:
            return 'Math'
        elif 'and' in object_name or 'or' in object_name:
            return 'Logic'
        else:
            return 'Max'
    
    def guess_object_type(self, object_name: str) -> str:
        """オブジェクト名からタイプを推定"""
        if object_name.endswith('~'):
            return 'msp'
        elif object_name.startswith('jit.'):
            return 'jitter'
        elif object_name.startswith('live.'):
            return 'max4live'
        elif object_name.startswith('mc.'):
            return 'multichannel'
        else:
            return 'max'
    
    def guess_port_count(self, object_name: str) -> tuple:
        """オブジェクト名からポート数を推定"""
        # 基本的な推定ルール
        if object_name in ['equals', 'greaterthan', 'lessthan', 'greaterthaneq', 'lessthaneq']:
            return (2, 1)  # 比較演算子は通常2入力1出力
        elif object_name in ['bitand', 'bitor', 'logand', 'logor']:
            return (2, 1)  # 論理演算子も2入力1出力
        elif object_name.endswith('~'):
            return (2, 1)  # MSPオブジェクトの基本形
        elif object_name.startswith('jit.'):
            return (1, 1)  # Jitterオブジェクトの基本形
        else:
            return (1, 1)  # デフォルト
    
    def insert_new_objects(self, new_objects: List[Dict[str, Any]]) -> int:
        """新しいオブジェクトをデータベースに挿入"""
        inserted_count = 0
        
        for obj_data in new_objects:
            try:
                success = self.db.insert_object(obj_data)
                if success > 0:
                    inserted_count += 1
                    print(f"  追加成功: {obj_data['object_name']}")
                
            except Exception as e:
                print(f"  追加失敗: {obj_data['object_name']} - {e}")
                continue
        
        return inserted_count

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='maxhelpファイルからオブジェクト情報を収集')
    parser.add_argument('--config', required=True, help='データベース設定ファイルのパス')
    parser.add_argument('--help-dir', required=True, help='helpディレクトリのパス')
    
    args = parser.parse_args()
    
    try:
        collector = MaxhelpObjectCollector(args.config)
        
        print(f"helpディレクトリから新しいオブジェクトを収集中: {args.help_dir}")
        new_objects = collector.extract_objects_from_help_dir(args.help_dir)
        
        print(f"発見された新しいオブジェクト数: {len(new_objects)}")
        
        if new_objects:
            inserted_count = collector.insert_new_objects(new_objects)
            print(f"\\n=== 結果 ===")
            print(f"新しく追加されたオブジェクト: {inserted_count}個")
        else:
            print("新しいオブジェクトは見つかりませんでした。")
            
        collector.db.disconnect()
        
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()