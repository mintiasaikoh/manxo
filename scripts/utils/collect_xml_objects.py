#!/usr/bin/env python3
"""
Ableton Live 10のXMLファイルからMax/MSPオブジェクト情報を収集してデータベースに保存
"""

import os
import xml.etree.ElementTree as ET
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# プロジェクトルートのパスを追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.db_connector import DatabaseConnector

class XMLObjectCollector:
    """XMLファイルからオブジェクト情報を収集するクラス"""
    
    def __init__(self, config_path: str):
        """
        初期化
        
        Args:
            config_path: データベース設定ファイルのパス
        """
        self.db = DatabaseConnector(config_path)
        self.db.connect()
        self.collected_objects = {}
        
    def parse_xml_file(self, xml_file_path: str) -> List[Dict[str, Any]]:
        """
        XMLファイルを解析してオブジェクト情報を抽出
        
        Args:
            xml_file_path: XMLファイルのパス
            
        Returns:
            オブジェクト情報のリスト
        """
        objects = []
        
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            
            # オブジェクト要素を検索
            for obj_elem in root.iter():
                if obj_elem.tag in ['object', 'c74object', 'objref']:
                    obj_info = self.extract_object_info(obj_elem)
                    if obj_info:
                        objects.append(obj_info)
                        
        except Exception as e:
            print(f"XMLファイル解析エラー {xml_file_path}: {e}")
            
        return objects
    
    def extract_object_info(self, obj_elem) -> Optional[Dict[str, Any]]:
        """
        オブジェクト要素から情報を抽出
        
        Args:
            obj_elem: XMLオブジェクト要素
            
        Returns:
            オブジェクト情報辞書
        """
        obj_info = {}
        
        # オブジェクト名
        name = obj_elem.get('name') or obj_elem.get('type') or obj_elem.text
        if not name:
            return None
        
        obj_info['name'] = name.strip()
        obj_info['type'] = obj_elem.tag
        
        # 属性情報を収集
        for attr_name, attr_value in obj_elem.attrib.items():
            if attr_name in ['category', 'digest', 'module', 'description']:
                obj_info[attr_name] = attr_value
        
        # ポート情報を検索
        inlets = []
        outlets = []
        
        # inlet要素
        for inlet in obj_elem.findall('.//inlet'):
            inlet_info = {
                'id': inlet.get('id', len(inlets)),
                'type': inlet.get('type', 'unknown'),
                'description': inlet.get('digest', '')
            }
            inlets.append(inlet_info)
        
        # outlet要素
        for outlet in obj_elem.findall('.//outlet'):
            outlet_info = {
                'id': outlet.get('id', len(outlets)),
                'type': outlet.get('type', 'unknown'),
                'description': outlet.get('digest', '')
            }
            outlets.append(outlet_info)
        
        obj_info['inlets'] = inlets
        obj_info['outlets'] = outlets
        obj_info['inlet_count'] = len(inlets)
        obj_info['outlet_count'] = len(outlets)
        
        return obj_info
    
    def collect_from_directory(self, directory_path: str) -> Dict[str, int]:
        """
        ディレクトリ内のすべてのXMLファイルからオブジェクト情報を収集
        
        Args:
            directory_path: 検索ディレクトリのパス
            
        Returns:
            収集統計
        """
        stats = {
            'files_processed': 0,
            'objects_found': 0,
            'objects_inserted': 0
        }
        
        print(f"ディレクトリを検索中: {directory_path}")
        
        # XMLファイルを再帰的に検索
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.xml'):
                    file_path = os.path.join(root, file)
                    print(f"処理中: {file_path}")
                    
                    objects = self.parse_xml_file(file_path)
                    stats['files_processed'] += 1
                    stats['objects_found'] += len(objects)
                    
                    # データベースに挿入
                    for obj_info in objects:
                        try:
                            object_name = obj_info['name']
                            
                            # 重複チェック
                            if object_name in self.collected_objects:
                                continue
                                
                            self.collected_objects[object_name] = obj_info
                            
                            # データベースに挿入
                            object_data = {
                                'object_name': object_name,
                                'object_type': obj_info.get('type', 'unknown'),
                                'category': obj_info.get('category', ''),
                                'num_inlets': obj_info['inlet_count'],
                                'num_outlets': obj_info['outlet_count'],
                                'inlet_types': json.dumps([inlet.get('type', '') for inlet in obj_info['inlets']]),
                                'outlet_types': json.dumps([outlet.get('type', '') for outlet in obj_info['outlets']]),
                                'description': obj_info.get('description', '')
                            }
                            self.db.insert_object(object_data)
                            
                            stats['objects_inserted'] += 1
                            
                        except Exception as e:
                            print(f"オブジェクト挿入エラー {obj_info.get('name', 'unknown')}: {e}")
                            continue
        
        return stats
    
    def save_collected_data(self, output_file: str):
        """
        収集したデータをJSONファイルに保存
        
        Args:
            output_file: 出力ファイルパス
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.collected_objects, f, indent=2, ensure_ascii=False)

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='XMLファイルからMax/MSPオブジェクト情報を収集')
    parser.add_argument('--config', required=True, help='データベース設定ファイルのパス')
    parser.add_argument('--xml-dir', required=True, help='XMLファイルのディレクトリパス')
    parser.add_argument('--output', help='収集データのJSONファイル出力先')
    parser.add_argument('--init', action='store_true', help='データベースを初期化')
    
    args = parser.parse_args()
    
    try:
        collector = XMLObjectCollector(args.config)
        
        if args.init:
            print("データベースを初期化中...")
            collector.db.initialize_database()
            print("データベース初期化完了")
        
        # XMLファイルから収集
        print(f"XMLディレクトリから収集開始: {args.xml_dir}")
        stats = collector.collect_from_directory(args.xml_dir)
        
        # 収集データをファイルに保存
        if args.output:
            collector.save_collected_data(args.output)
            print(f"収集データを保存: {args.output}")
        
        # 結果表示
        print("\n=== 収集結果 ===")
        print(f"処理ファイル数: {stats['files_processed']}")
        print(f"発見オブジェクト数: {stats['objects_found']}")
        print(f"挿入オブジェクト数: {stats['objects_inserted']}")
        
        # データベース統計を表示
        print("\n=== データベース統計 ===")
        try:
            cursor = collector.db.connection.cursor()
            for table in ['objects', 'connections', 'connection_patterns', 'port_info']:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"{table}: {count} レコード")
        except Exception as e:
            print(f"統計取得エラー: {e}")
            
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()