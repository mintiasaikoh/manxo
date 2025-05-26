#!/usr/bin/env python3
"""
Max/MSPオブジェクトのポート情報を詳細に抽出・更新
XMLリファレンスファイルから各ポートの役割と説明を取得
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

class PortInfoEnhancer:
    """ポート情報を詳細化するクラス"""
    
    def __init__(self, config_path: str):
        """初期化"""
        self.db = DatabaseConnector(config_path)
        self.db.connect()
        
    def extract_detailed_port_info(self, xml_file_path: str) -> Optional[Dict[str, Any]]:
        """
        XMLリファレンスファイルから詳細なポート情報を抽出
        
        Args:
            xml_file_path: XMLファイルのパス
            
        Returns:
            オブジェクトの詳細ポート情報
        """
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            
            # オブジェクト名を取得
            object_name = None
            for name_elem in root.iter():
                if name_elem.tag in ['c74object', 'object'] and name_elem.get('name'):
                    object_name = name_elem.get('name')
                    break
            
            if not object_name:
                return None
            
            port_info = {
                'object_name': object_name,
                'inlets': [],
                'outlets': []
            }
            
            # インレット情報を詳細に抽出
            for inlet_elem in root.iter('inlet'):
                inlet_info = {
                    'id': inlet_elem.get('id', '0'),
                    'type': inlet_elem.get('type', 'unknown'),
                    'digest': inlet_elem.get('digest', ''),
                    'description': inlet_elem.text or ''
                }
                
                # 追加のメタデータ検索
                for child in inlet_elem:
                    if child.tag == 'assist':
                        inlet_info['assist'] = child.get('assist', '')
                    elif child.tag == 'digest':
                        inlet_info['digest'] = child.text or ''
                
                port_info['inlets'].append(inlet_info)
            
            # アウトレット情報を詳細に抽出
            for outlet_elem in root.iter('outlet'):
                outlet_info = {
                    'id': outlet_elem.get('id', '0'),
                    'type': outlet_elem.get('type', 'unknown'),
                    'digest': outlet_elem.get('digest', ''),
                    'description': outlet_elem.text or ''
                }
                
                # 追加のメタデータ検索
                for child in outlet_elem:
                    if child.tag == 'assist':
                        outlet_info['assist'] = child.get('assist', '')
                    elif child.tag == 'digest':
                        outlet_info['digest'] = child.text or ''
                
                port_info['outlets'].append(outlet_info)
            
            # ioverdescription要素からさらに詳細な情報を取得
            for io_elem in root.iter('iodescription'):
                for inlet in io_elem.iter('inlet'):
                    inlet_id = inlet.get('id', '0')
                    for i, inlet_info in enumerate(port_info['inlets']):
                        if inlet_info['id'] == inlet_id:
                            port_info['inlets'][i]['detailed_description'] = inlet.text or ''
                            break
                
                for outlet in io_elem.iter('outlet'):
                    outlet_id = outlet.get('id', '0')
                    for i, outlet_info in enumerate(port_info['outlets']):
                        if outlet_info['id'] == outlet_id:
                            port_info['outlets'][i]['detailed_description'] = outlet.text or ''
                            break
            
            return port_info
            
        except Exception as e:
            print(f"XML解析エラー {xml_file_path}: {e}")
            return None
    
    def update_object_ports(self, object_name: str, port_info: Dict[str, Any]):
        """
        データベースのオブジェクトポート情報を更新
        
        Args:
            object_name: オブジェクト名
            port_info: 詳細ポート情報
        """
        try:
            cursor = self.db.connection.cursor()
            
            # 詳細なポート情報を構築
            detailed_inlets = []
            for inlet in port_info['inlets']:
                detailed_inlet = {
                    'id': inlet['id'],
                    'type': inlet['type'],
                    'digest': inlet.get('digest', ''),
                    'description': inlet.get('description', ''),
                    'assist': inlet.get('assist', ''),
                    'detailed_description': inlet.get('detailed_description', '')
                }
                detailed_inlets.append(detailed_inlet)
            
            detailed_outlets = []
            for outlet in port_info['outlets']:
                detailed_outlet = {
                    'id': outlet['id'],
                    'type': outlet['type'],
                    'digest': outlet.get('digest', ''),
                    'description': outlet.get('description', ''),
                    'assist': outlet.get('assist', ''),
                    'detailed_description': outlet.get('detailed_description', '')
                }
                detailed_outlets.append(detailed_outlet)
            
            # データベースを更新
            update_sql = """
            UPDATE objects SET 
                inlet_types = %s,
                outlet_types = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE object_name = %s
            """
            
            cursor.execute(update_sql, (
                json.dumps(detailed_inlets),
                json.dumps(detailed_outlets),
                object_name
            ))
            
            self.db.connection.commit()
            return True
            
        except Exception as e:
            print(f"データベース更新エラー {object_name}: {e}")
            self.db.connection.rollback()
            return False
        finally:
            cursor.close()
    
    def enhance_all_ports(self, xml_dir: str) -> Dict[str, int]:
        """
        XMLディレクトリ内の全ファイルから詳細ポート情報を抽出・更新
        
        Args:
            xml_dir: XMLファイルディレクトリ
            
        Returns:
            処理統計
        """
        stats = {
            'files_processed': 0,
            'objects_updated': 0,
            'objects_not_found': 0
        }
        
        print(f"XMLディレクトリを処理中: {xml_dir}")
        
        # .maxref.xmlファイルを検索（これらがオブジェクトリファレンス）
        for root, dirs, files in os.walk(xml_dir):
            for file in files:
                if file.endswith('.maxref.xml'):
                    file_path = os.path.join(root, file)
                    print(f"処理中: {file_path}")
                    
                    port_info = self.extract_detailed_port_info(file_path)
                    stats['files_processed'] += 1
                    
                    if port_info and port_info['object_name']:
                        # データベースに該当オブジェクトが存在するかチェック
                        cursor = self.db.connection.cursor()
                        cursor.execute('SELECT COUNT(*) FROM objects WHERE object_name = %s', (port_info['object_name'],))
                        exists = cursor.fetchone()[0] > 0
                        cursor.close()
                        
                        if exists:
                            success = self.update_object_ports(port_info['object_name'], port_info)
                            if success:
                                stats['objects_updated'] += 1
                                print(f"  更新成功: {port_info['object_name']}")
                            else:
                                print(f"  更新失敗: {port_info['object_name']}")
                        else:
                            stats['objects_not_found'] += 1
                            print(f"  データベースに存在しません: {port_info['object_name']}")
        
        return stats

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Max/MSPオブジェクトのポート情報を詳細化')
    parser.add_argument('--config', required=True, help='データベース設定ファイルのパス')
    parser.add_argument('--xml-dir', required=True, help='XMLリファレンスディレクトリのパス')
    
    args = parser.parse_args()
    
    try:
        enhancer = PortInfoEnhancer(args.config)
        
        print("ポート情報の詳細化を開始...")
        stats = enhancer.enhance_all_ports(args.xml_dir)
        
        print("\n=== 処理結果 ===")
        print(f"処理ファイル数: {stats['files_processed']}")
        print(f"更新オブジェクト数: {stats['objects_updated']}")
        print(f"データベースに存在しないオブジェクト数: {stats['objects_not_found']}")
        
        enhancer.db.disconnect()
        
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()