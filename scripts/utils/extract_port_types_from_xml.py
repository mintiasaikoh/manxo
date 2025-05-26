#!/usr/bin/env python3
"""
Ableton Live内のMax/MSP XMLリファレンスファイルからポートタイプ情報を抽出
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import json
import logging
from db_connector import DatabaseConnector
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class XMLPortTypeExtractor:
    def __init__(self, db_config_path: str):
        self.db = DatabaseConnector(db_config_path)
        self.base_path = "/Applications/Ableton Live 12 Suite.app/Contents/App-Resources/Max/Max.app/Contents/Resources/C74/docs/refpages"
        
    def extract_all_port_types(self):
        """すべてのXMLファイルからポートタイプ情報を抽出"""
        object_port_info = {}
        
        # すべてのrefpageディレクトリを探索
        ref_dirs = ['max-ref', 'msp-ref', 'jit-ref', 'm4l-ref']
        
        for ref_dir in ref_dirs:
            dir_path = os.path.join(self.base_path, ref_dir)
            if not os.path.exists(dir_path):
                logger.warning(f"ディレクトリが見つかりません: {dir_path}")
                continue
                
            logger.info(f"処理中: {ref_dir}")
            
            # XMLファイルを処理
            xml_files = list(Path(dir_path).glob("*.maxref.xml"))
            
            for xml_file in tqdm(xml_files, desc=f"{ref_dir}のXMLファイル"):
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    
                    # オブジェクト名を取得
                    object_name = root.get('name')
                    if not object_name:
                        continue
                        
                    # インレット情報を取得
                    inlet_types = []
                    inletlist = root.find('inletlist')
                    if inletlist is not None:
                        for inlet in inletlist.findall('inlet'):
                            inlet_type = inlet.get('type', 'unknown')
                            # INLET_TYPEやOUTLET_TYPEではない実際の値のみ収集
                            if inlet_type not in ['INLET_TYPE', 'OUTLET_TYPE', 'TEXT_HERE']:
                                inlet_types.append(inlet_type)
                                
                    # アウトレット情報を取得
                    outlet_types = []
                    outletlist = root.find('outletlist')
                    if outletlist is not None:
                        for outlet in outletlist.findall('outlet'):
                            outlet_type = outlet.get('type', 'unknown')
                            if outlet_type not in ['INLET_TYPE', 'OUTLET_TYPE', 'TEXT_HERE']:
                                outlet_types.append(outlet_type)
                                
                    # 有効な情報がある場合のみ保存
                    if inlet_types or outlet_types:
                        object_port_info[object_name] = {
                            'inlet_types': inlet_types,
                            'outlet_types': outlet_types,
                            'source': ref_dir,
                            'file': str(xml_file.name)
                        }
                        
                except Exception as e:
                    logger.error(f"XMLファイル処理エラー ({xml_file}): {e}")
                    
        logger.info(f"\n抽出完了: {len(object_port_info)}個のオブジェクト")
        
        # サンプルを表示
        logger.info("\n=== 抽出されたポートタイプ情報のサンプル ===")
        for i, (name, info) in enumerate(list(object_port_info.items())[:10]):
            logger.info(f"{name}:")
            logger.info(f"  inlets: {info['inlet_types']}")
            logger.info(f"  outlets: {info['outlet_types']}")
            
        return object_port_info
        
    def update_database_with_xml_info(self, object_port_info):
        """抽出した情報でデータベースを更新"""
        self.db.connect()
        
        try:
            logger.info("\n=== objectsテーブルを更新 ===")
            
            update_count = 0
            for object_name, port_info in tqdm(object_port_info.items(), desc="データベース更新"):
                # 既存のレコードを確認
                existing = self.db.execute_query(
                    "SELECT id FROM objects WHERE object_name = %s",
                    (object_name,)
                )
                
                inlet_types_json = json.dumps(port_info['inlet_types'])
                outlet_types_json = json.dumps(port_info['outlet_types'])
                
                if existing:
                    # 更新
                    self.db.execute_query("""
                        UPDATE objects 
                        SET inlet_types = %s::jsonb,
                            outlet_types = %s::jsonb,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE object_name = %s
                    """, (inlet_types_json, outlet_types_json, object_name))
                else:
                    # 新規挿入
                    self.db.execute_query("""
                        INSERT INTO objects (object_name, inlet_types, outlet_types, num_inlets, num_outlets)
                        VALUES (%s, %s::jsonb, %s::jsonb, %s, %s)
                    """, (
                        object_name,
                        inlet_types_json,
                        outlet_types_json,
                        len(port_info['inlet_types']),
                        len(port_info['outlet_types'])
                    ))
                    
                update_count += 1
                
            self.db.connection.commit()
            logger.info(f"更新完了: {update_count}件")
            
            # object_connectionsテーブルも更新
            logger.info("\n=== object_connectionsテーブルのポートタイプを更新 ===")
            
            # バッチ処理で更新
            batch_size = 1000
            total_updated = 0
            
            for obj_name, port_info in object_port_info.items():
                if port_info['inlet_types']:
                    # インレットタイプを更新
                    inlet_array = '{' + ','.join(f'"{t}"' for t in port_info['inlet_types']) + '}'
                    updated = self.db.execute_query("""
                        UPDATE object_connections
                        SET target_inlet_types = %s::text[]
                        WHERE target_object_type = %s
                        AND (target_inlet_types = '{}' OR target_inlet_types IS NULL)
                    """, (inlet_array, obj_name))
                    
                if port_info['outlet_types']:
                    # アウトレットタイプを更新
                    outlet_array = '{' + ','.join(f'"{t}"' for t in port_info['outlet_types']) + '}'
                    updated = self.db.execute_query("""
                        UPDATE object_connections
                        SET source_outlet_types = %s::text[]
                        WHERE source_object_type = %s
                        AND (source_outlet_types = '{}' OR source_outlet_types IS NULL)
                    """, (outlet_array, obj_name))
                    
                if total_updated % batch_size == 0:
                    self.db.connection.commit()
                    
                total_updated += 1
                
            self.db.connection.commit()
            logger.info(f"接続テーブル更新完了: {total_updated}件のオブジェクトタイプを処理")
            
        finally:
            self.db.disconnect()
            
    def verify_improvements(self):
        """改善結果を検証"""
        self.db.connect()
        
        try:
            # object_connectionsの統計
            result = self.db.execute_query("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN source_outlet_types != '{}' 
                               AND NOT (source_outlet_types @> ARRAY['']::text[]) 
                               AND NOT (source_outlet_types @> ARRAY['unknown']::text[]) THEN 1 END) as valid_outlets,
                    COUNT(CASE WHEN target_inlet_types != '{}' 
                               AND NOT (target_inlet_types @> ARRAY['']::text[]) 
                               AND NOT (target_inlet_types @> ARRAY['unknown']::text[]) THEN 1 END) as valid_inlets
                FROM object_connections
            """)
            
            logger.info("\n=== XMLデータ適用後の統計 ===")
            for row in result:
                logger.info(f"総接続数: {row['total']:,}")
                logger.info(f"有効なsource_outlet_types: {row['valid_outlets']:,} ({row['valid_outlets']/row['total']*100:.1f}%)")
                logger.info(f"有効なtarget_inlet_types: {row['valid_inlets']:,} ({row['valid_inlets']/row['total']*100:.1f}%)")
                
            # サンプル表示
            logger.info("\n=== 改善された接続の例 ===")
            samples = self.db.execute_query("""
                SELECT 
                    source_object_type,
                    source_outlet_types,
                    target_object_type,
                    target_inlet_types
                FROM object_connections
                WHERE source_outlet_types != '{}'
                AND target_inlet_types != '{}'
                AND NOT (source_outlet_types @> ARRAY['']::text[])
                AND NOT (target_inlet_types @> ARRAY['']::text[])
                AND NOT (source_outlet_types @> ARRAY['unknown']::text[])
                AND NOT (target_inlet_types @> ARRAY['unknown']::text[])
                ORDER BY RANDOM()
                LIMIT 20
            """)
            
            for i, row in enumerate(samples[:10]):
                logger.info(f"\n{i+1}. {row['source_object_type']} → {row['target_object_type']}")
                logger.info(f"   outlets: {row['source_outlet_types']}")
                logger.info(f"   inlets: {row['target_inlet_types']}")
                
        finally:
            self.db.disconnect()


def main():
    extractor = XMLPortTypeExtractor('scripts/db_settings.ini')
    
    # XMLファイルからポートタイプ情報を抽出
    object_port_info = extractor.extract_all_port_types()
    
    # データベースを更新
    extractor.update_database_with_xml_info(object_port_info)
    
    # 改善結果を検証
    extractor.verify_improvements()
    
    logger.info("\n✅ XMLファイルからのポートタイプ情報抽出・適用が完了しました！")


if __name__ == "__main__":
    main()