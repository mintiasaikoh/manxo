#!/usr/bin/env python3
"""
実際の接続パターンからポートタイプを推測
データベース内の689,098接続を分析して、頻繁に接続されるオブジェクトのペアから
ポートタイプの互換性を推測する
"""

import logging
from db_connector import DatabaseConnector
from collections import defaultdict
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConnectionPatternAnalyzer:
    def __init__(self, db_config_path: str):
        self.db = DatabaseConnector(db_config_path)
        
    def analyze_connection_patterns(self):
        """接続パターンを分析してポートタイプを推測"""
        self.db.connect()
        
        try:
            # 最も頻繁な接続パターンを取得
            logger.info("=== 頻繁な接続パターンの分析 ===")
            
            # 音声オブジェクト間の接続
            signal_connections = self.db.execute_query("""
                SELECT 
                    source_object_type,
                    target_object_type,
                    COUNT(*) as count
                FROM object_connections
                WHERE source_object_type LIKE '%~'
                AND target_object_type LIKE '%~'
                GROUP BY source_object_type, target_object_type
                ORDER BY count DESC
                LIMIT 50
            """)
            
            logger.info("\n音声オブジェクト間の接続（上位50）:")
            for conn in signal_connections[:20]:
                logger.info(f"{conn['source_object_type']} → {conn['target_object_type']}: {conn['count']}回")
                
            # 数値オブジェクトへの接続
            number_connections = self.db.execute_query("""
                SELECT 
                    source_object_type,
                    target_object_type,
                    COUNT(*) as count
                FROM object_connections
                WHERE target_object_type IN ('number', 'flonum', 'int', 'float')
                GROUP BY source_object_type, target_object_type
                ORDER BY count DESC
                LIMIT 50
            """)
            
            logger.info("\n数値オブジェクトへの接続（上位20）:")
            for conn in number_connections[:20]:
                logger.info(f"{conn['source_object_type']} → {conn['target_object_type']}: {conn['count']}回")
                
            # 接続パターンからルールを生成
            return self.generate_inference_rules(signal_connections, number_connections)
            
        finally:
            self.db.disconnect()
            
    def generate_inference_rules(self, signal_conns, number_conns):
        """接続パターンから推測ルールを生成"""
        rules = {}
        
        # 音声オブジェクトのルール
        signal_objects = set()
        for conn in signal_conns:
            signal_objects.add(conn['source_object_type'])
            signal_objects.add(conn['target_object_type'])
            
        for obj in signal_objects:
            if obj not in rules:
                rules[obj] = {
                    'inlet_types': ['signal'],
                    'outlet_types': ['signal']
                }
                
        # 特殊な音声オブジェクト
        special_signal_objects = {
            'dac~': {'inlet_types': ['signal', 'signal'], 'outlet_types': []},
            'adc~': {'inlet_types': [], 'outlet_types': ['signal', 'signal']},
            'sig~': {'inlet_types': ['float'], 'outlet_types': ['signal']},
            'snapshot~': {'inlet_types': ['signal'], 'outlet_types': ['float']},
            'number~': {'inlet_types': ['signal'], 'outlet_types': ['signal', 'float']}
        }
        rules.update(special_signal_objects)
        
        # 数値を出力するオブジェクト
        number_sources = defaultdict(int)
        for conn in number_conns:
            number_sources[conn['source_object_type']] += conn['count']
            
        # 上位のソースは数値を出力すると推測
        for obj, count in sorted(number_sources.items(), key=lambda x: x[1], reverse=True)[:30]:
            if obj not in rules and not obj.endswith('~'):
                rules[obj] = {
                    'inlet_types': ['float/int'],
                    'outlet_types': ['float/int']
                }
                
        return rules
        
    def update_database_with_inferred_types(self, rules):
        """推測したタイプでデータベースを更新"""
        self.db.connect()
        
        try:
            logger.info("\n=== 推測したポートタイプでデータベースを更新 ===")
            
            update_count = 0
            for obj_name, types in rules.items():
                # objectsテーブルを更新
                existing = self.db.execute_query(
                    "SELECT id FROM objects WHERE object_name = %s",
                    (obj_name,)
                )
                
                if existing:
                    self.db.execute_query("""
                        UPDATE objects
                        SET inlet_types = %s::jsonb,
                            outlet_types = %s::jsonb
                        WHERE object_name = %s
                        AND (inlet_types::text = '[]' OR inlet_types::text LIKE '%INLET_TYPE%')
                    """, (
                        json.dumps(types['inlet_types']),
                        json.dumps(types['outlet_types']),
                        obj_name
                    ))
                    
                # object_connectionsテーブルも更新
                if types['inlet_types']:
                    inlet_array = '{' + ','.join(f'"{t}"' for t in types['inlet_types']) + '}'
                    self.db.execute_query("""
                        UPDATE object_connections
                        SET target_inlet_types = %s::text[]
                        WHERE target_object_type = %s
                        AND (target_inlet_types = '{}' OR target_inlet_types IS NULL)
                    """, (inlet_array, obj_name))
                    
                if types['outlet_types']:
                    outlet_array = '{' + ','.join(f'"{t}"' for t in types['outlet_types']) + '}'
                    self.db.execute_query("""
                        UPDATE object_connections
                        SET source_outlet_types = %s::text[]
                        WHERE source_object_type = %s
                        AND (source_outlet_types = '{}' OR source_outlet_types IS NULL)
                    """, (outlet_array, obj_name))
                    
                update_count += 1
                
            self.db.connection.commit()
            logger.info(f"更新完了: {update_count}件のオブジェクトタイプ")
            
            # 改善結果を確認
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
            
            for row in result:
                logger.info(f"\n最終結果:")
                logger.info(f"  有効なsource_outlet_types: {row['valid_outlets']:,} ({row['valid_outlets']/row['total']*100:.1f}%)")
                logger.info(f"  有効なtarget_inlet_types: {row['valid_inlets']:,} ({row['valid_inlets']/row['total']*100:.1f}%)")
                
        finally:
            self.db.disconnect()


def main():
    analyzer = ConnectionPatternAnalyzer('scripts/db_settings.ini')
    
    # 接続パターンを分析
    rules = analyzer.analyze_connection_patterns()
    
    # データベースを更新
    if rules:
        analyzer.update_database_with_inferred_types(rules)
    
    logger.info("\n✅ 接続パターンからの推測が完了しました！")


if __name__ == "__main__":
    main()