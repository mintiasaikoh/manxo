#!/usr/bin/env python3
"""
ポートタイプ情報を修正するスクリプト

問題：
1. Maxパッチファイルにはinlettypeが存在しない（outlettypeのみ）
2. object_detailsのinlet_typesが全てNULL
3. object_connectionsのポートタイプが不完全

解決策：
1. objectsテーブル（マスタ）の情報を使用
2. outlettypeから可能な限り情報を復元
3. 標準的なオブジェクトの既知のポートタイプを適用
"""

import logging
from db_connector import DatabaseConnector
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PortTypeFixer:
    def __init__(self, db_config_path: str):
        self.db = DatabaseConnector(db_config_path)
        
    def analyze_current_state(self):
        """現在の状態を分析"""
        self.db.connect()
        
        try:
            # object_detailsのoutlet_types情報を確認
            logger.info("=== outlet_types情報の分析 ===")
            result = self.db.execute_query("""
                SELECT 
                    object_type,
                    COUNT(*) as count,
                    array_to_string(outlet_types, ',') as outlet_types_str
                FROM object_details
                WHERE outlet_types IS NOT NULL 
                AND array_length(outlet_types, 1) > 0
                GROUP BY object_type, outlet_types
                ORDER BY count DESC
                LIMIT 20
            """)
            
            logger.info("\n最も一般的なoutlet_types:")
            for row in result:
                logger.info(f"{row['object_type']}: [{row['outlet_types_str']}] ({row['count']}件)")
                
            # 空文字列の割合を確認
            empty_result = self.db.execute_query("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN outlet_types @> ARRAY['']::text[] THEN 1 END) as has_empty
                FROM object_details
                WHERE outlet_types IS NOT NULL
            """)
            
            for row in empty_result:
                logger.info(f"\n空文字列を含むoutlet_types: {row['has_empty']}/{row['total']} ({row['has_empty']/row['total']*100:.1f}%)")
                
        finally:
            self.db.disconnect()
            
    def create_standard_port_types(self):
        """標準的なオブジェクトのポートタイプマッピングを作成"""
        standard_types = {
            # 音声オブジェクト
            'cycle~': {
                'inlet_types': ['signal/float', 'signal/float'],
                'outlet_types': ['signal']
            },
            'dac~': {
                'inlet_types': ['signal', 'signal'],
                'outlet_types': []
            },
            'adc~': {
                'inlet_types': [],
                'outlet_types': ['signal', 'signal']
            },
            '+~': {
                'inlet_types': ['signal', 'signal'],
                'outlet_types': ['signal']
            },
            '*~': {
                'inlet_types': ['signal', 'signal'],
                'outlet_types': ['signal']
            },
            
            # 数値オブジェクト
            '+': {
                'inlet_types': ['float/int', 'float/int'],
                'outlet_types': ['float/int']
            },
            '-': {
                'inlet_types': ['float/int', 'float/int'],
                'outlet_types': ['float/int']
            },
            '*': {
                'inlet_types': ['float/int', 'float/int'],
                'outlet_types': ['float/int']
            },
            '/': {
                'inlet_types': ['float/int', 'float/int'],
                'outlet_types': ['float/int']
            },
            
            # メッセージオブジェクト
            'message': {
                'inlet_types': ['anything'],
                'outlet_types': ['anything']
            },
            'bang': {
                'inlet_types': ['bang'],
                'outlet_types': ['bang']
            },
            'button': {
                'inlet_types': ['anything'],
                'outlet_types': ['bang']
            },
            
            # コントロールオブジェクト
            'metro': {
                'inlet_types': ['bang/int', 'float/int'],
                'outlet_types': ['bang']
            },
            'toggle': {
                'inlet_types': ['int/bang'],
                'outlet_types': ['int']
            },
            'flonum': {
                'inlet_types': ['float'],
                'outlet_types': ['float', 'bang']
            },
            'number': {
                'inlet_types': ['int'],
                'outlet_types': ['int', 'bang']
            },
            
            # リストオブジェクト
            'pack': {
                'inlet_types': ['anything'] * 10,  # 可変だが最大10と仮定
                'outlet_types': ['list']
            },
            'unpack': {
                'inlet_types': ['list'],
                'outlet_types': ['anything'] * 10
            },
            
            # ルーティング
            'route': {
                'inlet_types': ['anything'],
                'outlet_types': ['anything'] * 10
            },
            'select': {
                'inlet_types': ['anything', 'anything'],
                'outlet_types': ['bang'] * 10
            },
            
            # MIDI
            'notein': {
                'inlet_types': [],
                'outlet_types': ['int', 'int', 'int']  # pitch, velocity, channel
            },
            'noteout': {
                'inlet_types': ['int', 'int', 'int'],
                'outlet_types': []
            },
            'ctlin': {
                'inlet_types': [],
                'outlet_types': ['int', 'int', 'int']  # value, controller, channel
            },
            'ctlout': {
                'inlet_types': ['int', 'int', 'int'],
                'outlet_types': []
            }
        }
        
        return standard_types
        
    def update_object_connections_with_masters(self):
        """objectsテーブルのマスタ情報を使ってobject_connectionsを更新"""
        self.db.connect()
        
        try:
            logger.info("=== objectsマスタからポートタイプ情報を取得 ===")
            
            # objectsテーブルから有効なポートタイプ情報を取得
            masters = self.db.execute_query("""
                SELECT 
                    object_name,
                    inlet_types,
                    outlet_types
                FROM objects
                WHERE inlet_types IS NOT NULL 
                AND outlet_types IS NOT NULL
                AND NOT (inlet_types @> '["INLET_TYPE"]'::jsonb)
                AND NOT (outlet_types @> '["OUTLET_TYPE"]'::jsonb)
            """)
            
            master_dict = {}
            for row in masters:
                if row['inlet_types'] and row['outlet_types']:
                    master_dict[row['object_name']] = {
                        'inlet_types': row['inlet_types'],
                        'outlet_types': row['outlet_types']
                    }
                    
            logger.info(f"有効なマスタ情報: {len(master_dict)}件")
            
            # 標準タイプも追加
            standard_types = self.create_standard_port_types()
            for obj_name, types in standard_types.items():
                if obj_name not in master_dict:
                    master_dict[obj_name] = types
                    
            logger.info(f"総マスタ情報: {len(master_dict)}件")
            
            # サンプルを表示
            logger.info("\nマスタ情報のサンプル:")
            for i, (name, types) in enumerate(list(master_dict.items())[:5]):
                logger.info(f"  {name}: in={types['inlet_types']}, out={types['outlet_types']}")
                
            # 更新を実行（バッチ処理）
            logger.info("\n=== object_connectionsの更新を開始 ===")
            
            # 更新対象の接続を取得
            connections = self.db.execute_query("""
                SELECT DISTINCT 
                    source_object_type,
                    target_object_type
                FROM object_connections
                WHERE source_outlet_types = '{}'::text[]
                OR target_inlet_types = '{}'::text[]
                OR source_outlet_types @> ARRAY['']::text[]
                OR target_inlet_types @> ARRAY['']::text[]
                LIMIT 1000
            """)
            
            update_count = 0
            for conn in tqdm(connections, desc="接続タイプを更新中"):
                source_type = conn['source_object_type']
                target_type = conn['target_object_type']
                
                updates = []
                
                # ソースのアウトレットタイプを更新
                if source_type in master_dict:
                    outlet_types = master_dict[source_type]['outlet_types']
                    if outlet_types:
                        updates.append(f"source_outlet_types = ARRAY{outlet_types}::text[]")
                        
                # ターゲットのインレットタイプを更新
                if target_type in master_dict:
                    inlet_types = master_dict[target_type]['inlet_types']
                    if inlet_types:
                        updates.append(f"target_inlet_types = ARRAY{inlet_types}::text[]")
                        
                if updates:
                    update_sql = f"""
                        UPDATE object_connections
                        SET {', '.join(updates)}
                        WHERE source_object_type = %s
                        AND target_object_type = %s
                        AND (source_outlet_types = '{{}}'::text[]
                             OR target_inlet_types = '{{}}'::text[]
                             OR source_outlet_types @> ARRAY['']::text[]
                             OR target_inlet_types @> ARRAY['']::text[])
                    """
                    
                    self.db.execute_query(update_sql, (source_type, target_type))
                    update_count += 1
                    
            # コミットは手動で実行
            self.db.connection.commit()
            logger.info(f"\n更新完了: {update_count}件の接続タイプを更新しました")
            
        finally:
            self.db.disconnect()
            
    def verify_updates(self):
        """更新結果を検証"""
        self.db.connect()
        
        try:
            # 更新後の統計
            result = self.db.execute_query("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN source_outlet_types != '{}' 
                               AND NOT (source_outlet_types @> ARRAY['']::text[]) THEN 1 END) as valid_source,
                    COUNT(CASE WHEN target_inlet_types != '{}' 
                               AND NOT (target_inlet_types @> ARRAY['']::text[]) THEN 1 END) as valid_target
                FROM object_connections
            """)
            
            logger.info("\n=== 更新後の統計 ===")
            for row in result:
                logger.info(f"総接続数: {row['total']:,}")
                logger.info(f"有効なsource_outlet_types: {row['valid_source']:,} ({row['valid_source']/row['total']*100:.1f}%)")
                logger.info(f"有効なtarget_inlet_types: {row['valid_target']:,} ({row['valid_target']/row['total']*100:.1f}%)")
                
            # サンプルを表示
            logger.info("\n=== 更新されたサンプル ===")
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
                LIMIT 10
            """)
            
            for i, row in enumerate(samples):
                logger.info(f"\n例{i+1}: {row['source_object_type']} → {row['target_object_type']}")
                logger.info(f"  source outlets: {row['source_outlet_types']}")
                logger.info(f"  target inlets: {row['target_inlet_types']}")
                
        finally:
            self.db.disconnect()


def main():
    fixer = PortTypeFixer('scripts/db_settings.ini')
    
    # 現在の状態を分析
    fixer.analyze_current_state()
    
    # マスタ情報で更新
    fixer.update_object_connections_with_masters()
    
    # 結果を検証
    fixer.verify_updates()
    
    logger.info("\n✅ ポートタイプ情報の修正が完了しました！")


if __name__ == "__main__":
    main()