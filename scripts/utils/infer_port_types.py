#!/usr/bin/env python3
"""
ポートタイプを推測して補完するスクリプト

Max/MSPの一般的なルール：
1. signal → signal (音声信号)
2. float → float/int (数値)
3. int → int/float (数値)
4. bang → anything (トリガー)
5. list → list/anything (リスト)
6. 空文字 → anything (汎用)
"""

import logging
from db_connector import DatabaseConnector
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PortTypeInferencer:
    def __init__(self, db_config_path: str):
        self.db = DatabaseConnector(db_config_path)
        
    def create_inference_rules(self):
        """ポートタイプの推測ルールを定義"""
        # アウトレットタイプからインレットタイプを推測
        inference_rules = {
            'signal': ['signal', 'signal/float'],
            'float': ['float', 'float/int', 'number', 'anything'],
            'int': ['int', 'float/int', 'number', 'anything'],
            'bang': ['bang', 'anything'],
            'list': ['list', 'anything'],
            'symbol': ['symbol', 'anything'],
            '': ['anything'],  # 空文字列
            
            # 特殊なタイプ
            'float/int': ['float/int', 'float', 'int', 'number', 'anything'],
            'signal/float': ['signal/float', 'signal', 'float', 'anything'],
            'Signal': ['signal', 'signal/float'],  # 大文字のSignal
            'Signal/Float': ['signal/float', 'signal', 'float'],
            
            # Genオブジェクト
            'number': ['number', 'float', 'int', 'float/int'],
            'double': ['double', 'float', 'number'],
            
            # その他
            'FullPacket': ['FullPacket', 'anything'],
            'clearSelection': ['anything'],
        }
        
        return inference_rules
        
    def infer_connection_types(self):
        """接続のポートタイプを推測して更新"""
        self.db.connect()
        
        try:
            # 推測ルールを取得
            rules = self.create_inference_rules()
            
            logger.info("=== ポートタイプの推測を開始 ===")
            
            # 更新が必要な接続を取得（バッチサイズを増やす）
            batch_size = 10000
            offset = 0
            total_updated = 0
            
            while True:
                connections = self.db.execute_query("""
                    SELECT 
                        id,
                        source_object_type,
                        target_object_type,
                        source_outlet_types,
                        target_inlet_types,
                        source_port,
                        target_port
                    FROM object_connections
                    WHERE (target_inlet_types = '{}' OR target_inlet_types @> ARRAY['']::text[])
                    AND source_outlet_types IS NOT NULL
                    AND array_length(source_outlet_types, 1) > 0
                    ORDER BY id
                    LIMIT %s OFFSET %s
                """, (batch_size, offset))
                
                if not connections:
                    break
                    
                logger.info(f"バッチ {offset//batch_size + 1}: {len(connections)}件を処理中...")
                
                updates = []
                
                for conn in tqdm(connections, desc="推測中", leave=False):
                    # ソースのアウトレットタイプを取得
                    source_types = conn['source_outlet_types']
                    source_port = conn['source_port']
                    
                    if source_types and len(source_types) > source_port:
                        outlet_type = source_types[source_port]
                    else:
                        outlet_type = source_types[0] if source_types else ''
                        
                    # 推測ルールを適用
                    if outlet_type in rules:
                        inferred_types = rules[outlet_type]
                    else:
                        # デフォルトルール
                        inferred_types = ['anything']
                        
                    # ターゲットオブジェクトタイプに基づく調整
                    target_type = conn['target_object_type']
                    
                    # 特定のオブジェクトタイプ用の調整
                    if target_type in ['dac~', 'gain~', 'record~', '*~', '+~', 'cycle~']:
                        if 'signal' in outlet_type.lower():
                            inferred_types = ['signal']
                    elif target_type in ['flonum']:
                        if any(t in outlet_type for t in ['float', 'number', 'int']):
                            inferred_types = ['float']
                    elif target_type in ['number']:
                        if any(t in outlet_type for t in ['int', 'float', 'number']):
                            inferred_types = ['int']
                    elif target_type in ['message', 'print']:
                        inferred_types = ['anything']
                        
                    # 更新クエリを準備
                    updates.append((
                        conn['id'],
                        inferred_types
                    ))
                    
                # バッチ更新
                if updates:
                    for conn_id, inlet_types in updates:
                        # 単一のインレットタイプとして設定
                        inlet_type_array = f"ARRAY['{inlet_types[0]}']::text[]"
                        
                        self.db.execute_query("""
                            UPDATE object_connections
                            SET target_inlet_types = %s
                            WHERE id = %s
                        """, (inlet_type_array, conn_id))
                        
                    self.db.connection.commit()
                    total_updated += len(updates)
                    logger.info(f"  {len(updates)}件を更新しました")
                    
                offset += batch_size
                
            logger.info(f"\n合計 {total_updated}件の接続のインレットタイプを推測・更新しました")
            
        finally:
            self.db.disconnect()
            
    def verify_inference_results(self):
        """推測結果を検証"""
        self.db.connect()
        
        try:
            # 更新後の統計
            result = self.db.execute_query("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN target_inlet_types != '{}' 
                               AND NOT (target_inlet_types @> ARRAY['']::text[]) 
                               AND NOT (target_inlet_types @> ARRAY['unknown']::text[]) THEN 1 END) as valid_inlets
                FROM object_connections
            """)
            
            logger.info("\n=== 推測後の統計 ===")
            for row in result:
                logger.info(f"総接続数: {row['total']:,}")
                logger.info(f"有効なtarget_inlet_types: {row['valid_inlets']:,} ({row['valid_inlets']/row['total']*100:.1f}%)")
                
            # ポートタイプの分布
            logger.info("\n=== インレットタイプの分布（上位20） ===")
            distribution = self.db.execute_query("""
                SELECT 
                    target_inlet_types[1] as inlet_type,
                    COUNT(*) as count
                FROM object_connections
                WHERE array_length(target_inlet_types, 1) > 0
                GROUP BY inlet_type
                ORDER BY count DESC
                LIMIT 20
            """)
            
            for row in distribution:
                logger.info(f"{row['inlet_type']}: {row['count']:,}件")
                
            # 接続パターンの例
            logger.info("\n=== 推測された接続パターンの例 ===")
            samples = self.db.execute_query("""
                SELECT 
                    source_object_type,
                    source_outlet_types[1] as outlet_type,
                    target_object_type,
                    target_inlet_types[1] as inlet_type
                FROM object_connections
                WHERE target_inlet_types != '{}'
                AND NOT (target_inlet_types @> ARRAY['']::text[])
                AND NOT (target_inlet_types @> ARRAY['unknown']::text[])
                ORDER BY RANDOM()
                LIMIT 15
            """)
            
            for i, row in enumerate(samples):
                logger.info(f"{i+1}. {row['source_object_type']}({row['outlet_type']}) → {row['target_object_type']}({row['inlet_type']})")
                
        finally:
            self.db.disconnect()


def main():
    inferencer = PortTypeInferencer('scripts/db_settings.ini')
    
    # ポートタイプを推測
    inferencer.infer_connection_types()
    
    # 結果を検証
    inferencer.verify_inference_results()
    
    logger.info("\n✅ ポートタイプの推測が完了しました！")


if __name__ == "__main__":
    main()