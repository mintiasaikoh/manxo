#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Max/MSPパッチ分析レポート生成スクリプト
データベースから統計情報を抽出して見やすいレポートを生成
"""

import sys
from typing import Dict, List, Any
from db_connector import DatabaseConnector
from datetime import datetime
import json
from decimal import Decimal

class DecimalEncoder(json.JSONEncoder):
    """Decimal型を処理できるJSONエンコーダー"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

class AnalysisReporter:
    """分析レポート生成クラス"""
    
    def __init__(self, config_path: str):
        """
        初期化
        
        Args:
            config_path: データベース設定ファイルのパス
        """
        self.db = DatabaseConnector(config_path)
    
    def generate_value_statistics(self) -> Dict[str, Any]:
        """値の統計情報を生成"""
        try:
            self.db.connect()
            
            # 基本統計
            basic_stats = self.db.execute_query("""
                SELECT 
                    COUNT(*) as total_connections,
                    COUNT(DISTINCT patch_file) as unique_patches,
                    COUNT(source_value) as connections_with_source_value,
                    COUNT(target_value) as connections_with_target_value,
                    COUNT(CASE WHEN source_value IS NOT NULL OR target_value IS NOT NULL THEN 1 END) as connections_with_any_value
                FROM object_connections
            """)[0]
            
            # オブジェクトタイプ別の値保持率
            type_value_stats = self.db.execute_query("""
                SELECT 
                    source_object_type,
                    COUNT(*) as count,
                    COUNT(source_value) as with_value,
                    ROUND(COUNT(source_value) * 100.0 / COUNT(*), 1) as value_percentage
                FROM object_connections
                GROUP BY source_object_type
                HAVING COUNT(*) > 10
                ORDER BY COUNT(*) DESC
                LIMIT 20
            """)
            
            # よく使われる値（messageボックス）
            common_messages = self.db.execute_query("""
                SELECT 
                    source_value,
                    COUNT(*) as usage_count
                FROM object_connections
                WHERE source_object_type = 'message' 
                AND source_value IS NOT NULL
                AND source_value != ''
                GROUP BY source_value
                ORDER BY COUNT(*) DESC
                LIMIT 20
            """)
            
            # 階層別統計
            hierarchy_stats = self.db.execute_query("""
                SELECT 
                    hierarchy_depth,
                    COUNT(*) as connections,
                    COUNT(DISTINCT parent_context) as unique_contexts
                FROM object_connections
                GROUP BY hierarchy_depth
                ORDER BY hierarchy_depth
            """)
            
            # デバイスタイプ別統計（amxdファイル）
            device_stats = self.db.execute_query("""
                SELECT 
                    device_type,
                    COUNT(DISTINCT patch_file) as patch_count,
                    COUNT(*) as connection_count
                FROM object_connections
                WHERE file_type = 'amxd'
                AND device_type IS NOT NULL
                GROUP BY device_type
                ORDER BY patch_count DESC
            """)
            
            return {
                'basic_stats': basic_stats,
                'type_value_stats': type_value_stats,
                'common_messages': common_messages,
                'hierarchy_stats': hierarchy_stats,
                'device_stats': device_stats
            }
            
        finally:
            self.db.disconnect()
    
    def generate_connection_patterns(self) -> List[Dict[str, Any]]:
        """接続パターンの統計を生成"""
        try:
            self.db.connect()
            
            # 頻出接続パターン
            patterns = self.db.execute_query("""
                SELECT 
                    source_object_type || ' → ' || target_object_type as pattern,
                    COUNT(*) as count,
                    COUNT(DISTINCT patch_file) as in_patches,
                    
                    -- 値付きの例を1つ取得
                    (SELECT source_value || ' → ' || COALESCE(target_value, '')
                     FROM object_connections oc2 
                     WHERE oc2.source_object_type = oc.source_object_type 
                     AND oc2.target_object_type = oc.target_object_type
                     AND oc2.source_value IS NOT NULL
                     LIMIT 1) as example_with_value
                     
                FROM object_connections oc
                GROUP BY source_object_type, target_object_type
                HAVING COUNT(*) > 50
                ORDER BY COUNT(*) DESC
                LIMIT 30
            """)
            
            return patterns
            
        finally:
            self.db.disconnect()
    
    def generate_gen_statistics(self) -> Dict[str, Any]:
        """Gen関連の統計を生成"""
        try:
            self.db.connect()
            
            # gen~内部のオブジェクト統計
            gen_objects = self.db.execute_query("""
                SELECT 
                    source_object_type as object_type,
                    COUNT(*) as usage_count,
                    COUNT(DISTINCT parent_context) as in_contexts
                FROM object_connections
                WHERE parent_context LIKE '%gen%' 
                OR parent_context LIKE '%jit.gen%'
                GROUP BY source_object_type
                
                UNION
                
                SELECT 
                    target_object_type as object_type,
                    COUNT(*) as usage_count,
                    COUNT(DISTINCT parent_context) as in_contexts
                FROM object_connections
                WHERE parent_context LIKE '%gen%' 
                OR parent_context LIKE '%jit.gen%'
                GROUP BY target_object_type
                
                ORDER BY usage_count DESC
                LIMIT 20
            """)
            
            # Paramオブジェクトの定義
            gen_params = self.db.execute_query("""
                SELECT 
                    source_value as param_definition,
                    COUNT(*) as usage_count
                FROM object_connections
                WHERE source_object_type = 'newobj'
                AND source_value LIKE 'Param %'
                GROUP BY source_value
                ORDER BY COUNT(*) DESC
                LIMIT 20
            """)
            
            return {
                'gen_objects': gen_objects,
                'gen_params': gen_params
            }
            
        finally:
            self.db.disconnect()
    
    def print_report(self):
        """レポートを標準出力に表示"""
        print("=" * 80)
        print("Max/MSPパッチ分析レポート")
        print(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # 値の統計
        value_stats = self.generate_value_statistics()
        basic = value_stats['basic_stats']
        
        print("\n## 基本統計")
        print(f"総接続数: {basic['total_connections']:,}")
        print(f"ユニークパッチ数: {basic['unique_patches']:,}")
        print(f"値付き接続数: {basic['connections_with_any_value']:,} ({basic['connections_with_any_value']/basic['total_connections']*100:.1f}%)")
        print(f"  - ソース値あり: {basic['connections_with_source_value']:,}")
        print(f"  - ターゲット値あり: {basic['connections_with_target_value']:,}")
        
        print("\n## オブジェクトタイプ別の値保持率（上位20）")
        print(f"{'オブジェクトタイプ':<20} {'使用数':>10} {'値あり':>10} {'保持率':>8}")
        print("-" * 50)
        for stat in value_stats['type_value_stats']:
            print(f"{stat['source_object_type']:<20} {stat['count']:>10,} {stat['with_value']:>10,} {stat['value_percentage']:>7.1f}%")
        
        print("\n## よく使われるメッセージ（上位20）")
        print(f"{'メッセージ内容':<40} {'使用回数':>10}")
        print("-" * 52)
        for msg in value_stats['common_messages']:
            message = msg['source_value'][:40] if len(msg['source_value']) > 40 else msg['source_value']
            print(f"{message:<40} {msg['usage_count']:>10,}")
        
        print("\n## 階層構造の統計")
        print(f"{'階層深度':>8} {'接続数':>12} {'ユニークコンテキスト':>20}")
        print("-" * 42)
        for stat in value_stats['hierarchy_stats']:
            print(f"{stat['hierarchy_depth']:>8} {stat['connections']:>12,} {stat['unique_contexts']:>20}")
        
        if value_stats['device_stats']:
            print("\n## Max for Liveデバイスタイプ統計")
            print(f"{'デバイスタイプ':<20} {'パッチ数':>10} {'接続数':>12}")
            print("-" * 44)
            for stat in value_stats['device_stats']:
                print(f"{stat['device_type']:<20} {stat['patch_count']:>10} {stat['connection_count']:>12,}")
        
        # 接続パターン
        patterns = self.generate_connection_patterns()
        print("\n## 頻出接続パターン（上位30）")
        print(f"{'パターン':<40} {'使用数':>10} {'パッチ数':>10} {'値の例':<30}")
        print("-" * 92)
        for pattern in patterns[:30]:
            pattern_str = pattern['pattern'][:40] if len(pattern['pattern']) > 40 else pattern['pattern']
            example = pattern['example_with_value'][:30] if pattern['example_with_value'] else ''
            print(f"{pattern_str:<40} {pattern['count']:>10,} {pattern['in_patches']:>10} {example:<30}")
        
        # Gen統計
        gen_stats = self.generate_gen_statistics()
        if gen_stats['gen_objects']:
            print("\n## Gen/Gen~内部オブジェクト統計")
            print(f"{'オブジェクト':<20} {'使用数':>10} {'コンテキスト数':>15}")
            print("-" * 47)
            for obj in gen_stats['gen_objects'][:15]:
                print(f"{obj['object_type']:<20} {obj['usage_count']:>10} {obj['in_contexts']:>15}")
        
        if gen_stats['gen_params']:
            print("\n## Genパラメータ定義（上位20）")
            print(f"{'パラメータ定義':<60} {'使用数':>10}")
            print("-" * 72)
            for param in gen_stats['gen_params']:
                param_def = param['param_definition'][:60] if len(param['param_definition']) > 60 else param['param_definition']
                print(f"{param_def:<60} {param['usage_count']:>10}")
    
    def save_report_json(self, filename: str = "analysis_report.json"):
        """レポートをJSONファイルとして保存"""
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'value_statistics': self.generate_value_statistics(),
            'connection_patterns': self.generate_connection_patterns(),
            'gen_statistics': self.generate_gen_statistics()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, cls=DecimalEncoder)
        
        print(f"\nレポートを{filename}に保存しました")

def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Max/MSPパッチ分析レポート生成')
    parser.add_argument('--config', default='scripts/db_settings.ini', help='設定ファイルパス')
    parser.add_argument('--json', help='JSON形式で保存するファイル名')
    
    args = parser.parse_args()
    
    reporter = AnalysisReporter(args.config)
    
    # レポート表示
    reporter.print_report()
    
    # JSON保存（オプション）
    if args.json:
        reporter.save_report_json(args.json)

if __name__ == '__main__':
    main()