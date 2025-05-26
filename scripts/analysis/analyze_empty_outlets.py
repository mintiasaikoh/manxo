#!/usr/bin/env python3
"""空のアウトレットタイプを詳しく分析"""

import sys
sys.path.append('/Users/mymac/manxo/scripts')
from db_connector import DatabaseConnector
import json

def main():
    db = DatabaseConnector('scripts/db_settings.ini')
    db.connect()
    
    print("=== 空のアウトレットタイプ分析 ===\n")
    
    # 1. 全体の統計
    result = db.execute_query("""
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN source_outlet_types::text = '[\"\"]' THEN 1 END) as single_empty,
        COUNT(CASE WHEN source_outlet_types::text LIKE '[\"\", \"\"%' THEN 1 END) as multi_empty,
        COUNT(CASE WHEN source_outlet_types::text NOT LIKE '%\"\"%' AND source_outlet_types IS NOT NULL THEN 1 END) as non_empty
    FROM object_connections
    """)
    r = result[0]
    print(f"総接続数: {r['total']:,}")
    print(f"単一の空文字列 ['']: {r['single_empty']:,} ({r['single_empty']/r['total']*100:.1f}%)")
    print(f"複数の空文字列を含む: {r['multi_empty']:,} ({r['multi_empty']/r['total']*100:.1f}%)")
    print(f"空でないタイプ: {r['non_empty']:,} ({r['non_empty']/r['total']*100:.1f}%)")
    
    # 2. オブジェクトタイプごとの分析（単一の空文字列のみ）
    print("\n\n=== 単一の空文字列 [''] を持つオブジェクト TOP30 ===")
    result = db.execute_query("""
    SELECT 
        source_object_type,
        COUNT(*) as count
    FROM object_connections
    WHERE source_outlet_types::text = '[\"\"]'
    GROUP BY source_object_type
    ORDER BY count DESC
    LIMIT 30
    """)
    
    for i, r in enumerate(result, 1):
        print(f"{i:2d}. {r['source_object_type']:20s} : {r['count']:,}")
    
    # 3. MSPオブジェクト（〜で終わる）の分析
    print("\n\n=== MSPオブジェクト（~で終わる）で空文字列のもの ===")
    result = db.execute_query("""
    SELECT 
        source_object_type,
        source_outlet_types,
        COUNT(*) as count
    FROM object_connections
    WHERE source_object_type LIKE '%~'
    AND source_outlet_types::text LIKE '%\"\"%'
    GROUP BY source_object_type, source_outlet_types
    ORDER BY count DESC
    LIMIT 20
    """)
    
    for r in result:
        print(f"{r['source_object_type']:20s} {str(r['source_outlet_types']):30s} : {r['count']:,}")
    
    # 4. よく使われるオブジェクトの現在の状態
    print("\n\n=== 主要オブジェクトの現在のアウトレットタイプ ===")
    common_objects = ['message', 'flonum', 'number', 'bang', 'toggle', 'metro', 
                      'cycle~', 'dac~', 'adc~', 'route', 'select', 'gate']
    
    for obj in common_objects:
        result = db.execute_query(f"""
        SELECT 
            source_outlet_types,
            COUNT(*) as count
        FROM object_connections
        WHERE source_object_type = '{obj}'
        GROUP BY source_outlet_types
        ORDER BY count DESC
        LIMIT 3
        """)
        
        print(f"\n{obj}:")
        for r in result:
            print(f"  {str(r['source_outlet_types']):30s} : {r['count']:,}")
    
    db.disconnect()

if __name__ == "__main__":
    main()