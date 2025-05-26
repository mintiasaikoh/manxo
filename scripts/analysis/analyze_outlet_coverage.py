#!/usr/bin/env python3
"""アウトレットタイプのカバレッジを分析"""

import sys
sys.path.append('/Users/mymac/manxo/scripts')
from db_connector import DatabaseConnector
import json

def main():
    db = DatabaseConnector('scripts/db_settings.ini')
    db.connect()
    
    print("=== アウトレットタイプカバレッジ分析 ===\n")
    
    # 1. object_connectionsテーブルの source_outlet_types を確認
    query1 = """
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN source_outlet_types IS NOT NULL AND array_length(source_outlet_types, 1) > 0 THEN 1 END) as has_outlet_types,
        COUNT(CASE WHEN source_outlet_types IS NULL OR array_length(source_outlet_types, 1) IS NULL THEN 1 END) as no_outlet_types
    FROM object_connections
    """
    result = db.execute_query(query1)
    if result:
        r = result[0]
        print(f"object_connections テーブル:")
        print(f"  総レコード数: {r['total']:,}")
        print(f"  アウトレットタイプあり: {r['has_outlet_types']:,} ({r['has_outlet_types']/r['total']*100:.1f}%)")
        print(f"  アウトレットタイプなし: {r['no_outlet_types']:,} ({r['no_outlet_types']/r['total']*100:.1f}%)")
        print()
    
    # 2. objectsテーブルの outlet_types を確認
    query2 = """
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN outlet_types IS NOT NULL AND jsonb_array_length(outlet_types) > 0 THEN 1 END) as has_outlet_types,
        COUNT(CASE WHEN outlet_types IS NULL OR jsonb_array_length(outlet_types) = 0 THEN 1 END) as no_outlet_types
    FROM objects
    """
    result = db.execute_query(query2)
    if result:
        r = result[0]
        print(f"objects テーブル:")
        print(f"  総オブジェクト数: {r['total']:,}")
        print(f"  アウトレットタイプあり: {r['has_outlet_types']:,} ({r['has_outlet_types']/r['total']*100:.1f}%)")
        print(f"  アウトレットタイプなし: {r['no_outlet_types']:,} ({r['no_outlet_types']/r['total']*100:.1f}%)")
        print()
    
    # 3. アウトレット情報がないオブジェクトTOP20
    query3 = """
    SELECT 
        source_object_type,
        COUNT(*) as connection_count
    FROM object_connections
    WHERE source_outlet_types IS NULL OR array_length(source_outlet_types, 1) IS NULL
    GROUP BY source_object_type
    ORDER BY connection_count DESC
    LIMIT 20
    """
    print("アウトレット情報がない接続の多いオブジェクトTOP20:")
    results = db.execute_query(query3)
    for i, r in enumerate(results, 1):
        print(f"  {i:2d}. {r['source_object_type']:20s} : {r['connection_count']:,} 接続")
    print()
    
    # 4. アウトレット情報があるオブジェクトの例
    query4 = """
    SELECT DISTINCT 
        source_object_type,
        source_outlet_types
    FROM object_connections
    WHERE source_outlet_types IS NOT NULL 
    AND array_length(source_outlet_types, 1) > 0
    AND source_outlet_types::text NOT LIKE '%unknown%'
    LIMIT 20
    """
    print("アウトレット情報があるオブジェクトの例:")
    results = db.execute_query(query4)
    for r in results:
        print(f"  {r['source_object_type']:20s} : {r['source_outlet_types']}")
    
    db.disconnect()

if __name__ == "__main__":
    main()