#!/usr/bin/env python3
"""アウトレットタイプ情報を改善する"""

import sys
sys.path.append('/Users/mymac/manxo/scripts')
from db_connector import DatabaseConnector
import json

def main():
    db = DatabaseConnector('scripts/db_settings.ini')
    db.connect()
    
    print("=== アウトレットタイプの改善 ===\n")
    
    # 1. 空文字列のアウトレットタイプを持つオブジェクトを分析
    query = """
    SELECT DISTINCT
        source_object_type,
        source_outlet_types,
        COUNT(*) as count
    FROM object_connections
    WHERE source_outlet_types::text LIKE '%\"\",%' OR source_outlet_types::text = '[\"\"]'
    GROUP BY source_object_type, source_outlet_types
    ORDER BY count DESC
    LIMIT 30
    """
    
    print("空文字列のアウトレットタイプを持つオブジェクト TOP30:")
    results = db.execute_query(query)
    for r in results:
        print(f"  {r['source_object_type']:20s} {str(r['source_outlet_types']):20s} : {r['count']:,}")
    
    # 2. パターンベースの改善案
    improvements = {
        # MSP（オーディオ）オブジェクトは通常 signal を出力
        'dac~': ['signal'],
        'adc~': ['signal'],
        'cycle~': ['signal'],
        'noise~': ['signal'],
        'saw~': ['signal'],
        'rect~': ['signal'],
        'tri~': ['signal'],
        'phasor~': ['signal'],
        'line~': ['signal'],
        'sig~': ['signal'],
        '*~': ['signal'],
        '+~': ['signal'],
        '-~': ['signal'],
        '/~': ['signal'],
        
        # 数値系オブジェクト
        'number': ['float'],
        'flonum': ['float'],
        'int': ['int'],
        'float': ['float'],
        
        # メッセージ系
        'message': [''],  # メッセージは汎用
        'prepend': [''],  # prepend も汎用
        'append': [''],   # append も汎用
        
        # トリガー系
        'bang': ['bang'],
        'button': ['bang'],
        'toggle': ['int'],
        'sel': ['bang', ''],  # 第1アウトレット=マッチ、第2=パススルー
        'select': ['bang', ''],
        'route': ['', ''],     # routeは複数の汎用アウトレット
        
        # タイミング系
        'metro': ['bang'],
        'delay': ['bang'],
        'pipe': [''],
        'timer': ['float'],
        
        # リスト系
        'pack': ['list'],
        'unpack': ['', '', '', ''],  # unpackは要素数に応じて
        'zl': ['list', 'int'],       # 第1=リスト、第2=長さなど
        
        # MIDI系
        'notein': ['int', 'int', 'int'],  # pitch, velocity, channel
        'noteout': [],  # 出力なし
        'ctlin': ['int', 'int', 'int'],    # value, controller, channel
        'ctlout': [],   # 出力なし
        
        # Jitter系
        'jit.movie': ['jit_matrix', ''],
        'jit.grab': ['jit_matrix', ''],
        'jit.matrix': ['jit_matrix', ''],
        'jit.pwindow': [],  # 表示のみ
    }
    
    print("\n\n=== 改善可能なオブジェクト ===")
    
    # 改善対象を特定
    update_count = 0
    for obj_type, new_types in improvements.items():
        # 現在の状況を確認
        check_query = f"""
        SELECT COUNT(*) as count
        FROM object_connections
        WHERE source_object_type = '{obj_type}'
        AND (source_outlet_types::text LIKE '%\"\",%' OR source_outlet_types::text = '[\"\"]')
        """
        result = db.execute_query(check_query)
        if result and result[0]['count'] > 0:
            count = result[0]['count']
            print(f"\n{obj_type}: {count:,} 件を改善可能")
            print(f"  現在: [''] → 改善後: {new_types}")
            
            # 実際に更新
            if new_types:  # 空リストでない場合のみ更新
                update_query = f"""
                UPDATE object_connections
                SET source_outlet_types = '{json.dumps(new_types)}'::text[]
                WHERE source_object_type = '{obj_type}'
                AND (source_outlet_types::text LIKE '%\"\",%' OR source_outlet_types::text = '[\"\"]')
                """
                db.execute_query(update_query)
                update_count += count
    
    print(f"\n\n合計 {update_count:,} 件のアウトレットタイプを改善しました！")
    
    # 3. 改善後の統計
    print("\n=== 改善後の統計 ===")
    result = db.execute_query("""
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN source_outlet_types::text LIKE '%\"\",%' OR source_outlet_types::text = '[\"\"]' THEN 1 END) as empty_types
    FROM object_connections
    """)
    r = result[0]
    print(f"総接続数: {r['total']:,}")
    print(f"空のアウトレットタイプ: {r['empty_types']:,} ({r['empty_types']/r['total']*100:.1f}%)")
    print(f"改善済み: {r['total'] - r['empty_types']:,} ({(r['total'] - r['empty_types'])/r['total']*100:.1f}%)")
    
    db.disconnect()

if __name__ == "__main__":
    main()