#!/usr/bin/env python3
"""スマートなアウトレットタイプ改善戦略"""

import sys
sys.path.append('/Users/mymac/manxo/scripts')
from db_connector import DatabaseConnector
import json

def main():
    db = DatabaseConnector('scripts/db_settings.ini')
    db.connect()
    
    print("=== スマートなアウトレットタイプ改善 ===\n")
    
    # まず現在の状況を正確に把握
    result = db.execute_query("""
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN source_outlet_types = ARRAY['']::text[] THEN 1 END) as single_empty,
        COUNT(CASE WHEN source_outlet_types[1] = '' THEN 1 END) as first_empty,
        COUNT(CASE WHEN array_to_string(source_outlet_types, '') = '' THEN 1 END) as all_empty
    FROM object_connections
    """)
    r = result[0]
    print(f"総接続数: {r['total']:,}")
    print(f"単一の空文字列 ['']: {r['single_empty']:,} ({r['single_empty']/r['total']*100:.1f}%)")
    print(f"最初が空文字列: {r['first_empty']:,} ({r['first_empty']/r['total']*100:.1f}%)")
    print(f"全て空文字列: {r['all_empty']:,} ({r['all_empty']/r['total']*100:.1f}%)")
    
    # 改善マッピング（より包括的）
    improvements = {
        # メッセージ系 - 汎用を維持
        'message': [''],  # メッセージは汎用のまま
        
        # 数値系 - 明確な型指定
        'number': ['int', 'bang'],     # numberはintを出力
        'flonum': ['float', 'bang'],   # flonumはfloatを出力
        'int': ['int'],
        'float': ['float'],
        'i': ['int'],
        'f': ['float'],
        
        # MSPオブジェクト（基本）
        'cycle~': ['signal'],
        'saw~': ['signal'],
        'rect~': ['signal'],
        'tri~': ['signal'],
        'noise~': ['signal'],
        'pink~': ['signal'],
        'phasor~': ['signal'],
        'sig~': ['signal'],
        'line~': ['signal'],
        'curve~': ['signal'],
        
        # MSP演算
        '*~': ['signal'],
        '+~': ['signal'],
        '-~': ['signal'],
        '/~': ['signal'],
        '!-~': ['signal'],
        '!/~': ['signal'],
        '%~': ['signal'],
        '==~': ['signal'],
        '!=~': ['signal'],
        '<~': ['signal'],
        '>~': ['signal'],
        '<=~': ['signal'],
        '>=~': ['signal'],
        
        # タイミング系
        'metro': ['bang'],
        'delay': ['bang'],
        'pipe': [''],  # pipeは汎用
        'timer': ['float'],
        'clocker': ['float'],
        
        # 制御系
        'bang': ['bang'],
        'button': ['bang'],
        'toggle': ['int'],
        'gate': [''],  # gateは入力をそのまま通す
        
        # リスト処理
        'pack': ['list'],
        'pak': ['list'],
        'join': ['list'],
        'zl.join': ['list'],
        'zl.group': ['list'],
        'zl.reg': ['list'],
        
        # MIDI系
        'notein': ['int', 'int', 'int'],      # pitch, velocity, channel
        'ctlin': ['int', 'int', 'int'],       # value, controller, channel
        'bendin': ['int', 'int'],              # value, channel
        'pgmin': ['int', 'int'],               # program, channel
        'touchin': ['int', 'int'],             # pressure, channel
        'midiin': ['int', 'int'],              # byte, port
        
        # Jitter系
        'jit.movie': ['jit_matrix', ''],
        'jit.grab': ['jit_matrix', ''],
        'jit.matrix': ['jit_matrix', ''],
        'jit.qt.movie': ['jit_matrix', ''],
        'jit.noise': ['jit_matrix', ''],
        
        # Live UI系
        'live.dial': ['float', 'float'],       # value, raw value
        'live.slider': ['float', 'float'],
        'live.numbox': ['float', 'float'],
        'live.toggle': ['int'],
        'live.button': ['bang'],
        'live.tab': ['int'],
        'live.text': ['', ''],  # 汎用出力
        
        # 演算系
        '+': ['float'],  # または int（入力依存）
        '-': ['float'],
        '*': ['float'],
        '/': ['float'],
        '%': ['int'],
        '==': ['int'],   # 比較結果は0/1
        '!=': ['int'],
        '<': ['int'],
        '>': ['int'],
        '<=': ['int'],
        '>=': ['int'],
        '&&': ['int'],
        '||': ['int'],
        '!': ['int'],
    }
    
    print("\n\n=== 改善処理開始 ===")
    total_improved = 0
    
    for obj_type, new_types in improvements.items():
        # 現在の単一空文字列を持つ接続を確認
        check_query = """
        SELECT COUNT(*) as count
        FROM object_connections
        WHERE source_object_type = %s
        AND source_outlet_types = ARRAY['']::text[]
        """
        result = db.execute_query(check_query, (obj_type,))
        
        if result and result[0]['count'] > 0:
            count = result[0]['count']
            
            # 既に適切なタイプを持つものを確認
            existing_query = """
            SELECT DISTINCT source_outlet_types, COUNT(*) as cnt
            FROM object_connections
            WHERE source_object_type = %s
            AND source_outlet_types != ARRAY['']::text[]
            GROUP BY source_outlet_types
            ORDER BY cnt DESC
            LIMIT 3
            """
            existing = db.execute_query(existing_query, (obj_type,))
            
            print(f"\n{obj_type}: {count:,} 件")
            if existing:
                print(f"  既存のタイプ例:")
                for e in existing:
                    print(f"    {e['source_outlet_types']} : {e['cnt']:,} 件")
            print(f"  改善: [''] → {new_types}")
            
            # 更新を実行
            if new_types != ['']:  # 汎用の [''] のままにしないもののみ更新
                update_query = """
                UPDATE object_connections
                SET source_outlet_types = %s::text[]
                WHERE source_object_type = %s
                AND source_outlet_types = ARRAY['']::text[]
                """
                db.execute_query(update_query, (new_types, obj_type))
                total_improved += count
    
    print(f"\n\n=== 完了: 合計 {total_improved:,} 件を改善 ===")
    
    # 改善後の統計
    result = db.execute_query("""
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN source_outlet_types = ARRAY['']::text[] THEN 1 END) as single_empty,
        COUNT(CASE WHEN source_outlet_types[1] != '' OR array_length(source_outlet_types, 1) = 1 THEN 1 END) as has_type
    FROM object_connections
    """)
    r = result[0]
    print(f"\n改善後の統計:")
    print(f"  総接続数: {r['total']:,}")
    print(f"  単一の空文字列: {r['single_empty']:,} ({r['single_empty']/r['total']*100:.1f}%)")
    print(f"  型情報あり: {r['has_type']:,} ({r['has_type']/r['total']*100:.1f}%)")
    
    db.disconnect()

if __name__ == "__main__":
    main()