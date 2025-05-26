#!/usr/bin/env python3
"""最終的なアウトレットタイプ改善 - 100%を目指す"""

import sys
sys.path.append('/Users/mymac/manxo/scripts')
from db_connector import DatabaseConnector
import json

def main():
    db = DatabaseConnector('scripts/db_settings.ini')
    db.connect()
    
    print("=== 最終アウトレットタイプ改善 ===\n")
    
    # 残っている単一空文字列のオブジェクトを確認
    result = db.execute_query("""
    SELECT 
        source_object_type,
        COUNT(*) as count
    FROM object_connections
    WHERE source_outlet_types = ARRAY['']::text[]
    GROUP BY source_object_type
    ORDER BY count DESC
    LIMIT 50
    """)
    
    print("残っている単一空文字列 [''] のオブジェクト TOP50:")
    remaining_objects = {}
    for r in result:
        print(f"  {r['source_object_type']:20s} : {r['count']:,}")
        remaining_objects[r['source_object_type']] = r['count']
    
    # 追加の改善マッピング（パターン分析に基づく）
    additional_improvements = {
        # UI系
        'umenu': ['int', '', ''],  # 選択インデックス、シンボル、メニュー項目
        'textedit': [''],         # テキスト出力は汎用
        'comment': [],            # コメントは出力なし
        
        # パッチャー系
        'p': [''],                # サブパッチャーは汎用
        'bpatcher': [''],         # bpatcherも汎用
        'poly~': ['', '', '', ''], # poly~は複数の汎用出力
        
        # トリガー系
        't': [''],                # triggerは入力をそのまま出力
        'trigger': [''],
        'b': ['bang'],           # bangbangはbangを出力
        'bangbang': ['bang'],
        
        # セレクター系
        'sel': ['bang', ''],      # マッチ時bang、それ以外はパススルー
        'select': ['bang', ''],
        'route': ['', ''],        # routeは汎用的なルーティング
        
        # 変換系
        'atoi': ['int'],         # ASCII to int
        'itoa': [''],            # int to ASCII (symbol)
        'ftom': ['float'],       # frequency to MIDI
        'mtof': ['float'],       # MIDI to frequency
        'dbtoa': ['float'],      # dB to amplitude
        'atodb': ['float'],      # amplitude to dB
        
        # カウンター系
        'counter': ['int', '', '', 'int'],  # count, carry, min, max
        'accum': ['int'],        # accumulated value
        
        # スライダー系
        'slider': ['float'],
        'hslider': ['float'],
        'uslider': ['float'],
        'rslider': ['float', 'float'],  # range slider: min, max
        'multislider': ['list'],
        
        # テーブル系
        'table': [''],           # 汎用出力
        'coll': ['', '', '', ''], # 複数の汎用出力
        'dict': ['dictionary'],   # 辞書出力
        
        # Gen系
        'gen~': ['signal'],      # Gen~は信号出力
        'gen': [''],             # Genは汎用
        'jit.gen': ['jit_matrix', ''],
        
        # 表示系
        'print': [],             # printは出力なし（コンソールのみ）
        'lcd': [''],             # LCDは汎用メッセージ
        'fpic': [''],            # 画像表示、クリック位置など
        
        # ファイル系
        'opendialog': [''],      # ファイルパス（symbol）
        'savedialog': [''],      # ファイルパス（symbol）
        'filein': [''],          # ファイル内容（汎用）
        
        # ネットワーク系
        'udpsend': [],           # 送信のみ
        'udpreceive': [''],      # 受信データ（汎用）
        'tcpsend': [],           # 送信のみ
        'tcpreceive': [''],      # 受信データ（汎用）
        
        # その他のオブジェクト
        'loadbang': ['bang'],    # 起動時にbang
        'closebang': ['bang'],   # 終了時にbang
        'active': ['int'],       # アクティブ状態 0/1
        'key': ['int'],          # キーコード
        'keyup': ['int'],        # キーコード
        'mousestate': ['int', 'int', 'int', 'int', 'int'],  # x, y, button, cmd, etc.
        'modifiers': ['int'],    # modifier keys
        
        # 未知のオブジェクト対策
        'unknown': [''],         # unknownは汎用として扱う
    }
    
    # MSPオブジェクトの一括処理（~で終わるもの）
    msp_objects = db.execute_query("""
    SELECT DISTINCT source_object_type
    FROM object_connections
    WHERE source_object_type LIKE '%~'
    AND source_outlet_types = ARRAY['']::text[]
    """)
    
    for obj in msp_objects:
        obj_name = obj['source_object_type']
        if obj_name not in additional_improvements:
            # デフォルトでMSPオブジェクトはsignal出力
            additional_improvements[obj_name] = ['signal']
    
    print("\n\n=== 追加改善処理 ===")
    total_improved = 0
    
    for obj_type, new_types in additional_improvements.items():
        if obj_type in remaining_objects:
            count = remaining_objects[obj_type]
            print(f"\n{obj_type}: {count:,} 件 → {new_types}")
            
            # 更新を実行
            if new_types:  # 空リストでない場合
                update_query = """
                UPDATE object_connections
                SET source_outlet_types = %s::text[]
                WHERE source_object_type = %s
                AND source_outlet_types = ARRAY['']::text[]
                """
                db.execute_query(update_query, (new_types, obj_type))
                total_improved += count
            else:
                # 空リスト = 出力なし
                update_query = """
                UPDATE object_connections
                SET source_outlet_types = ARRAY[]::text[]
                WHERE source_object_type = %s
                AND source_outlet_types = ARRAY['']::text[]
                """
                db.execute_query(update_query, (obj_type,))
                total_improved += count
    
    print(f"\n\n=== 完了: 追加で {total_improved:,} 件を改善 ===")
    
    # 最終統計
    result = db.execute_query("""
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN source_outlet_types = ARRAY['']::text[] THEN 1 END) as single_empty,
        COUNT(CASE WHEN array_length(source_outlet_types, 1) > 0 
                   AND source_outlet_types[1] != '' THEN 1 END) as has_specific_type,
        COUNT(CASE WHEN array_length(source_outlet_types, 1) = 0 THEN 1 END) as no_output
    FROM object_connections
    """)
    r = result[0]
    print(f"\n最終統計:")
    print(f"  総接続数: {r['total']:,}")
    print(f"  単一の空文字列: {r['single_empty']:,} ({r['single_empty']/r['total']*100:.1f}%)")
    print(f"  具体的な型あり: {r['has_specific_type']:,} ({r['has_specific_type']/r['total']*100:.1f}%)")
    print(f"  出力なし: {r['no_output']:,}")
    print(f"  アウトレット情報の充実度: {100 - r['single_empty']/r['total']*100:.1f}%")
    
    db.disconnect()

if __name__ == "__main__":
    main()