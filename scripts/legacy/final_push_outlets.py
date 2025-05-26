#!/usr/bin/env python3
"""最後のプッシュ - 主要オブジェクトのアウトレットタイプを改善"""

import sys
sys.path.append('/Users/mymac/manxo/scripts')
from db_connector import DatabaseConnector

def main():
    db = DatabaseConnector('scripts/db_settings.ini')
    db.connect()
    
    print("=== 最終プッシュ：主要オブジェクトの改善 ===\n")
    
    # 最も重要なオブジェクトのマッピング
    critical_improvements = {
        # メッセージング系（汎用型を維持）
        'message': [''],          # メッセージは汎用
        'inlet': [''],            # インレットは汎用
        'outlet': [],             # アウトレットは出力なし（別のパッチに）
        'r': [''],                # receiveは汎用
        'receive': [''],          # receiveは汎用  
        's': [],                  # sendは出力なし
        'send': [],               # sendは出力なし
        
        # プリペンド/アペンド系（汎用型）
        'prepend': [''],          # 汎用メッセージ
        'append': [''],           # 汎用メッセージ
        
        # ロード系
        'loadmess': [''],         # ロード時メッセージ（汎用）
        'loadbang': ['bang'],     # ロード時bang
        
        # 数値処理系
        'scale': ['float'],       # スケーリングされた値
        'clip': ['float'],        # クリップされた値  
        'round': ['int'],         # 丸められた整数
        'expr': ['float'],        # 式の結果（通常float）
        'vexpr': ['list'],        # ベクトル式の結果
        'if': [''],               # 条件分岐は汎用
        
        # 変換系
        'tosymbol': ['symbol'],   # シンボルに変換
        'fromsymbol': [''],       # シンボルから変換（汎用）
        'atoi': ['int'],          # ASCII to int
        'itoa': ['symbol'],       # int to ASCII
        
        # スイッチ/ゲート系
        'switch': [''],           # スイッチは入力をそのまま
        'gate': [''],             # ゲートも入力をそのまま
        
        # パラメータ系
        'param': [''],            # パラメータは汎用
        'attrui': [''],           # 属性UIは汎用
        
        # ランダム系
        'random': ['int'],        # ランダム整数
        'drunk': ['int'],         # ドランクウォーク整数
        'urn': ['int', 'bang'],   # 整数とbang（全て出力時）
        
        # イテレータ系
        'iter': [''],             # リストの各要素（汎用）
        
        # Live API系
        'live.object': ['', ''], # id, result
        'live.path': ['', ''],    # path, id
        'live.observer': [''],    # observed value
        
        # 値保持系
        'v': [''],                # value holder（汎用）
        'value': [''],            # value holder（汎用）
        'pv': [''],               # patcher value（汎用）
        
        # 履歴系
        'history': ['float'],     # 前の値
        
        # パッチ制御系
        'pcontrol': [''],         # パッチ制御メッセージ
        'thispatcher': [],        # 出力なし（パッチ制御のみ）
        
        # ストレージ系
        'pattrstorage': ['', '', ''],  # 複数の汎用出力
        
        # その他
        'in': [''],               # 汎用入力
        'out': [],                # 出力なし（別のコンテキストへ）
        'thru': [''],             # パススルー（汎用）
        'speedlim': [''],         # 速度制限されたメッセージ（汎用）
        'deferlow': [''],         # 遅延メッセージ（汎用）
        'sprintf': ['symbol'],    # フォーマット済み文字列
        'translate': [''],        # 変換結果（汎用）
        'zmap': ['float'],        # マッピングされた値
        'dict.iter': ['', ''],    # key, value
        'midiinfo': ['', ''],     # 複数の情報出力
        
        # JavaScript系
        'js': [''],               # JavaScriptは汎用出力
        'jsui': [''],             # JavaScript UIも汎用
        'node.script': [''],      # Node.jsも汎用
        
        # 特殊な名前
        '_': [''],                # アンダースコアは汎用
        '0': [''],                # 数字も汎用として扱う
        
        # ローカルバージョン
        'll.r': [''],             # local receiveも汎用
        'll.s': [],               # local sendは出力なし
    }
    
    print("処理対象オブジェクト:")
    total_to_improve = 0
    
    # 現在の状況確認と更新
    for obj_type, new_types in critical_improvements.items():
        check_query = """
        SELECT COUNT(*) as count
        FROM object_connections
        WHERE source_object_type = %s
        AND source_outlet_types = ARRAY['']::text[]
        """
        result = db.execute_query(check_query, (obj_type,))
        
        if result and result[0]['count'] > 0:
            count = result[0]['count']
            total_to_improve += count
            print(f"  {obj_type:20s}: {count:6,} 件 → {new_types if new_types else '[]'}")
    
    print(f"\n合計: {total_to_improve:,} 件を改善予定")
    
    # 実際の更新処理
    print("\n更新処理開始...")
    updated = 0
    
    for obj_type, new_types in critical_improvements.items():
        if new_types is not None:  # Noneでない場合
            if new_types:  # 空リストでない場合
                update_query = """
                UPDATE object_connections
                SET source_outlet_types = %s::text[]
                WHERE source_object_type = %s
                AND source_outlet_types = ARRAY['']::text[]
                """
                result = db.execute_query(update_query, (new_types, obj_type))
            else:  # 空リスト = 出力なし
                update_query = """
                UPDATE object_connections
                SET source_outlet_types = ARRAY[]::text[]
                WHERE source_object_type = %s
                AND source_outlet_types = ARRAY['']::text[]
                """
                result = db.execute_query(update_query, (obj_type,))
    
    # 最終結果
    print("\n=== 最終結果 ===")
    result = db.execute_query("""
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN source_outlet_types = ARRAY['']::text[] THEN 1 END) as single_empty,
        COUNT(CASE WHEN array_length(source_outlet_types, 1) > 0 
                   AND source_outlet_types != ARRAY['']::text[] THEN 1 END) as has_info,
        COUNT(CASE WHEN array_length(source_outlet_types, 1) = 0 THEN 1 END) as no_output
    FROM object_connections
    """)
    
    r = result[0]
    print(f"総接続数: {r['total']:,}")
    print(f"単一の空文字列のみ: {r['single_empty']:,} ({r['single_empty']/r['total']*100:.1f}%)")
    print(f"型情報あり: {r['has_info']:,} ({r['has_info']/r['total']*100:.1f}%)")
    print(f"出力なし: {r['no_output']:,}")
    print(f"\n✨ アウトレット型情報の充実度: {100 - r['single_empty']/r['total']*100:.1f}% ✨")
    
    # 残っているものを確認
    if r['single_empty'] > 0:
        print("\n残っている単一空文字列のオブジェクト（TOP10）:")
        remaining = db.execute_query("""
        SELECT source_object_type, COUNT(*) as count
        FROM object_connections
        WHERE source_outlet_types = ARRAY['']::text[]
        GROUP BY source_object_type
        ORDER BY count DESC
        LIMIT 10
        """)
        for rem in remaining:
            print(f"  {rem['source_object_type']:20s}: {rem['count']:,}")
    
    db.disconnect()

if __name__ == "__main__":
    main()