#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
無効なオブジェクトエントリのクリーンアップ
ドキュメントページのタイトルや非オブジェクトエントリを削除し、
実際のオブジェクトのポート情報を修正する
"""

import sys
import configparser
from db_connector import DatabaseConnector

def cleanup_invalid_objects():
    """無効なオブジェクトエントリをクリーンアップ"""
    
    db = DatabaseConnector('scripts/db_settings.ini')
    
    try:
        db.connect()
        
        print("=== 無効オブジェクトのクリーンアップ開始 ===")
        
        # 1. 明らかにドキュメントページのタイトルであるエントリを削除
        invalid_patterns = [
            "Messages",
            "Features", 
            "Properties",
            "Operators",
            "Object (",
            "Wrapper ",
            "GL Object"
        ]
        
        deleted_count = 0
        for pattern in invalid_patterns:
            query = f"DELETE FROM objects WHERE object_name LIKE '%{pattern}%'"
            result = db.execute_query(query)
            print(f"削除パターン '{pattern}': 削除")
        
        # 削除後の確認
        query = "SELECT COUNT(*) as total FROM objects"
        result = db.execute_query(query)
        new_total = result[0]['total']
        print(f"クリーンアップ後の総オブジェクト数: {new_total}")
        
        # 2. 実際のオブジェクトの既知のポート情報を修正
        known_ports = {
            'buffer': {'inlets': 1, 'outlets': 4},
            'bpatcher': {'inlets': 1, 'outlets': 1}, 
            'codebox': {'inlets': 1, 'outlets': 1},
            'comment': {'inlets': 0, 'outlets': 0},
            'active': {'inlets': 1, 'outlets': 1},
            'loadmess': {'inlets': 1, 'outlets': 1},
            'message': {'inlets': 2, 'outlets': 1},
            'number': {'inlets': 2, 'outlets': 1},
            'toggle': {'inlets': 1, 'outlets': 1},
            'button': {'inlets': 1, 'outlets': 1},
            'slider': {'inlets': 1, 'outlets': 1},
            'dial': {'inlets': 1, 'outlets': 1},
            'gain~': {'inlets': 2, 'outlets': 1},
            'live.gain~': {'inlets': 2, 'outlets': 1},
            'dict.view': {'inlets': 1, 'outlets': 1},
            'data': {'inlets': 1, 'outlets': 1},
            'channels': {'inlets': 1, 'outlets': 1},
            'frame': {'inlets': 1, 'outlets': 1},
            'length': {'inlets': 1, 'outlets': 1},
            'ms': {'inlets': 1, 'outlets': 1},
            'size': {'inlets': 1, 'outlets': 1}
        }
        
        updated_count = 0
        for obj_name, port_info in known_ports.items():
            query = """
            UPDATE objects 
            SET num_inlets = %s, num_outlets = %s 
            WHERE object_name = %s AND (num_inlets = 0 AND num_outlets = 0)
            """
            # PostgreSQLの場合、execute_queryではなく直接実行
            try:
                cursor = db.connection.cursor()
                cursor.execute(query, (port_info['inlets'], port_info['outlets'], obj_name))
                if cursor.rowcount > 0:
                    updated_count += 1
                    print(f"更新: {obj_name} -> inlets={port_info['inlets']}, outlets={port_info['outlets']}")
                cursor.close()
                db.connection.commit()
            except Exception as e:
                print(f"更新エラー {obj_name}: {e}")
        
        print(f"\nポート情報を更新したオブジェクト数: {updated_count}")
        
        # 3. 最終統計
        query = """
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN num_inlets > 0 OR num_outlets > 0 THEN 1 END) as with_ports,
            COUNT(CASE WHEN num_inlets = 0 AND num_outlets = 0 THEN 1 END) as without_ports
        FROM objects
        """
        result = db.execute_query(query)
        stats = result[0]
        
        print(f"\n=== クリーンアップ後の統計 ===")
        print(f"総オブジェクト数: {stats['total']}")
        print(f"ポート情報あり: {stats['with_ports']}")
        print(f"ポート情報なし: {stats['without_ports']}")
        print(f"ポート情報カバー率: {stats['with_ports']/stats['total']*100:.1f}%")
        
    finally:
        db.disconnect()

if __name__ == "__main__":
    cleanup_invalid_objects()