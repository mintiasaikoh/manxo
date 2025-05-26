#!/usr/bin/env python3
"""
重複データの削除と分析
object_connectionsテーブル用に更新
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from configparser import ConfigParser
import logging

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def connect_to_db():
    """データベースに接続"""
    config = ConfigParser()
    # スクリプトディレクトリからの相対パスで設定ファイルを読み込み
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'db_settings.ini')
    config.read(config_path)
    
    conn = psycopg2.connect(
        host=config['database']['host'],
        port=config['database']['port'],
        database=config['database']['database'],
        user=config['database']['user'],
        password=config['database']['password']
    )
    return conn

def analyze_duplicates():
    """重複の詳細分析"""
    conn = connect_to_db()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    print("=== 重複分析開始 ===")
    
    # 重複前の統計
    cursor.execute("SELECT COUNT(*) FROM object_connections")
    total_before = cursor.fetchone()['count']
    
    # ユニーク接続数
    cursor.execute("""
        SELECT COUNT(DISTINCT (patch_file, source_object_type, source_port, target_object_type, target_port))
        FROM object_connections
    """)
    unique_connections = cursor.fetchone()['count']
    
    print(f"重複削除前:")
    print(f"  総接続数: {total_before:,}")
    print(f"  ユニーク接続数: {unique_connections:,}")
    print(f"  重複数: {total_before - unique_connections:,}")
    print(f"  重複率: {((total_before - unique_connections) / total_before * 100):.1f}%")
    
    # 最も重複の多いファイルの詳細
    cursor.execute("""
        SELECT 
            patch_file,
            COUNT(*) as total_connections,
            COUNT(DISTINCT (source_object_type, source_port, target_object_type, target_port)) as unique_connections,
            COUNT(*) - COUNT(DISTINCT (source_object_type, source_port, target_object_type, target_port)) as duplicates
        FROM object_connections
        GROUP BY patch_file
        HAVING COUNT(*) > COUNT(DISTINCT (source_object_type, source_port, target_object_type, target_port))
        ORDER BY duplicates DESC
        LIMIT 10
    """)
    
    most_duplicated = cursor.fetchall()
    print(f"\n最も重複の多いファイル:")
    for patch in most_duplicated:
        filename = patch['patch_file'].split('/')[-1] if '/' in patch['patch_file'] else patch['patch_file']
        print(f"  {filename[:50]}: {patch['total_connections']}総数 → {patch['unique_connections']}ユニーク ({patch['duplicates']}重複)")
    
    cursor.close()
    conn.close()

def cleanup_duplicates():
    """重複データの削除"""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    print(f"\n=== 重複削除開始 ===")
    
    # 重複削除前のカウント
    cursor.execute("SELECT COUNT(*) FROM object_connections")
    before_count = cursor.fetchone()[0]
    
    # 重複削除のクエリ
    # 各ユニークな接続組み合わせで最新（最大ID）のみを残す
    delete_query = """
        WITH duplicates AS (
            SELECT id,
                   ROW_NUMBER() OVER (
                       PARTITION BY patch_file, source_object_type, source_port, 
                                  target_object_type, target_port 
                       ORDER BY created_at DESC, id DESC
                   ) as rn
            FROM object_connections
        )
        DELETE FROM object_connections
        WHERE id IN (SELECT id FROM duplicates WHERE rn > 1)
    """
    
    logger.info("重複削除実行中...")
    cursor.execute(delete_query)
    deleted_count = cursor.rowcount
    
    # 削除後のカウント
    cursor.execute("SELECT COUNT(*) FROM object_connections")
    after_count = cursor.fetchone()[0]
    
    conn.commit()
    
    print(f"重複削除結果:")
    print(f"  削除前: {before_count:,} レコード")
    print(f"  削除後: {after_count:,} レコード")
    print(f"  削除数: {deleted_count:,} レコード")
    print(f"  削除率: {(deleted_count / before_count * 100):.1f}%")
    
    # object_detailsテーブルの重複も削除
    if deleted_count > 0:
        print(f"\n=== object_detailsテーブルの重複削除 ===")
        
        cursor.execute("SELECT COUNT(*) FROM object_details")
        obj_before = cursor.fetchone()[0]
        
        obj_delete_query = """
            WITH duplicates AS (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY patch_file, object_id 
                           ORDER BY created_at DESC, id DESC
                       ) as rn
                FROM object_details
            )
            DELETE FROM object_details
            WHERE id IN (SELECT id FROM duplicates WHERE rn > 1)
        """
        
        cursor.execute(obj_delete_query)
        obj_deleted = cursor.rowcount
        
        cursor.execute("SELECT COUNT(*) FROM object_details")
        obj_after = cursor.fetchone()[0]
        
        conn.commit()
        
        print(f"object_details削除結果:")
        print(f"  削除前: {obj_before:,} レコード")
        print(f"  削除後: {obj_after:,} レコード")
        print(f"  削除数: {obj_deleted:,} レコード")
    
    cursor.close()
    conn.close()

def add_unique_constraint():
    """ユニーク制約の追加"""
    conn = connect_to_db()
    cursor = conn.cursor()
    
    print(f"\n=== ユニーク制約追加 ===")
    
    try:
        # ユニーク制約を追加
        constraint_query = """
            ALTER TABLE object_connections 
            ADD CONSTRAINT unique_object_connection 
            UNIQUE (patch_file, source_object_type, source_port, target_object_type, target_port)
        """
        
        cursor.execute(constraint_query)
        conn.commit()
        print("object_connectionsにユニーク制約を正常に追加しました")
        
    except psycopg2.errors.DuplicateObject as e:
        conn.rollback()
        print("object_connectionsのユニーク制約は既に存在します")
    except Exception as e:
        conn.rollback()
        print(f"ユニーク制約追加エラー: {e}")
    
    # object_detailsにもユニーク制約を追加
    try:
        obj_constraint_query = """
            ALTER TABLE object_details 
            ADD CONSTRAINT unique_object_detail 
            UNIQUE (patch_file, object_id)
        """
        
        cursor.execute(obj_constraint_query)
        conn.commit()
        print("object_detailsにユニーク制約を正常に追加しました")
        
    except psycopg2.errors.DuplicateObject as e:
        conn.rollback()
        print("object_detailsのユニーク制約は既に存在します")
    except Exception as e:
        conn.rollback()
        print(f"object_detailsユニーク制約追加エラー: {e}")
    
    cursor.close()
    conn.close()

def final_statistics():
    """最終統計"""
    conn = connect_to_db()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    print(f"\n=== 最終統計 ===")
    
    # 接続数
    cursor.execute("SELECT COUNT(*) FROM object_connections")
    total_connections = cursor.fetchone()['count']
    
    # パッチ数
    cursor.execute("SELECT COUNT(DISTINCT patch_file) FROM object_connections")
    unique_patches = cursor.fetchone()['count']
    
    # オブジェクト数
    cursor.execute("SELECT COUNT(*) FROM object_details")
    total_objects = cursor.fetchone()['count']
    
    # 最も使用されているオブジェクト（ソース側）
    cursor.execute("""
        SELECT 
            source_object_type,
            COUNT(*) as usage_count
        FROM object_connections
        WHERE source_object_type IS NOT NULL AND source_object_type != ''
        GROUP BY source_object_type
        ORDER BY usage_count DESC
        LIMIT 15
    """)
    
    top_sources = cursor.fetchall()
    
    # 最も接続を受けるオブジェクト（ターゲット側）
    cursor.execute("""
        SELECT 
            target_object_type,
            COUNT(*) as usage_count
        FROM object_connections
        WHERE target_object_type IS NOT NULL AND target_object_type != ''
        GROUP BY target_object_type
        ORDER BY usage_count DESC
        LIMIT 15
    """)
    
    top_targets = cursor.fetchall()
    
    print(f"データベース統計:")
    print(f"  総接続数: {total_connections:,}")
    print(f"  分析済みパッチ数: {unique_patches:,}")
    print(f"  総オブジェクト数: {total_objects:,}")
    
    print(f"\n最も使用されているソースオブジェクト:")
    for obj in top_sources:
        print(f"  {obj['source_object_type']}: {obj['usage_count']:,}回")
    
    print(f"\n最も接続を受けるターゲットオブジェクト:")
    for obj in top_targets:
        print(f"  {obj['target_object_type']}: {obj['usage_count']:,}回")
    
    # 位置情報とポートタイプ情報の統計
    cursor.execute("""
        SELECT 
            COUNT(*) FILTER (WHERE source_position IS NOT NULL) as with_position,
            COUNT(*) FILTER (WHERE source_outlet_types IS NOT NULL) as with_port_types,
            COUNT(*) as total
        FROM object_connections
    """)
    
    info_stats = cursor.fetchone()
    print(f"\n追加情報の統計:")
    print(f"  位置情報を持つ接続: {info_stats['with_position']:,} ({info_stats['with_position']/info_stats['total']*100:.1f}%)")
    print(f"  ポートタイプ情報を持つ接続: {info_stats['with_port_types']:,} ({info_stats['with_port_types']/info_stats['total']*100:.1f}%)")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    analyze_duplicates()
    cleanup_duplicates()
    add_unique_constraint()
    final_statistics()