#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
connectionsテーブルからobject_connectionsテーブルへのデータ移行スクリプト
値の再抽出も含む
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from db_connector import DatabaseConnector
from analyze_patch_connections import PatchConnectionAnalyzer
from datetime import datetime
import time

class ConnectionMigrator:
    """接続データの移行クラス"""
    
    def __init__(self, config_path: str):
        """
        初期化
        
        Args:
            config_path: データベース設定ファイルのパス
        """
        self.db = DatabaseConnector(config_path)
        self.analyzer = PatchConnectionAnalyzer(config_path)
        self.processed_count = 0
        self.batch_size = 1000  # 一度に処理する接続数
        
    def get_unique_patches(self) -> List[str]:
        """
        connectionsテーブルから一意のパッチファイルリストを取得
        
        Returns:
            パッチファイルパスのリスト
        """
        try:
            self.db.connect()
            query = """
                SELECT DISTINCT patch_id 
                FROM connections 
                ORDER BY patch_id
            """
            results = self.db.execute_query(query)
            return [r['patch_id'] for r in results]
        finally:
            self.db.disconnect()
    
    def migrate_patch_connections(self, patch_file: str) -> int:
        """
        特定のパッチファイルの接続を移行
        
        Args:
            patch_file: パッチファイルパス
            
        Returns:
            移行した接続数
        """
        # ファイルが存在するか確認
        if not Path(patch_file).exists():
            print(f"  ファイルが存在しません: {patch_file}")
            return 0
        
        # パッチを再分析して値を取得
        print(f"  分析中: {patch_file}")
        analysis_result = self.analyzer.analyze_patch_file(patch_file)
        
        if not analysis_result or not analysis_result.get('connections'):
            print(f"  分析失敗またはNo connections")
            return 0
        
        # object_connectionsテーブルに保存
        success = self.analyzer.store_connections_to_new_db(analysis_result)
        
        if success:
            return len(analysis_result['connections'])
        else:
            return 0
    
    def migrate_all(self, limit: Optional[int] = None):
        """
        すべての接続データを移行
        
        Args:
            limit: 処理するパッチファイル数の上限（テスト用）
        """
        print("=== 接続データ移行開始 ===")
        start_time = time.time()
        
        # 一意のパッチファイルリストを取得
        patch_files = self.get_unique_patches()
        total_patches = len(patch_files)
        
        if limit:
            patch_files = patch_files[:limit]
            print(f"テストモード: 最初の{limit}パッチのみ処理")
        
        print(f"移行対象パッチ数: {len(patch_files)} / {total_patches}")
        
        # オブジェクトIDキャッシュを構築
        self.analyzer.load_object_id_cache()
        
        # バッチ処理
        processed_patches = 0
        total_connections = 0
        failed_patches = []
        
        try:
            self.db.connect()
            
            # 既存のデータをクリア（オプション）
            if input("既存のobject_connectionsデータを削除しますか？ (y/N): ").lower() == 'y':
                self.db.execute_query("TRUNCATE TABLE object_connections")
                print("既存データを削除しました")
            
            # トランザクション開始
            self.db.execute_query("BEGIN")
            
            for i, patch_file in enumerate(patch_files):
                if i % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (len(patch_files) - i) / rate if rate > 0 else 0
                    print(f"\n進捗: {i}/{len(patch_files)} ({i/len(patch_files)*100:.1f}%) - ETA: {eta/60:.1f}分")
                
                # 定期的にコミット
                if i > 0 and i % 50 == 0:
                    self.db.execute_query("COMMIT")
                    self.db.execute_query("BEGIN")
                    print(f"  {i}パッチ処理済み、コミット実行")
                
                try:
                    connections_migrated = self.migrate_patch_connections(patch_file)
                    if connections_migrated > 0:
                        processed_patches += 1
                        total_connections += connections_migrated
                    else:
                        failed_patches.append(patch_file)
                except Exception as e:
                    print(f"  エラー: {e}")
                    failed_patches.append(patch_file)
            
            # 最終コミット
            self.db.execute_query("COMMIT")
            
        except Exception as e:
            print(f"\n移行エラー: {e}")
            self.db.execute_query("ROLLBACK")
        finally:
            self.db.disconnect()
        
        # 結果表示
        elapsed_time = time.time() - start_time
        print(f"\n=== 移行完了 ===")
        print(f"処理時間: {elapsed_time/60:.1f}分")
        print(f"処理パッチ数: {processed_patches}/{len(patch_files)}")
        print(f"移行接続数: {total_connections}")
        print(f"失敗パッチ数: {len(failed_patches)}")
        
        if failed_patches and len(failed_patches) <= 10:
            print("\n失敗パッチ:")
            for fp in failed_patches[:10]:
                print(f"  - {fp}")
    
    def verify_migration(self):
        """移行結果を検証"""
        try:
            self.db.connect()
            
            # 統計情報を取得
            old_count = self.db.execute_query("SELECT COUNT(*) as count FROM connections")[0]['count']
            new_count = self.db.execute_query("SELECT COUNT(*) as count FROM object_connections")[0]['count']
            
            # 値付きデータの統計
            value_stats = self.db.execute_query("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(source_value) as with_source_value,
                    COUNT(target_value) as with_target_value
                FROM object_connections
            """)[0]
            
            print(f"\n=== 移行検証結果 ===")
            print(f"旧テーブル（connections）: {old_count}件")
            print(f"新テーブル（object_connections）: {new_count}件")
            print(f"値付きソース: {value_stats['with_source_value']} ({value_stats['with_source_value']/new_count*100:.1f}%)")
            print(f"値付きターゲット: {value_stats['with_target_value']} ({value_stats['with_target_value']/new_count*100:.1f}%)")
            
        finally:
            self.db.disconnect()

def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description='接続データの移行')
    parser.add_argument('--config', default='scripts/db_settings.ini', help='設定ファイルパス')
    parser.add_argument('--limit', type=int, help='処理するパッチ数の上限（テスト用）')
    parser.add_argument('--verify', action='store_true', help='移行結果の検証のみ実行')
    
    args = parser.parse_args()
    
    migrator = ConnectionMigrator(args.config)
    
    if args.verify:
        migrator.verify_migration()
    else:
        migrator.migrate_all(limit=args.limit)

if __name__ == '__main__':
    main()