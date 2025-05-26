#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PostgreSQL データベース接続とテーブル管理
Max/MSP パッチ分析結果をPostgreSQLデータベースに保存・管理するためのスクリプト
"""

import os
import sys
import argparse
import configparser
import logging
from typing import Dict, List, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import json

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseConnector:
    """PostgreSQLデータベース接続管理クラス"""
    
    def __init__(self, config_file: str):
        """
        初期化
        
        Args:
            config_file: データベース設定ファイルのパス
        """
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.connection = None
        
    def connect(self) -> bool:
        """
        データベースに接続
        
        Returns:
            接続成功時True
        """
        try:
            db_config = self.config['database']
            self.connection = psycopg2.connect(
                host=db_config['host'],
                port=int(db_config['port']),
                database=db_config['database'],
                user=db_config['user'],
                password=db_config['password']
            )
            logger.info("データベースに接続しました")
            return True
        except Exception as e:
            logger.error(f"データベース接続エラー: {e}")
            return False
    
    def disconnect(self):
        """データベース接続を切断"""
        if self.connection:
            self.connection.close()
            logger.info("データベース接続を切断しました")
    
    def init_database(self) -> bool:
        """
        データベースとテーブルを初期化
        
        Returns:
            初期化成功時True
        """
        if not self.connection:
            logger.error("データベースに接続していません")
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # テーブル作成のSQL
            create_tables_sql = """
            -- オブジェクト情報テーブル
            CREATE TABLE IF NOT EXISTS objects (
                id SERIAL PRIMARY KEY,
                object_name VARCHAR(255) NOT NULL,
                object_type VARCHAR(100),
                category VARCHAR(100),
                num_inlets INTEGER DEFAULT 0,
                num_outlets INTEGER DEFAULT 0,
                inlet_types JSONB,
                outlet_types JSONB,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- 接続情報テーブル
            CREATE TABLE IF NOT EXISTS connections (
                id SERIAL PRIMARY KEY,
                patch_id VARCHAR(255),
                source_object_id VARCHAR(255),
                source_port INTEGER,
                target_object_id VARCHAR(255),
                target_port INTEGER,
                connection_type VARCHAR(50) DEFAULT 'message',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- 接続パターンテーブル
            CREATE TABLE IF NOT EXISTS connection_patterns (
                id SERIAL PRIMARY KEY,
                pattern_name VARCHAR(255) UNIQUE,
                source_type VARCHAR(100),
                target_type VARCHAR(100),
                frequency INTEGER DEFAULT 1,
                significance_score DECIMAL(5,3),
                pattern_data JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- ポート情報テーブル
            CREATE TABLE IF NOT EXISTS port_info (
                id SERIAL PRIMARY KEY,
                object_name VARCHAR(255) NOT NULL,
                port_number INTEGER NOT NULL,
                port_type VARCHAR(20) NOT NULL, -- 'inlet' or 'outlet'
                data_type VARCHAR(50),
                description TEXT,
                is_dynamic BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(object_name, port_number, port_type)
            );
            
            -- パッチ分析情報テーブル
            CREATE TABLE IF NOT EXISTS patch_analysis (
                id SERIAL PRIMARY KEY,
                patch_name VARCHAR(255),
                patch_path TEXT,
                total_objects INTEGER,
                total_connections INTEGER,
                analysis_data JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- インデックス作成
            CREATE INDEX IF NOT EXISTS idx_objects_name ON objects(object_name);
            CREATE INDEX IF NOT EXISTS idx_objects_type ON objects(object_type);
            CREATE INDEX IF NOT EXISTS idx_connections_patch ON connections(patch_id);
            CREATE INDEX IF NOT EXISTS idx_connections_source ON connections(source_object_id);
            CREATE INDEX IF NOT EXISTS idx_connections_target ON connections(target_object_id);
            CREATE INDEX IF NOT EXISTS idx_patterns_types ON connection_patterns(source_type, target_type);
            CREATE INDEX IF NOT EXISTS idx_port_info_object ON port_info(object_name);
            """
            
            cursor.execute(create_tables_sql)
            self.connection.commit()
            
            logger.info("データベーステーブルを初期化しました")
            return True
            
        except Exception as e:
            logger.error(f"データベース初期化エラー: {e}")
            self.connection.rollback()
            return False
        finally:
            cursor.close()
    
    def insert_object(self, object_data: Dict[str, Any]) -> int:
        """
        オブジェクト情報を挿入
        
        Args:
            object_data: オブジェクト情報の辞書
            
        Returns:
            挿入されたレコードのID
        """
        if not self.connection:
            return -1
            
        try:
            cursor = self.connection.cursor()
            
            insert_sql = """
            INSERT INTO objects (
                object_name, object_type, category, num_inlets, num_outlets,
                inlet_types, outlet_types, description
            ) VALUES (
                %(object_name)s, %(object_type)s, %(category)s, %(num_inlets)s, 
                %(num_outlets)s, %(inlet_types)s, %(outlet_types)s, %(description)s
            ) ON CONFLICT (object_name) DO UPDATE SET
                object_type = EXCLUDED.object_type,
                category = EXCLUDED.category,
                num_inlets = EXCLUDED.num_inlets,
                num_outlets = EXCLUDED.num_outlets,
                inlet_types = EXCLUDED.inlet_types,
                outlet_types = EXCLUDED.outlet_types,
                description = EXCLUDED.description,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id;
            """
            
            cursor.execute(insert_sql, object_data)
            object_id = cursor.fetchone()[0]
            self.connection.commit()
            
            return object_id
            
        except Exception as e:
            logger.error(f"オブジェクト挿入エラー: {e}")
            self.connection.rollback()
            return -1
        finally:
            cursor.close()
    
    def insert_connection(self, connection_data: Dict[str, Any]) -> int:
        """
        接続情報を挿入
        
        Args:
            connection_data: 接続情報の辞書
            
        Returns:
            挿入されたレコードのID
        """
        if not self.connection:
            return -1
            
        try:
            cursor = self.connection.cursor()
            
            insert_sql = """
            INSERT INTO connections (
                patch_id, source_object_id, source_port, target_object_id, 
                target_port, connection_type
            ) VALUES (
                %(patch_id)s, %(source_object_id)s, %(source_port)s, 
                %(target_object_id)s, %(target_port)s, %(connection_type)s
            ) RETURNING id;
            """
            
            cursor.execute(insert_sql, connection_data)
            connection_id = cursor.fetchone()[0]
            self.connection.commit()
            
            return connection_id
            
        except Exception as e:
            logger.error(f"接続挿入エラー: {e}")
            self.connection.rollback()
            return -1
        finally:
            cursor.close()
    
    def get_connection_patterns(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        接続パターンを取得
        
        Args:
            limit: 取得する最大レコード数
            
        Returns:
            接続パターンのリスト
        """
        if not self.connection:
            return []
            
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            query_sql = """
            SELECT * FROM connection_patterns 
            ORDER BY frequency DESC, significance_score DESC
            LIMIT %s;
            """
            
            cursor.execute(query_sql, (limit,))
            patterns = cursor.fetchall()
            
            return [dict(pattern) for pattern in patterns]
            
        except Exception as e:
            logger.error(f"パターン取得エラー: {e}")
            return []
        finally:
            cursor.close()
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        任意のクエリを実行
        
        Args:
            query: 実行するSQLクエリ
            params: クエリパラメータ
            
        Returns:
            クエリ結果のリスト
        """
        if not self.connection:
            return []
            
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, params)
            
            if query.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
                return [dict(row) for row in results]
            else:
                self.connection.commit()
                return []
                
        except Exception as e:
            logger.error(f"クエリ実行エラー: {e}")
            self.connection.rollback()
            return []
        finally:
            cursor.close()


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="Max/MSP PostgreSQL データベース管理")
    parser.add_argument("--config", "-c", required=True, help="データベース設定ファイル")
    parser.add_argument("--init", action="store_true", help="データベースを初期化")
    parser.add_argument("--test", action="store_true", help="データベース接続をテスト")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        logger.error(f"設定ファイルが見つかりません: {args.config}")
        return 1
    
    # データベース接続
    db = DatabaseConnector(args.config)
    
    if not db.connect():
        logger.error("データベース接続に失敗しました")
        return 1
    
    try:
        if args.init:
            logger.info("データベースを初期化中...")
            if db.init_database():
                logger.info("データベースの初期化が完了しました")
            else:
                logger.error("データベースの初期化に失敗しました")
                return 1
        
        if args.test:
            logger.info("データベース接続をテスト中...")
            result = db.execute_query("SELECT version();")
            if result:
                logger.info(f"PostgreSQL バージョン: {result[0]['version']}")
                logger.info("データベース接続テスト成功")
            else:
                logger.error("データベース接続テスト失敗")
                return 1
    
    finally:
        db.disconnect()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())