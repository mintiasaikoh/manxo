#!/usr/bin/env python3
"""
分析結果をPostgreSQLデータベースに取り込むスクリプト
Max/MSPパッチ分析結果を構造化されたデータベースに保存する
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

# プロジェクトルートのパスを追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.db_connector import DatabaseConnector

class AnalysisImporter:
    """分析結果をデータベースに取り込むクラス"""
    
    def __init__(self, config_path: str):
        """
        初期化
        
        Args:
            config_path: データベース設定ファイルのパス
        """
        self.db = DatabaseConnector(config_path)
        
    def import_objects_from_analysis(self, analysis_file: str) -> int:
        """
        分析結果からオブジェクト情報を取り込む
        
        Args:
            analysis_file: 分析結果JSONファイルのパス
            
        Returns:
            取り込んだオブジェクト数
        """
        if not os.path.exists(analysis_file):
            print(f"分析ファイルが見つかりません: {analysis_file}")
            return 0
            
        with open(analysis_file, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        objects_imported = 0
        
        # 分析データからオブジェクト情報を抽出
        if 'objects' in analysis_data:
            for obj_data in analysis_data['objects']:
                try:
                    object_id = obj_data.get('id', f"obj_{objects_imported}")
                    object_name = obj_data.get('name', 'unknown')
                    object_type = obj_data.get('type', 'unknown')
                    box_type = obj_data.get('box_type', 'unknown')
                    
                    # ポート情報の処理
                    inlets = obj_data.get('inlets', 0)
                    outlets = obj_data.get('outlets', 0)
                    
                    if isinstance(inlets, list):
                        inlets = len(inlets)
                    if isinstance(outlets, list):
                        outlets = len(outlets)
                    
                    # 座標情報
                    patching_rect = obj_data.get('patching_rect', [0, 0, 0, 0])
                    x = patching_rect[0] if len(patching_rect) > 0 else 0
                    y = patching_rect[1] if len(patching_rect) > 1 else 0
                    
                    self.db.insert_object(
                        object_id=object_id,
                        object_name=object_name,
                        object_type=object_type,
                        box_type=box_type,
                        inlets=inlets,
                        outlets=outlets,
                        x_position=x,
                        y_position=y
                    )
                    objects_imported += 1
                    
                except Exception as e:
                    print(f"オブジェクト取り込みエラー: {e}")
                    continue
        
        return objects_imported
    
    def import_connections_from_analysis(self, analysis_file: str) -> int:
        """
        分析結果から接続情報を取り込む
        
        Args:
            analysis_file: 分析結果JSONファイルのパス
            
        Returns:
            取り込んだ接続数
        """
        if not os.path.exists(analysis_file):
            print(f"分析ファイルが見つかりません: {analysis_file}")
            return 0
            
        with open(analysis_file, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        connections_imported = 0
        
        # 接続情報の処理
        if 'connections' in analysis_data:
            for conn_data in analysis_data['connections']:
                try:
                    source_id = conn_data.get('source_id')
                    source_port = conn_data.get('source_port', 0)
                    target_id = conn_data.get('target_id')
                    target_port = conn_data.get('target_port', 0)
                    
                    if source_id and target_id:
                        self.db.insert_connection(
                            source_id=str(source_id),
                            source_port=source_port,
                            target_id=str(target_id),
                            target_port=target_port
                        )
                        connections_imported += 1
                        
                except Exception as e:
                    print(f"接続取り込みエラー: {e}")
                    continue
        
        return connections_imported
    
    def import_patterns_from_file(self, patterns_file: str) -> int:
        """
        パターンファイルから接続パターンを取り込む
        
        Args:
            patterns_file: パターンファイルのパス
            
        Returns:
            取り込んだパターン数
        """
        if not os.path.exists(patterns_file):
            print(f"パターンファイルが見つかりません: {patterns_file}")
            return 0
            
        with open(patterns_file, 'r', encoding='utf-8') as f:
            patterns_data = json.load(f)
        
        patterns_imported = 0
        
        if 'patterns' in patterns_data:
            patterns = patterns_data['patterns']
        else:
            patterns = patterns_data
        
        for pattern_name, pattern_data in patterns.items():
            try:
                frequency = pattern_data.get('frequency', 1)
                description = pattern_data.get('description', '')
                objects_involved = pattern_data.get('objects', [])
                
                self.db.insert_pattern(
                    pattern_name=pattern_name,
                    frequency=frequency,
                    description=description,
                    objects_involved=json.dumps(objects_involved)
                )
                patterns_imported += 1
                
            except Exception as e:
                print(f"パターン取り込みエラー: {e}")
                continue
        
        return patterns_imported
    
    def import_directory(self, analysis_dir: str) -> Dict[str, int]:
        """
        ディレクトリ内の全分析結果を取り込む
        
        Args:
            analysis_dir: 分析結果ディレクトリのパス
            
        Returns:
            取り込み結果の統計
        """
        results = {
            'objects': 0,
            'connections': 0,
            'patterns': 0,
            'files_processed': 0
        }
        
        if not os.path.exists(analysis_dir):
            print(f"分析ディレクトリが見つかりません: {analysis_dir}")
            return results
        
        # 分析結果JSONファイルを検索
        for root, dirs, files in os.walk(analysis_dir):
            for file in files:
                if file.endswith('.json') and ('analysis' in file or 'result' in file):
                    file_path = os.path.join(root, file)
                    print(f"処理中: {file_path}")
                    
                    try:
                        objects_count = self.import_objects_from_analysis(file_path)
                        connections_count = self.import_connections_from_analysis(file_path)
                        
                        results['objects'] += objects_count
                        results['connections'] += connections_count
                        results['files_processed'] += 1
                        
                        print(f"  - オブジェクト: {objects_count}")
                        print(f"  - 接続: {connections_count}")
                        
                    except Exception as e:
                        print(f"ファイル処理エラー {file_path}: {e}")
                        continue
        
        return results

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='分析結果をPostgreSQLデータベースに取り込み')
    parser.add_argument('--config', required=True, help='データベース設定ファイルのパス')
    parser.add_argument('--analysis-dir', help='分析結果ディレクトリのパス')
    parser.add_argument('--analysis-file', help='単一の分析結果ファイルのパス')
    parser.add_argument('--patterns-file', help='パターンファイルのパス')
    parser.add_argument('--init', action='store_true', help='データベースを初期化')
    
    args = parser.parse_args()
    
    try:
        importer = AnalysisImporter(args.config)
        
        if args.init:
            print("データベースを初期化中...")
            importer.db.initialize_database()
            print("データベース初期化完了")
        
        total_stats = {
            'objects': 0,
            'connections': 0,
            'patterns': 0,
            'files_processed': 0
        }
        
        # ディレクトリ全体の処理
        if args.analysis_dir:
            print(f"分析ディレクトリを処理中: {args.analysis_dir}")
            dir_stats = importer.import_directory(args.analysis_dir)
            for key, value in dir_stats.items():
                total_stats[key] += value
        
        # 単一ファイルの処理
        if args.analysis_file:
            print(f"分析ファイルを処理中: {args.analysis_file}")
            objects_count = importer.import_objects_from_analysis(args.analysis_file)
            connections_count = importer.import_connections_from_analysis(args.analysis_file)
            total_stats['objects'] += objects_count
            total_stats['connections'] += connections_count
            total_stats['files_processed'] += 1
        
        # パターンファイルの処理
        if args.patterns_file:
            print(f"パターンファイルを処理中: {args.patterns_file}")
            patterns_count = importer.import_patterns_from_file(args.patterns_file)
            total_stats['patterns'] += patterns_count
        
        # 結果表示
        print("\n=== 取り込み結果 ===")
        print(f"処理ファイル数: {total_stats['files_processed']}")
        print(f"オブジェクト数: {total_stats['objects']}")
        print(f"接続数: {total_stats['connections']}")
        print(f"パターン数: {total_stats['patterns']}")
        
        # データベース統計を表示
        db_stats = importer.db.get_statistics()
        print("\n=== データベース統計 ===")
        for table, count in db_stats.items():
            print(f"{table}: {count} レコード")
            
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()