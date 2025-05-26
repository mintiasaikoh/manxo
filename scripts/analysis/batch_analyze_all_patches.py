#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Max_Projects内の全maxpat/amxdファイルをバッチ分析
6,424個のファイルを効率的に処理してデータベースに格納
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import multiprocessing as mp
from analyze_patch_connections import PatchConnectionAnalyzer

class BatchPatchAnalyzer:
    """大量のパッチファイルをバッチ処理するクラス"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.results_file = "batch_analysis_results.json"
        
    def find_all_patch_files(self, directory: str) -> Dict[str, List[str]]:
        """すべてのパッチファイルを発見"""
        
        maxpat_files = list(Path(directory).rglob("*.maxpat"))
        amxd_files = list(Path(directory).rglob("*.amxd"))
        
        return {
            'maxpat': [str(f) for f in maxpat_files],
            'amxd': [str(f) for f in amxd_files]
        }
    
    def analyze_file_batch(self, file_batch: List[str], batch_id: int) -> Dict[str, Any]:
        """ファイルバッチを分析"""
        
        print(f"バッチ {batch_id} 開始: {len(file_batch)}個のファイル")
        
        analyzer = PatchConnectionAnalyzer(self.config_path)
        analyzer.load_object_id_cache()  # キャッシュを1回だけロード
        
        batch_results = {
            'batch_id': batch_id,
            'files_processed': 0,
            'connections_stored': 0,
            'failed_files': [],
            'success_files': []
        }
        
        for file_path in file_batch:
            try:
                # ファイル分析
                analysis = analyzer.analyze_patch_file(file_path)
                
                if analysis:
                    # データベースに格納
                    stored_count = analyzer.store_connections_to_db(analysis)
                    
                    batch_results['files_processed'] += 1
                    batch_results['connections_stored'] += stored_count
                    batch_results['success_files'].append({
                        'file': file_path,
                        'objects': analysis['total_objects'],
                        'connections': analysis['total_connections'],
                        'stored': stored_count
                    })
                    
                    if batch_results['files_processed'] % 10 == 0:
                        print(f"  バッチ {batch_id}: {batch_results['files_processed']}個処理済み")
                        
                else:
                    batch_results['failed_files'].append(file_path)
                    
            except Exception as e:
                print(f"  エラー {file_path}: {e}")
                batch_results['failed_files'].append(file_path)
        
        print(f"バッチ {batch_id} 完了: {batch_results['files_processed']}個処理, {batch_results['connections_stored']}個接続格納")
        return batch_results
    
    def analyze_all_files(self, directory: str, batch_size: int = 50, max_files: int = None) -> Dict[str, Any]:
        """全ファイルを分析"""
        
        print(f"=== 全パッチファイル分析開始 ===")
        print(f"対象ディレクトリ: {directory}")
        
        # ファイルリスト取得
        all_files = self.find_all_patch_files(directory)
        maxpat_files = all_files['maxpat']
        amxd_files = all_files['amxd']
        
        print(f"発見したファイル:")
        print(f"  maxpatファイル: {len(maxpat_files)}個")
        print(f"  amxdファイル: {len(amxd_files)}個")
        print(f"  合計: {len(maxpat_files) + len(amxd_files)}個")
        
        # 処理対象を制限（テスト用）
        if max_files:
            all_patch_files = (maxpat_files + amxd_files)[:max_files]
            print(f"制限により {max_files}個のファイルのみ処理")
        else:
            all_patch_files = maxpat_files + amxd_files
        
        # バッチに分割
        batches = [all_patch_files[i:i + batch_size] for i in range(0, len(all_patch_files), batch_size)]
        print(f"バッチ数: {len(batches)} (バッチサイズ: {batch_size})")
        
        # 結果格納
        total_results = {
            'start_time': time.time(),
            'total_files': len(all_patch_files),
            'total_batches': len(batches),
            'batch_results': [],
            'summary': {
                'files_processed': 0,
                'connections_stored': 0,
                'failed_files': 0,
                'success_rate': 0.0
            }
        }
        
        # バッチ処理実行（シーケンシャル処理で安全性確保）
        for i, batch in enumerate(batches):
            try:
                batch_result = self.analyze_file_batch(batch, i + 1)
                total_results['batch_results'].append(batch_result)
                
                # 累計更新
                total_results['summary']['files_processed'] += batch_result['files_processed']
                total_results['summary']['connections_stored'] += batch_result['connections_stored']
                total_results['summary']['failed_files'] += len(batch_result['failed_files'])
                
                # 進捗表示
                progress = (i + 1) / len(batches) * 100
                print(f"\n進捗: {progress:.1f}% ({i + 1}/{len(batches)}バッチ完了)")
                print(f"累計: {total_results['summary']['files_processed']}個処理, {total_results['summary']['connections_stored']}個接続格納")
                
                # 中間結果保存
                if (i + 1) % 5 == 0:  # 5バッチごとに保存
                    self.save_results(total_results)
                    
            except Exception as e:
                print(f"バッチ {i + 1} でエラー: {e}")
                continue
        
        # 最終統計
        total_results['end_time'] = time.time()
        total_results['duration'] = total_results['end_time'] - total_results['start_time']
        
        if total_results['summary']['files_processed'] > 0:
            total_results['summary']['success_rate'] = (
                total_results['summary']['files_processed'] / 
                (total_results['summary']['files_processed'] + total_results['summary']['failed_files'])
            ) * 100
        
        # 最終結果保存
        self.save_results(total_results)
        
        return total_results
    
    def save_results(self, results: Dict[str, Any]):
        """結果を保存"""
        try:
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"結果を保存: {self.results_file}")
        except Exception as e:
            print(f"結果保存エラー: {e}")
    
    def print_final_summary(self, results: Dict[str, Any]):
        """最終サマリー表示"""
        summary = results['summary']
        
        print(f"\n=== 全体分析完了サマリー ===")
        print(f"処理時間: {results.get('duration', 0):.1f}秒")
        print(f"処理ファイル数: {summary['files_processed']}")
        print(f"格納接続数: {summary['connections_stored']}")
        print(f"失敗ファイル数: {summary['failed_files']}")
        print(f"成功率: {summary['success_rate']:.1f}%")
        
        if summary['files_processed'] > 0:
            avg_connections = summary['connections_stored'] / summary['files_processed']
            print(f"ファイル当たり平均接続数: {avg_connections:.1f}")

def main():
    """メイン実行"""
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python batch_analyze_all_patches.py <directory> [max_files]")
        print("例: python batch_analyze_all_patches.py Max_Projects 100")
        sys.exit(1)
    
    directory = sys.argv[1]
    max_files = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    analyzer = BatchPatchAnalyzer('scripts/db_settings.ini')
    
    print(f"大量パッチファイル分析開始")
    if max_files:
        print(f"テストモード: 最大 {max_files} ファイル")
    
    results = analyzer.analyze_all_files(directory, batch_size=50, max_files=max_files)
    analyzer.print_final_summary(results)

if __name__ == "__main__":
    main()