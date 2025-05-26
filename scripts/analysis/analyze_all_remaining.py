#!/usr/bin/env python3
"""全ての未処理ファイルを一括解析（サイズ制限付き）"""

import sys
sys.path.append('/Users/mymac/manxo')
from scripts.db_connector import DatabaseConnector
from pathlib import Path
import subprocess
import time
import random

def main():
    db = DatabaseConnector('scripts/db_settings.ini')
    db.connect()
    
    # 既に分析済みのファイル名を取得
    result = db.execute_query("""
    SELECT DISTINCT SUBSTRING(patch_file FROM '[^/]+$'] as filename
    FROM object_connections
    """)
    analyzed_files = {r['filename'] for r in result}
    print(f"既に分析済み: {len(analyzed_files)} ファイル")
    
    # 全ての未処理ファイルを取得
    all_maxpat = list(Path('/Users/mymac/manxo/Max_Projects/all_maxpat').glob('*.maxpat'))
    all_amxd = list(Path('/Users/mymac/manxo/Max_Projects/all_amxd').glob('*.amxd'))
    
    unanalyzed_files = []
    for f in all_maxpat:
        if f.name not in analyzed_files:
            unanalyzed_files.append(('maxpat', f))
    for f in all_amxd:
        if f.name not in analyzed_files:
            unanalyzed_files.append(('amxd', f))
    
    print(f"未処理ファイル: {len(unanalyzed_files)} 個")
    
    # サイズでフィルタリング（5MB以下のみ）
    processable_files = []
    for file_type, filepath in unanalyzed_files:
        size_mb = filepath.stat().st_size / (1024 * 1024)
        if size_mb <= 5.0:  # 5MB以下
            processable_files.append((file_type, filepath, size_mb))
    
    print(f"処理可能ファイル（5MB以下）: {len(processable_files)} 個")
    
    # ランダムに100個選択（全部処理すると時間がかかるため）
    if len(processable_files) > 100:
        random.shuffle(processable_files)
        processable_files = processable_files[:100]
        print(f"ランダムに100個を選択")
    
    print(f"\n=== {len(processable_files)} ファイルを解析開始 ===\n")
    
    success_count = 0
    error_count = 0
    total_connections = 0
    
    for i, (file_type, filepath, size_mb) in enumerate(processable_files, 1):
        print(f"[{i}/{len(processable_files)}] {filepath.name} ({size_mb:.2f} MB)")
        
        # analyze_patch_connections.pyを実行
        try:
            cmd = [
                sys.executable,
                'scripts/analyze_patch_connections.py',
                str(filepath)
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=20,  # 20秒タイムアウト
                cwd='/Users/mymac/manxo'
            )
            
            if result.returncode == 0:
                # 接続数を抽出
                for line in result.stdout.split('\n'):
                    if '総接続数:' in line:
                        try:
                            conn_count = int(line.split(':')[1].strip())
                            total_connections += conn_count
                            print(f"   ✅ {conn_count} 接続")
                            success_count += 1
                            break
                        except:
                            print(f"   ✅ 完了")
                            success_count += 1
                            break
            else:
                print(f"   ❌ エラー")
                error_count += 1
                
        except subprocess.TimeoutExpired:
            print(f"   ❌ タイムアウト")
            error_count += 1
        except Exception as e:
            print(f"   ❌ 例外: {type(e).__name__}")
            error_count += 1
        
        # 進捗表示
        if i % 10 == 0:
            print(f"\n--- 進捗: {i}/{len(processable_files)} ({i/len(processable_files)*100:.1f}%) ---")
            print(f"成功: {success_count}, エラー: {error_count}, 新規接続: {total_connections:,}\n")
        
        # サーバー負荷軽減のため少し待機
        time.sleep(0.2)
    
    # 最終結果
    print(f"\n\n=== 最終結果 ===")
    print(f"処理ファイル数: {len(processable_files)}")
    print(f"成功: {success_count} ({success_count/len(processable_files)*100:.1f}%)")
    print(f"エラー: {error_count} ({error_count/len(processable_files)*100:.1f}%)")
    print(f"新規接続総数: {total_connections:,}")
    
    # 更新後の統計
    result = db.execute_query("""
    SELECT 
        file_type,
        COUNT(DISTINCT patch_file) as file_count,
        COUNT(*) as connection_count
    FROM object_connections
    GROUP BY file_type
    """)
    
    print("\n最終データベース統計:")
    total_files = 0
    total_conns = 0
    for r in result:
        print(f"  {r['file_type']}: {r['file_count']:,} ファイル, {r['connection_count']:,} 接続")
        total_files += r['file_count']
        total_conns += r['connection_count']
    print(f"  合計: {total_files:,} ファイル, {total_conns:,} 接続")
    
    # カバレッジ計算
    actual_maxpat = len([f for f in Path('/Users/mymac/manxo/Max_Projects/all_maxpat').glob('*.maxpat')])
    actual_amxd = len([f for f in Path('/Users/mymac/manxo/Max_Projects/all_amxd').glob('*.amxd')])
    
    print(f"\nカバレッジ:")
    print(f"  maxpat: {result[1]['file_count']/actual_maxpat*100:.1f}% ({result[1]['file_count']}/{actual_maxpat})")
    print(f"  amxd: {result[0]['file_count']/actual_amxd*100:.1f}% ({result[0]['file_count']}/{actual_amxd})")
    
    db.disconnect()

if __name__ == "__main__":
    main()