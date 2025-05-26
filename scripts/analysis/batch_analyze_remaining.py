#!/usr/bin/env python3
"""残りのファイルをバッチ解析（エラーをスキップ）"""

import sys
sys.path.append('/Users/mymac/manxo')
from scripts.db_connector import DatabaseConnector
from pathlib import Path
import subprocess
import json
import time

def main():
    db = DatabaseConnector('scripts/db_settings.ini')
    db.connect()
    
    # 既に分析済みのファイル名を取得
    result = db.execute_query("""
    SELECT DISTINCT SUBSTRING(patch_file FROM '[^/]+$') as filename
    FROM object_connections
    """)
    analyzed_files = {r['filename'] for r in result}
    print(f"既に分析済み: {len(analyzed_files)} ファイル")
    
    # 高重要度ファイルと追加の興味深いファイル
    target_files = [
        # Gen関連
        ('maxpat', 'mi-gen_smc2019_slide2.maxpat'),
        ('maxpat', 'sdinulescu_AnaloGen_WorkingExamples-All.maxpat'),
        ('maxpat', 'wwerkk_MC-FP_synth-gen.maxpat'),
        ('maxpat', 'cskonopka_arduivis_codegen-output.maxpat'),
        # MC関連
        ('maxpat', 'NoOne511_UE5-MaxMSP-OSC-comm_3dgran-fromtut_08MC.maxpat'),
        # Jitter関連
        ('maxpat', 'jit.grab@.maxpat'),
        ('maxpat', 'jit.videoplanesP.maxpat'),
        ('maxpat', 'jit.videoplanes.maxpat'),
        ('maxpat', 'jit.display@.maxpat'),
        # RNBO関連
        ('maxpat', 'jinpavg_rnbo-unity-webgl_game-audio.maxpat'),
        # 高度な信号処理
        ('maxpat', 'kevin-roark_johnny_granular_zone.maxpat'),
        ('maxpat', 'bltzr_Z_jalg.Zconvolution~.maxpat'),
        # Spektral/Convolution
        ('maxpat', 'jackhwalters_The-Fast-Fourier-Transform-and-Spectral-Manipulation-in-M.maxpat'),
        # M4L関連（小さめファイル）
        ('amxd', 'lfo-midi-live-10.amxd'),
        ('amxd', 'Scaler 3 Wrapper.amxd'),
        ('amxd', 'Note Echo XL.amxd'),
        # その他興味深いファイル
        ('maxpat', 'ppooll_host.maxpat'),
        ('maxpat', 'serialosc.maxpat_serialosc-old.maxpat'),
    ]
    
    print(f"\n=== {len(target_files)} ファイルを解析予定 ===\n")
    
    success_count = 0
    error_count = 0
    skip_count = 0
    
    for file_type, filename in target_files:
        if filename in analyzed_files:
            print(f"✅ {filename} - 既に分析済み")
            skip_count += 1
            continue
        
        filepath = Path(f'/Users/mymac/manxo/Max_Projects/all_{file_type}') / filename
        
        if not filepath.exists():
            print(f"❌ {filename} - ファイルが見つかりません")
            error_count += 1
            continue
        
        # ファイルサイズ確認
        size_kb = filepath.stat().st_size / 1024
        print(f"\n📋 {filename} ({size_kb:.1f} KB)")
        
        if size_kb > 10000:  # 10MB以上はスキップ
            print(f"   ⚠️  大きすぎるファイル（{size_kb/1024:.1f} MB）をスキップ")
            skip_count += 1
            continue
        
        # analyze_patch_connections.pyを子プロセスとして実行
        try:
            cmd = [
                sys.executable,
                'scripts/analyze_patch_connections.py',
                str(filepath)
            ]
            
            # タイムアウト付きで実行（30秒）
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30,
                cwd='/Users/mymac/manxo'
            )
            
            if result.returncode == 0:
                # 出力から結果を解析
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines:
                    if '総接続数:' in line:
                        conn_count = line.split(':')[1].strip()
                        print(f"   ✅ 成功: {conn_count} 接続")
                        success_count += 1
                        break
                else:
                    print(f"   ⚠️  解析完了（詳細不明）")
                    success_count += 1
            else:
                error_msg = result.stderr.strip().split('\n')[-1] if result.stderr else "不明なエラー"
                print(f"   ❌ エラー: {error_msg}")
                error_count += 1
                
        except subprocess.TimeoutExpired:
            print(f"   ❌ タイムアウト（30秒）")
            error_count += 1
        except Exception as e:
            print(f"   ❌ 実行エラー: {str(e)}")
            error_count += 1
        
        # 少し待機
        time.sleep(0.5)
    
    # 結果サマリー
    print(f"\n\n=== 解析結果サマリー ===")
    print(f"成功: {success_count} ファイル")
    print(f"エラー: {error_count} ファイル") 
    print(f"スキップ: {skip_count} ファイル")
    print(f"合計: {len(target_files)} ファイル")
    
    # 更新後の統計
    result = db.execute_query("""
    SELECT 
        file_type,
        COUNT(DISTINCT patch_file) as file_count,
        COUNT(*) as connection_count
    FROM object_connections
    GROUP BY file_type
    """)
    
    print("\n更新後のデータベース統計:")
    for r in result:
        print(f"  {r['file_type']}: {r['file_count']:,} ファイル, {r['connection_count']:,} 接続")
    
    db.disconnect()

if __name__ == "__main__":
    main()