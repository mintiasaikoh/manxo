#!/usr/bin/env python3
"""残りの重要ファイルを解析する"""

import sys
sys.path.append('/Users/mymac/manxo')
from scripts.db_connector import DatabaseConnector
from scripts.analyze_patch_connections import PatchConnectionAnalyzer
from pathlib import Path
import json

def main():
    db = DatabaseConnector('scripts/db_settings.ini')
    db.connect()
    
    # 既に分析済みのファイル名を取得
    result = db.execute_query("""
    SELECT DISTINCT SUBSTRING(patch_file FROM '[^/]+$') as filename
    FROM object_connections
    """)
    analyzed_files = {r['filename'] for r in result}
    
    # HIGH重要度のファイルリスト
    high_priority_files = [
        # maxpatファイル
        'mi-gen_smc2019_slide2.maxpat',
        'NoOne511_UE5-MaxMSP-OSC-comm_3dgran-fromtut_08MC.maxpat',
        'jit.grab@.maxpat',
        'sdinulescu_AnaloGen_WorkingExamples-All.maxpat',
        'wwerkk_MC-FP_synth-gen.maxpat',
        'jit.videoplanesP.maxpat',
        'jit.videoplanes.maxpat',
        'jit.display@.maxpat',
        'cskonopka_arduivis_codegen-output.maxpat',
        'jinpavg_rnbo-unity-webgl_game-audio.maxpat',
        # amxdファイル
        'SHEPARD.TONE.MIDI.GENERATOR.amxd',
    ]
    
    # 巨大ファイルも追加
    large_files = [
        'RhythmVAE_M4L_M4L.RhythmVAE.amxd',
        'max  devices_XO_DrumRack.amxd',
        'XO_DrumRack.amxd',
    ]
    
    all_targets = high_priority_files + large_files
    
    print("=== 未処理重要ファイルの解析開始 ===\n")
    
    # PatchConnectionAnalyzerを使用
    analyzer = PatchConnectionAnalyzer(db)
    success_count = 0
    error_count = 0
    
    for filename in all_targets:
        if filename in analyzed_files:
            print(f"✅ {filename} - 既に分析済み")
            continue
            
        # ファイルパスを探す
        file_type = 'amxd' if filename.endswith('.amxd') else 'maxpat'
        base_dir = f'/Users/mymac/manxo/Max_Projects/all_{file_type}'
        filepath = Path(base_dir) / filename
        
        if not filepath.exists():
            print(f"❌ {filename} - ファイルが見つかりません")
            continue
            
        print(f"\n📋 {filename} を解析中...")
        
        try:
            # ファイルサイズ確認
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"   サイズ: {size_mb:.1f} MB")
            
            if size_mb > 50:  # 50MB以上は警告
                print(f"   ⚠️  巨大ファイル！解析をスキップします")
                continue
            
            # 解析実行
            connections, objects = analyzer.analyze_patch(str(filepath))
            
            if connections:
                print(f"   ✅ 成功: {len(connections)} 接続, {len(objects)} オブジェクト")
                success_count += 1
                
                # 興味深いオブジェクトを表示
                interesting_types = set()
                for obj in objects.values():
                    obj_type = obj.get('object_type', '')
                    if any(keyword in obj_type for keyword in ['gen', 'jit.', 'mc.', 'rnbo', 'poly~']):
                        interesting_types.add(obj_type)
                
                if interesting_types:
                    print(f"   興味深いオブジェクト: {', '.join(sorted(interesting_types)[:10])}")
            else:
                print(f"   ⚠️  接続が見つかりませんでした")
                
        except Exception as e:
            print(f"   ❌ エラー: {str(e)}")
            error_count += 1
    
    # 結果サマリー
    print(f"\n\n=== 解析結果サマリー ===")
    print(f"成功: {success_count} ファイル")
    print(f"エラー: {error_count} ファイル")
    
    # 更新後の統計を確認
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