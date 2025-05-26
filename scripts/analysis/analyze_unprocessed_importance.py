#!/usr/bin/env python3
"""未処理ファイルの重要度を分析"""

import sys
sys.path.append('/Users/mymac/manxo')
from scripts.db_connector import DatabaseConnector
from pathlib import Path
import re

def main():
    db = DatabaseConnector('scripts/db_settings.ini')
    db.connect()

    # 既に分析済みのファイル名を取得
    result = db.execute_query("""
    SELECT DISTINCT SUBSTRING(patch_file FROM '[^/]+$') as filename
    FROM object_connections
    """)
    analyzed_files = {r['filename'] for r in result}

    # ディスク上のファイルを取得
    all_maxpat = list(Path('/Users/mymac/manxo/Max_Projects/all_maxpat').glob('*.maxpat'))
    all_amxd = list(Path('/Users/mymac/manxo/Max_Projects/all_amxd').glob('*.amxd'))

    # 未分析ファイルを特定
    unanalyzed_maxpat = [f for f in all_maxpat if f.name not in analyzed_files]
    unanalyzed_amxd = [f for f in all_amxd if f.name not in analyzed_files]

    print('=== 未処理ファイルの重要度分析 ===\n')

    # 重要そうなキーワード
    important_keywords = {
        'high': ['RNBO', 'gen~', 'gen', 'jit.', 'live.', 'vst~', 'amxd~', 'poly~', 
                 'pfft~', 'mc.', 'spat', 'groove~', 'buffer~', 'sfplay~'],
        'medium': ['filter', 'reverb', 'delay', 'synth', 'sampler', 'sequencer', 
                   'drum', 'effect', 'midi', 'osc', 'modular', 'granular'],
        'low': ['test', 'example', 'demo', 'tutorial', 'help', 'template']
    }

    # ファイルサイズも確認
    def categorize_file(filepath):
        name_lower = filepath.name.lower()
        size_kb = filepath.stat().st_size / 1024
        
        importance = 'normal'
        matched_keywords = []
        
        # キーワードマッチング
        for level, keywords in important_keywords.items():
            for keyword in keywords:
                if keyword in name_lower:
                    if level == 'high':
                        importance = 'high'
                    elif level == 'medium' and importance != 'high':
                        importance = 'medium'
                    elif level == 'low' and importance == 'normal':
                        importance = 'low'
                    matched_keywords.append(keyword)
        
        # サイズによる判定
        if size_kb > 500:  # 500KB以上は複雑なパッチの可能性
            if importance == 'normal':
                importance = 'medium'
        
        return importance, matched_keywords, size_kb

    # 未分析ファイルを重要度で分類
    print('【maxpatファイル】')
    maxpat_by_importance = {'high': [], 'medium': [], 'low': [], 'normal': []}

    for f in unanalyzed_maxpat:
        importance, keywords, size = categorize_file(f)
        maxpat_by_importance[importance].append((f, keywords, size))

    for level in ['high', 'medium']:
        files = maxpat_by_importance[level]
        if files:
            print(f'\n{level.upper()}重要度: {len(files)}個')
            for f, keywords, size in sorted(files, key=lambda x: x[2], reverse=True)[:10]:
                kw_str = ', '.join(keywords) if keywords else ''
                print(f'  {f.name[:60]:60s} ({size:6.1f}KB) [{kw_str}]')

    print(f'\nLOW重要度: {len(maxpat_by_importance["low"])}個')
    print(f'通常: {len(maxpat_by_importance["normal"])}個')

    # amxdファイルも同様に
    print('\n\n【amxdファイル】')
    amxd_by_importance = {'high': [], 'medium': [], 'low': [], 'normal': []}

    for f in unanalyzed_amxd:
        importance, keywords, size = categorize_file(f)
        amxd_by_importance[importance].append((f, keywords, size))

    for level in ['high', 'medium']:
        files = amxd_by_importance[level]
        if files:
            print(f'\n{level.upper()}重要度: {len(files)}個')
            for f, keywords, size in sorted(files, key=lambda x: x[2], reverse=True)[:10]:
                kw_str = ', '.join(keywords) if keywords else ''
                print(f'  {f.name[:60]:60s} ({size:6.1f}KB) [{kw_str}]')

    print(f'\nLOW重要度: {len(amxd_by_importance["low"])}個')
    print(f'通常: {len(amxd_by_importance["normal"])}個')

    # 特に興味深いファイル名をピックアップ
    print('\n\n=== 特に興味深いファイル名 ===')

    interesting_patterns = [
        (r'RNBO', 'RNBO関連'),
        (r'gen[~\s]', 'Gen関連'),
        (r'(Red Panda|Mutable|Elektron|Moog)', '有名ハードウェア関連'),
        (r'(granular|spectral|convol)', '高度な信号処理'),
        (r'(AI|ML|machine.?learning|neural)', 'AI/機械学習関連'),
        (r'(Arduino|ESP|Teensy)', 'ハードウェア連携'),
        (r'(live\.|Live)', 'Max for Live関連'),
    ]

    for pattern, description in interesting_patterns:
        print(f'\n{description}:')
        found = False
        for f in unanalyzed_maxpat + unanalyzed_amxd:
            if re.search(pattern, f.name, re.IGNORECASE):
                size = f.stat().st_size / 1024
                print(f'  {f.name[:70]:70s} ({size:6.1f}KB)')
                found = True
        if not found:
            print('  （該当なし）')

    # 統計サマリー
    print('\n\n=== サマリー ===')
    total_unanalyzed = len(unanalyzed_maxpat) + len(unanalyzed_amxd)
    high_count = len(maxpat_by_importance['high']) + len(amxd_by_importance['high'])
    medium_count = len(maxpat_by_importance['medium']) + len(amxd_by_importance['medium'])
    
    print(f'未分析ファイル総数: {total_unanalyzed}')
    print(f'  HIGH重要度: {high_count} ({high_count/total_unanalyzed*100:.1f}%)')
    print(f'  MEDIUM重要度: {medium_count} ({medium_count/total_unanalyzed*100:.1f}%)')
    
    if high_count == 0:
        print('\n✅ 重要度の高いファイルは見つかりませんでした。')
        print('   既に主要なオブジェクトやパターンは分析済みです。')
    else:
        print(f'\n⚠️  {high_count}個の重要度の高いファイルが未分析です。')

    db.disconnect()

if __name__ == "__main__":
    main()