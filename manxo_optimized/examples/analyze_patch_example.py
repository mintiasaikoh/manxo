#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Max/MSP接続分析エンジン - 使用例

この例では、ボックスタイプ分析とサブパッチャー分析を実行する方法を示します。
"""
import os
import sys
from pathlib import Path

# 親ディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.box_types.improved_analyzer import ImprovedBoxTypeAnalyzer

def main():
    """メイン実行関数"""
    # テストデータディレクトリ
    data_dir = Path(__file__).parent.parent / "data"
    
    # テストファイルが存在するか確認
    test_files = list(data_dir.glob("**/*.maxpat"))
    if not test_files:
        print(f"エラー: テストファイルが見つかりません: {data_dir}")
        return
    
    # 最初のテストファイルを分析
    test_file = test_files[0]
    print(f"ファイル '{test_file}' を分析中...")
    
    # ボックスタイプ分析
    analyzer = ImprovedBoxTypeAnalyzer(debug=True)
    result = analyzer.analyze_patch_file(str(test_file))
    
    if result:
        # 結果を表示
        print(f"分析結果:")
        print(f"- パッチ名: {result['patch_name']}")
        print(f"- ボックス数: {result['box_count']}")
        print(f"- maxclass統計:")
        for mc, count in result.get('maxclass_counts', {}).items():
            print(f"  - {mc}: {count}")
    else:
        print("分析に失敗しました")

if __name__ == "__main__":
    main()
