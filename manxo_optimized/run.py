#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Max/MSP接続分析エンジン - メイン実行スクリプト
"""
import sys
import os
import argparse
from pathlib import Path

# 親ディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from core.box_types.improved_analyzer import ImprovedBoxTypeAnalyzer
from core.analyzers.subpatcher_analyzer import SubpatcherAnalyzer

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="Max/MSP パッチ分析ツール")
    parser.add_argument("input", help="分析する.maxpatファイルまたはディレクトリ")
    parser.add_argument("--output", "-o", help="出力ディレクトリ", default="./output")
    parser.add_argument("--analyze", "-a", choices=["box_types", "subpatchers", "all"], 
                        default="all", help="実行する分析タイプ")
    parser.add_argument("--debug", "-d", action="store_true", help="デバッグモード")
    args = parser.parse_args()
    
    # 出力ディレクトリの作成
    os.makedirs(args.output, exist_ok=True)
    
    # 指定された分析を実行
    if args.analyze in ["box_types", "all"]:
        print(f"ボックスタイプ分析を実行中: {args.input}")
        analyzer = ImprovedBoxTypeAnalyzer(debug=args.debug)
        if os.path.isdir(args.input):
            import glob
            files = glob.glob(os.path.join(args.input, "**/*.maxpat"), recursive=True)
            results = []
            for file in files:
                result = analyzer.analyze_patch_file(file)
                if result:
                    results.append(result)
            if results:
                analyzer.generate_report(results, args.output)
        else:
            result = analyzer.analyze_patch_file(args.input)
            if result:
                analyzer.generate_report([result], args.output)
    
    if args.analyze in ["subpatchers", "all"]:
        print(f"サブパッチャー分析を実行中: {args.input}")
        # サブパッチャー分析の実装をここに追加
        # ...
    
    print(f"分析完了。結果は {args.output} ディレクトリに保存されました。")

if __name__ == "__main__":
    main()
