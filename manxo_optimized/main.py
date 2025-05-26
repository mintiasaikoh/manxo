#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
メインエントリーポイント
コマンドラインからの実行用
"""

import argparse
import logging
import sys
import os

from src.generators.maxpat_generator import MaxPatGenerator
from src.utils.jitter_utils import JitterParamsIntegrator

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_patch(args):
    """maxpatファイルを生成"""
    generator = MaxPatGenerator()
    result = generator.generate(
        description=args.description,
        file_name=args.file_name,
        category=args.category,
        complexity=args.complexity
    )
    
    print(f"Generated maxpat file: {result['outputPath']}")
    return 0

def integrate_jitter_params(args):
    """jitterパラメーター指標を統合"""
    integrator = JitterParamsIntegrator(
        jitter_params_path=args.jitter_params_path,
        db_path=args.db_path
    )
    result = integrator.integrate()
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return 1
    
    print(f"Updated {result['updated']} objects (errors: {result['errors']})")
    return 0

def run_server(args):
    """MCPサーバーを実行"""
    from src.api.mcp_server import main as run_server
    run_server()
    return 0

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Max/MSP Tools")
    subparsers = parser.add_subparsers(dest="command", help="コマンド")
    
    # generate-patchコマンド
    generate_parser = subparsers.add_parser("generate-patch", help="maxpatファイルを生成")
    generate_parser.add_argument("description", help="パッチの説明")
    generate_parser.add_argument("--file-name", help="出力ファイル名")
    generate_parser.add_argument("--category", default="general", help="カテゴリ (audio, video, control, general)")
    generate_parser.add_argument("--complexity", default="medium", help="複雑さ (simple, medium, complex)")
    
    # integrate-jitter-paramsコマンド
    integrate_parser = subparsers.add_parser("integrate-jitter-params", help="jitterパラメーター指標を統合")
    integrate_parser.add_argument("--jitter-params-path", help="jitterパラメーター指標ファイルのパス")
    integrate_parser.add_argument("--db-path", help="データベースのパス")
    
    # run-serverコマンド
    server_parser = subparsers.add_parser("run-server", help="MCPサーバーを実行")
    
    args = parser.parse_args()
    
    if args.command == "generate-patch":
        return generate_patch(args)
    elif args.command == "integrate-jitter-params":
        return integrate_jitter_params(args)
    elif args.command == "run-server":
        return run_server(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
