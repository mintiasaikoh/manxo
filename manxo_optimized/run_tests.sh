#!/bin/bash
# テスト実行スクリプト

# 色付きの出力
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Max/MSP接続分析エンジン - テスト実行${NC}"
echo "================================="

# 現在のディレクトリをプロジェクトルートに設定
cd "$(dirname "$0")" || exit 1

# Pythonパスにプロジェクトディレクトリを追加
export PYTHONPATH=$(pwd):$PYTHONPATH

# テストディレクトリが存在するか確認
if [ ! -d "tests" ]; then
    echo -e "${RED}エラー: testsディレクトリが見つかりません${NC}"
    exit 1
fi

# テストを実行
echo -e "${BLUE}単体テストを実行中...${NC}"
python -m unittest discover tests

# テスト結果を確認
if [ $? -eq 0 ]; then
    echo -e "${GREEN}全てのテストが成功しました！${NC}"
else
    echo -e "${RED}テストが失敗しました${NC}"
fi
