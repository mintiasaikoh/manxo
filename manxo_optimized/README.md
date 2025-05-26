# Max/MSP接続分析エンジン (最適化版)

このプロジェクトは、Max/MSPのパッチファイルを分析するためのツールスイートです。特に「オブジェクトボックス」と「オブジェクト」の区別を適切に行い、パッチ構造を正確に解析します。

## モジュール構造

- **core/**: コアモジュール群
  - **box_types/**: ボックスタイプ関連コード
  - **graph/**: グラフ表現モジュール
  - **parsers/**: パーサーモジュール
  - **analyzers/**: 分析モジュール
  - **visualizers/**: 可視化モジュール
- **scripts/**: 実行スクリプト
- **tests/**: テストコード
- **docs/**: ドキュメント
- **examples/**: 使用例
- **data/**: テストデータ

## インストール方法

必要なパッケージをインストール:

```bash
pip install pandas numpy matplotlib networkx
```

## 使用方法

### ボックスタイプ分析

```bash
# 単一のMaxパッチファイルを分析
python run.py path/to/your/patch.maxpat --analyze box_types

# ディレクトリ内の全Maxパッチファイルを分析
python run.py path/to/directory --analyze box_types
```

### サブパッチャー分析

```bash
# サブパッチャー構造を分析
python run.py path/to/your/patch.maxpat --analyze subpatchers
```

### 総合分析

```bash
# 全ての分析を実行
python run.py path/to/your/patch.maxpat --analyze all
```

## 詳細ドキュメント

詳細については、`docs/`ディレクトリのドキュメントを参照してください。

- **MAX_BOX_TYPES.md**: ボックスタイプの基本説明
- **MAX_BOX_TYPES_COMPREHENSIVE.md**: 詳細なボックスタイプ解説
- **IMPLEMENTATION_PLAN.md**: 実装計画書

## コントリビューション

バグ報告や機能リクエストは、GitHubのIssueで報告してください。
