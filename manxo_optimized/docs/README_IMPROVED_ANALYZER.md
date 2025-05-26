# Max/MSP 改良型ボックスタイプ分析ツール

このプロジェクトは、Max/MSPのパッチファイルを解析して、ボックスタイプの詳細な分析を行うツールを提供します。特に「オブジェクトボックス」と「オブジェクト」の重要な区別を理解し、正確な分析を行います。

## 概要

Max/MSPパッチファイル（.maxpat）の構造を解析し、以下の情報を抽出・分析します：

- maxclassの種類と分布
- オブジェクトタイプ（newobjの実際の内容）
- ボックスタイプの統計
- オブジェクトカテゴリ分類
- パッチの階層構造

## セットアップ

1. リポジトリをクローンまたはダウンロード
2. 必要なパッケージをインストール:
   ```
   pip install pandas numpy
   ```

## 使用方法

### 不要ファイルの整理

まず、大量の不要ファイルを整理するために以下のスクリプトを実行します：

```bash
# 実行権限を付与
chmod +x cleanup_unnecessary_files.sh
# 実行
./cleanup_unnecessary_files.sh
```

このスクリプトは不要なファイルを一時ディレクトリに移動します。確認後、完全に削除するか決定できます。

### コアファイルの整理

次に、compass_artifact文書に基づいたコアファイルを整理するスクリプトを実行します：

```bash
# 実行権限を付与
chmod +x organize_core_files.sh
# 実行
./organize_core_files.sh
```

このスクリプトはプロジェクトのコアファイルを整理し、新しいディレクトリに配置します。

### 改良型ボックスタイプ分析ツールの使用

```bash
# 単一のMaxパッチファイルを分析
python improved_box_type_analyzer.py path/to/your/patch.maxpat --output ./analysis_results

# ディレクトリ内の全Maxパッチファイルを分析
python improved_box_type_analyzer.py path/to/directory --output ./analysis_results

# 詳細なデバッグ情報を表示
python improved_box_type_analyzer.py path/to/your/patch.maxpat --debug
```

## 出力形式

分析結果は以下の形式で出力されます：

1. **Markdownレポート** (`box_type_analysis_report.md`):
   - 基本統計情報
   - maxclassタイプの分布
   - オブジェクトタイプの分布
   - オブジェクトカテゴリの分布

2. **JSON詳細データ** (`box_type_analysis_data.json`):
   - 全ボックスの詳細情報（JSON形式）
   - ID、maxclass、テキスト、実際のオブジェクトタイプなどを含む

## 高度な使用例

### テスト用パッチで実行

```bash
# テストディレクトリのパッチで実行
python improved_box_type_analyzer.py ./test_data --output ./test_results
```

### 特定のパターンのファイルのみ分析

```bash
# .amxdファイルのみを分析
python improved_box_type_analyzer.py ./Max_Projects --pattern "**/*.amxd" --output ./amxd_results
```

## 主要コンポーネント解説

- **improved_box_type_analyzer.py**: メインの分析スクリプト
- **tools/graph_node.py**: グラフノードクラス階層（各ボックスタイプ用）
- **tools/graph_node_factory.py**: 適切なノードタイプを生成するファクトリークラス
- **tools/patch_loader.py**: Maxパッチファイルを読み込み、グラフ構造に変換
- **tools/patch_visualizer.py**: パッチグラフの可視化ツール
- **subpatcher_analyzer.py**: サブパッチャーの階層構造分析

## 理論的背景

このツールは以下の重要な概念に基づいています：

1. **オブジェクトボックスとオブジェクトの区別**:
   - オブジェクトボックス: パッチエディタ上の視覚的コンテナ (maxclass="newobj")
   - オブジェクト: 実際の機能単位 (textプロパティの最初の単語)

2. **maxclassによるボックスタイプの識別**:
   - newobj, message, comment, number, flonum, toggle, button, slider など

3. **サブパッチャーの階層処理**:
   - 埋め込みパッチャー（p, patcher）
   - 参照パッチャー（abstraction）
   - bpatcherによる視覚的サブパッチャー

詳細な概念説明は `docs/MAX_BOX_TYPES_COMPREHENSIVE.md` を参照してください。