# MANXO - Max/MSP AI Native eXperience Optimizer

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Max/MSP](https://img.shields.io/badge/Max/MSP-8.6+-orange.svg)](https://cycling74.com/)
[![Status](https://img.shields.io/badge/status-in%20development-yellow.svg)]()

自然言語からMax/MSPパッチを自動生成するAIシステム（開発中）

## 🚧 Development Status

**⚠️ このプロジェクトは現在開発初期段階です**

### 完了済み ✅
- Max/MSPパッチ分析エンジン
- PostgreSQLデータベース（689,098接続を分析済み）
- 基本的なCLI枠組み（プレースホルダー）

### 開発中 🚧
- Neural Knowledge Base実装 ([Issue #1](https://github.com/mintiasaikoh/manxo/issues/1))
- GNNモデルトレーニング ([Issue #2](https://github.com/mintiasaikoh/manxo/issues/2))
- 自然言語処理統合 ([Issue #3](https://github.com/mintiasaikoh/manxo/issues/3))

### 計画中 📅
- 実際のパッチ生成機能
- ストリーミング学習
- マルチモーダル入力対応

## 🎯 Vision

MANXOは、自然言語の説明からMax/MSPパッチを自動生成することを目指しています：

```
「リバーブエフェクトを作って」 → AI処理 → .maxpatファイル生成
```

しかし、**現時点では実際の生成機能は実装されていません**。

## 🚀 Getting Started (For Developers)

### Prerequisites

- Python 3.9+
- PostgreSQL 14+
- Max/MSP 8.6+ (パッチ分析用)

### Installation

```bash
# Clone repository
git clone https://github.com/mintiasaikoh/manxo.git
cd manxo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup database
createdb max_patch_analysis
python scripts/setup_database.py
```

### Current Functionality

現在利用可能な機能：

```bash
# パッチファイルの分析（データベースに保存）
python scripts/analyze_patch_connections.py /path/to/patch.maxpat

# CLIデモ（プレースホルダーのみ）
python scripts/manxo_cli.py "リバーブエフェクト"
# 注意：実際のパッチ生成はまだ実装されていません
```

## 🏗️ Architecture

```
[計画中のアーキテクチャ]
ユーザー入力 → NLP処理 → Neural KB検索 → GNN生成 → パッチ出力

[現在の実装]
パッチファイル → 分析 → PostgreSQLに保存
```

詳細は[CLAUDE.md](CLAUDE.md)を参照してください。

## 🤝 Contributing

このプロジェクトは貢献者を募集しています！

### 優先度の高いタスク

1. **Neural Knowledge Base実装** - PyTorchを使った学習可能インデックス
2. **GNNモデル実装** - PyTorch Geometricを使ったグラフニューラルネットワーク  
3. **テストカバレッジ向上** - 現在のカバレッジ: 約5%

詳細は[CONTRIBUTING.md](CONTRIBUTING.md)と[Issues](https://github.com/mintiasaikoh/manxo/issues)を確認してください。

## 📊 Data Status

- 分析済みパッチ: 11,894ファイル
- 収集済み接続パターン: 689,098
- オブジェクト情報: 1,269,614
- 対応オブジェクトタイプ: 1,598種類

## 🧪 Development Setup

```bash
# Run tests
pytest scripts/tests/

# Code formatting
black scripts/
flake8 scripts/

# Database status check
python scripts/db_connector.py
```

## 📚 Documentation

- [CLAUDE.md](CLAUDE.md) - 開発ガイド、コーディング規約
- [CONTRIBUTING.md](CONTRIBUTING.md) - 貢献ガイドライン
- [GitHub Issues](https://github.com/mintiasaikoh/manxo/issues) - タスク管理

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Max/MSP community for inspiration
- Contributors who help build this vision

---

**Note**: MANXO is an ambitious research project in early development. While we have successfully analyzed thousands of patches, the actual AI generation capabilities are still being developed. Join us in making this vision a reality!