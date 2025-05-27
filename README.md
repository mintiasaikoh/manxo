# MANXO - Max/MSP AI Native eXperience Optimizer

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Max/MSP](https://img.shields.io/badge/Max/MSP-8.6+-orange.svg)](https://cycling74.com/)

自然言語からMax/MSPパッチを自動生成するAIシステム

## 🎯 Overview

MANXOは、日本語や英語の自然な説明から、完全に動作するMax/MSPパッチ（.maxpat/.amxd）を生成する世界初のAIシステムです。

```
「雨の音を表現したアンビエント」 → AIが理解・生成 → 完全な.maxpatファイル
```

## ✨ Features

- 🗣️ **自然言語理解**: 日本語・英語の説明を理解
- 🎨 **創造的生成**: 抽象的な概念から具体的なパッチを生成
- ⚡ **高速生成**: キャッシュ利用で0.0秒、新規でも3-5秒
- 🏗️ **複雑な構造対応**: サブパッチャー、poly~、gen~、RNBO対応
- 🔍 **検証機能**: 生成されたパッチの構造を自動検証

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/mintiasaikoh/manxo.git
cd manxo

# Install dependencies
pip install -r requirements.txt

# Setup database
createdb manxo
python scripts/setup_database.py
```

### Basic Usage

```bash
# Generate a patch from natural language
python scripts/manxo_cli.py "リバーブエフェクトを作って"

# Interactive mode
python scripts/manxo_cli.py --interactive

# Batch generation
python scripts/manxo_cli.py --batch "reverb,delay,filter"
```

## 📊 Performance

- **分析済みデータ**: 11,894ファイル、689,098接続
- **GNN予測精度**: 98.57%
- **生成速度**: 0.0秒（キャッシュ）〜 5秒（複雑なパッチ）
- **対応オブジェクト**: 1,598種類（Live 12対応）

## 📚 Documentation

- [CLAUDE.md](CLAUDE.md) - Detailed system documentation
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contributing guidelines

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**MANXO** - Making Max/MSP AI-Native eXperience Optimal 🎵