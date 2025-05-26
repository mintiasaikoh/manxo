# CLAUDE.md

This file provides guidance to Claude Code when working with MANXO.

# MANXO - Max/MSP AI Native eXperience Optimizer

**最終更新**: 2025年5月27日

## プロジェクト概要

自然言語からMax/MSPパッチを自動生成するAIシステム

### システム状態
- **分析エンジン**: 100% 完了 ✅
- **689,098接続** を完全分析済み
- **1,269,614オブジェクト** の詳細情報を収集済み
- **ディープラーニング**: 開発中 🚧

## メインコマンド

### 分析実行
```bash
python scripts/analyze_patch_connections.py /path/to/patch.maxpat
```

### データベース確認
```bash
python scripts/db_connector.py
```

## 重要な指示

1. **既存ファイルの無断変更禁止**
2. **明示的指示のない作業禁止**
3. **NumPy/pandas優先使用**
4. **既存スクリプト参照必須**

## データベース

**object_connections**: 689,098件の値付き接続データ
**object_details**: 1,269,614個のオブジェクト詳細情報

PostgreSQL使用、Live 12完全対応