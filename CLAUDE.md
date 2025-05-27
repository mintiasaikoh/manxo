# CLAUDE.md - MANXO開発ガイド

This file provides comprehensive guidance for developing MANXO with Claude Code.

## 🎯 MANXO とは

**Max/MSP AI Native eXperience Optimizer** - 自然言語からMax/MSPパッチを自動生成するAIシステム

### 現在の開発フェーズ
- ✅ **Phase 1**: データ収集・分析 (完了)
- 🚧 **Phase 2**: Neural Knowledge Base実装 (現在)
- 📅 **Phase 3**: 自然言語→パッチ生成 (計画中)

## 🏗️ システムアーキテクチャ

```
[ユーザー入力] 
    ↓
[NLP処理層] - テキストを解析し、意図を理解
    ↓
[Neural KB] - 学習済みパッチパターンを検索
    ↓
[GNNモデル] - パッチ構造を予測・生成
    ↓
[検証層] - 生成パッチの妥当性確認
    ↓
[.maxpat/.amxd出力]
```

### 主要コンポーネント

1. **PostgreSQLデータベース**
   - 689,098接続パターン
   - 1,269,614オブジェクト情報
   - ポート接続タイプ情報

2. **Neural Knowledge Base（開発中）**
   - 学習可能なインデックス
   - 階層的埋め込み空間
   - マルチスケール注意機構

3. **GNNモデル（開発中）**
   - GraphSAGE/GCNアーキテクチャ
   - ノード：オブジェクト
   - エッジ：接続

## 🚀 開発を始める

### 1. 環境セットアップ

```bash
# PostgreSQLが必要
brew install postgresql
brew services start postgresql

# Python環境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# データベース作成
createdb max_patch_analysis

# テーブル作成
psql max_patch_analysis < scripts/create_tables.sql
```

### 2. 既存データの確認

```python
# データベース接続テスト
from scripts.db_connector import DatabaseConnector

db = DatabaseConnector('scripts/db_settings.ini')
db.connect()

# 接続数を確認
result = db.execute_query("SELECT COUNT(*) FROM object_connections")
print(f"Total connections: {result[0]['count']}")

# サンプル接続を表示
connections = db.execute_query("""
    SELECT source_object_type, source_value, 
           target_object_type, target_value 
    FROM object_connections 
    WHERE source_value IS NOT NULL 
    LIMIT 5
""")
for conn in connections:
    print(f"{conn['source_object_type']}('{conn['source_value']}') → {conn['target_object_type']}")

db.disconnect()
```

### 3. Neural KB実装の開始

```python
# Issue #1: Neural Knowledge Base Implementation
# 実装する必要があるクラス

import torch
import torch.nn as nn

class MaxPatchNeuralKB(nn.Module):
    def __init__(self, knowledge_size=689098, d_model=768):
        super().__init__()
        # 学習可能なインデックスキー
        self.index_keys = nn.Parameter(torch.randn(knowledge_size, d_model))
        self.index_values = nn.Parameter(torch.randn(knowledge_size, d_model))
        
        # 階層的埋め込み
        self.object_embedding = nn.Embedding(1600, d_model)  # 1600種類のオブジェクト
        self.hierarchy_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8), 
            num_layers=4
        )
        
    def forward(self, query):
        # クエリと知識ベースの類似度計算
        similarities = torch.matmul(query, self.index_keys.T)
        weights = torch.softmax(similarities, dim=-1)
        
        # 重み付き知識の取得
        knowledge = torch.matmul(weights, self.index_values)
        return knowledge

# 使用例
kb = MaxPatchNeuralKB()
query = torch.randn(1, 768)  # "リバーブエフェクト"のエンコード
knowledge = kb(query)
```

### 4. GNNモデルの実装

```python
# Issue #2: GNN Model Training
# PyTorch Geometricを使用

from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

class PatchGNN(nn.Module):
    def __init__(self, num_features, hidden_dim=256, num_classes=100):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, num_classes)
        
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        
        # グラフレベルの予測
        x = global_mean_pool(x, batch)
        return x
```

## 📝 開発タスク優先順位

### 今すぐ始められるタスク

1. **データベーススキーマの作成** (`scripts/create_tables.sql`)
```sql
CREATE TABLE IF NOT EXISTS object_connections (
    id SERIAL PRIMARY KEY,
    source_object_type VARCHAR NOT NULL,
    source_port INTEGER NOT NULL DEFAULT 0,
    target_object_type VARCHAR NOT NULL,
    target_port INTEGER NOT NULL DEFAULT 0,
    patch_file VARCHAR NOT NULL,
    file_type VARCHAR NOT NULL,
    source_value TEXT,
    target_value TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

2. **データセット準備スクリプト作成**
```bash
python scripts/create_graph_dataset.py
```

3. **簡単なパッチ生成テスト**
```python
# 基本的なオシレーター生成
from scripts.simple_patch_generator import generate_oscillator
patch = generate_oscillator(frequency=440)
patch.save("test_osc.maxpat")
```

### 次のステップ

1. Neural KBのプロトタイプ実装
2. 既存データでGNNをトレーニング
3. 簡単な自然言語→パッチのデモ作成

## 🔧 よく使うコマンド

```bash
# 単一パッチの分析
python scripts/analyze_patch_connections.py /path/to/patch.maxpat

# バッチ分析
python scripts/batch_analyze.py /path/to/patches/

# データベース状態確認
python scripts/check_db_status.py

# GNNトレーニング（GPU推奨）
python scripts/train_gnn.py --epochs 100 --batch-size 32
```

## ⚠️ 重要な指示

1. **スクリプトの無断改変禁止** - 変更前に必ず説明と承認を得る
2. **既存ファイルの書き換え禁止** - 特にREADME.mdなど
3. **NumPy/pandas優先使用** - ループ処理より行列演算
4. **既存コード確認必須** - 車輪の再発明を避ける

## 🎯 最終目標

```
ユーザー: "ドラムマシンにサイドチェインかけたダブステップ作って"
    ↓
MANXO: [複雑な.amxdファイル生成]
    ↓
Ableton Live: 完璧に動作するデバイス
```

これを実現するために、今は基礎となるNeural KBとGNNを構築中です。