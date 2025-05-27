# CLAUDE.md - MANXO開発ガイド

This file provides comprehensive guidance for developing MANXO with Claude Code.

## 🎯 MANXO とは

**Max/MSP AI Native eXperience Optimizer** - 自然言語からMax/MSPパッチを自動生成するAIシステム

### 現在の開発フェーズ
- ✅ **Phase 1**: データ収集・分析 (完了)
- 🚧 **Phase 2**: Neural Knowledge Base実装 (現在)
- 📅 **Phase 3**: 自然言語→パッチ生成 (計画中)

## 📝 コーディング規約

### 命名規則

#### クラス名
```python
# ✅ 良い例 - PascalCase使用
class MaxPatchNeuralKB:
class PatchGraphEncoder:
class ConnectionAnalyzer:

# ❌ 悪い例
class max_patch_neural_kb:  # snake_caseは使わない
class patchgraphencoder:     # 単語の区切りなし
```

#### 関数名・メソッド名
```python
# ✅ 良い例 - snake_case使用
def analyze_patch_connections(patch_file: str) -> Dict:
def load_from_database(db_config: str) -> List[Dict]:
def _private_method(self) -> None:  # プライベートは_で開始

# ❌ 悪い例
def AnalyzePatchConnections():  # PascalCaseは使わない
def loadFromDB():               # キャメルケースは使わない
```

#### 変数名
```python
# ✅ 良い例
connection_count = 689098
source_object_type = "newobj"
is_audio_effect = True
MAX_PATCH_SIZE = 1000000  # 定数は大文字

# ❌ 悪い例
connectionCount = 689098   # キャメルケース避ける
src_obj_typ = "newobj"    # 過度な省略避ける
```

### ディレクトリ構造

```
manxo/
├── scripts/           # 実行可能スクリプト
│   ├── models/       # ニューラルネットワークモデル
│   ├── utils/        # ユーティリティ関数
│   └── tests/        # テストコード
├── data/             # データファイル
├── models/           # 学習済みモデル
└── docs/             # ドキュメント
```

### インポート順序

```python
# 1. 標準ライブラリ
import os
import sys
from pathlib import Path

# 2. サードパーティライブラリ
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

# 3. ローカルモジュール
from scripts.db_connector import DatabaseConnector
from scripts.models.patch_gnn import PatchGNN
```

### 型ヒント使用

```python
from typing import Dict, List, Optional, Tuple, Union

def process_patch(
    patch_data: Dict[str, Any],
    max_objects: Optional[int] = None
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """パッチデータを処理し、オブジェクトと接続を返す。
    
    Args:
        patch_data: パッチのJSONデータ
        max_objects: 処理する最大オブジェクト数
        
    Returns:
        (objects, connections) のタプル
    """
    pass
```

### エラーハンドリング

```python
# ✅ 良い例 - 具体的な例外処理
try:
    db.connect()
except psycopg2.OperationalError as e:
    logger.error(f"Database connection failed: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    return None

# ❌ 悪い例 - 汎用的すぎる
try:
    db.connect()
except:
    pass  # エラーを握りつぶさない
```

### ドキュメンテーション

```python
class MaxPatchNeuralKB(nn.Module):
    """Max/MSPパッチのための Neural Knowledge Base。
    
    学習可能なインデックスを使用して、自然言語クエリから
    関連するパッチパターンを検索する。
    
    Attributes:
        knowledge_size: 知識ベースのサイズ（デフォルト: 689098）
        d_model: 埋め込み次元（デフォルト: 768）
        index_keys: 学習可能なインデックスキー
        index_values: 学習可能なインデックス値
    """
    
    def __init__(self, knowledge_size: int = 689098, d_model: int = 768):
        """Neural Knowledge Baseを初期化する。
        
        Args:
            knowledge_size: 知識ベースに保存する接続パターン数
            d_model: 埋め込みベクトルの次元数
        """
        super().__init__()
        self.knowledge_size = knowledge_size
        self.d_model = d_model
```

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

# Python環境 (Python 3.9以上)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# データベース作成
createdb max_patch_analysis

# テーブル作成とセットアップ
python scripts/setup_database.py
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
# scripts/models/neural_kb.py
import torch
import torch.nn as nn
from typing import Dict, Optional

class MaxPatchNeuralKB(nn.Module):
    """Max/MSPパッチのための Neural Knowledge Base。"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.knowledge_size = config.get('knowledge_size', 689098)
        self.d_model = config.get('d_model', 768)
        
        # 学習可能なインデックス
        self.index_keys = nn.Parameter(
            torch.randn(self.knowledge_size, self.d_model)
        )
        self.index_values = nn.Parameter(
            torch.randn(self.knowledge_size, self.d_model)
        )
        
        # 階層的埋め込み
        self.object_embedding = nn.Embedding(1600, self.d_model)
        self.hierarchy_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.d_model, nhead=8), 
            num_layers=4
        )
        
    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """クエリに基づいて知識を検索する。
        
        Args:
            query: クエリベクトル [batch_size, d_model]
            
        Returns:
            関連する知識ベクトル [batch_size, d_model]
        """
        # クエリと知識ベースの類似度計算
        similarities = torch.matmul(query, self.index_keys.T)
        weights = torch.softmax(similarities, dim=-1)
        
        # 重み付き知識の取得
        knowledge = torch.matmul(weights, self.index_values)
        return knowledge
```

### 4. GNNモデルの実装

```python
# scripts/models/patch_gnn.py
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

class PatchGNN(nn.Module):
    """パッチ構造を学習するGraph Neural Network。"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.num_features = config['num_features']
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_classes = config['num_classes']
        
        # グラフ畳み込み層
        self.conv1 = GCNConv(self.num_features, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.conv3 = GCNConv(self.hidden_dim, self.num_classes)
        
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """グラフ構造を処理する。
        
        Args:
            x: ノード特徴量 [num_nodes, num_features]
            edge_index: エッジインデックス [2, num_edges]
            batch: バッチインデックス [num_nodes]
            
        Returns:
            グラフレベルの予測 [batch_size, num_classes]
        """
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        
        if batch is not None:
            x = global_mean_pool(x, batch)
            
        return x
```

## 📝 開発タスク優先順位

### 今すぐ始められるタスク

1. **データベーススキーマの確認と調整**
```bash
psql max_patch_analysis -f scripts/create_tables.sql
```

2. **データセット準備スクリプト作成**
```python
# scripts/create_graph_dataset.py
python scripts/create_graph_dataset.py --limit 1000
```

3. **簡単なパッチ生成テスト**
```bash
python scripts/manxo_cli.py "Create a simple oscillator"
```

### 次のステップ

1. Neural KBのプロトタイプ実装 (Issue #1)
2. 既存データでGNNをトレーニング (Issue #2)
3. 簡単な自然言語→パッチのデモ作成 (Issue #3)

## 🔧 よく使うコマンド

```bash
# 単一パッチの分析
python scripts/analyze_patch_connections.py /path/to/patch.maxpat

# CLIでパッチ生成
python scripts/manxo_cli.py "リバーブエフェクトを作って"

# インタラクティブモード
python scripts/manxo_cli.py --interactive

# データベース状態確認
python scripts/db_connector.py

# テスト実行
pytest scripts/tests/

# コードフォーマット
black scripts/
flake8 scripts/
```

## ⚠️ 重要な指示

1. **スクリプトの無断改変禁止** - 変更前に必ず説明と承認を得る
2. **既存ファイルの書き換え禁止** - 特にREADME.mdなど
3. **NumPy/pandas優先使用** - ループ処理より行列演算
4. **既存コード確認必須** - 車輪の再発明を避ける
5. **型ヒント必須** - すべての関数に型ヒントを付ける
6. **テスト作成必須** - 新機能には必ずテストを書く

## 🎯 最終目標

```
ユーザー: "ドラムマシンにサイドチェインかけたダブステップ作って"
    ↓
MANXO: [複雑な.amxdファイル生成]
    ↓
Ableton Live: 完璧に動作するデバイス
```

これを実現するために、今は基礎となるNeural KBとGNNを構築中です。

## 📚 参考資料

- [Max/MSP Documentation](https://docs.cycling74.com/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)