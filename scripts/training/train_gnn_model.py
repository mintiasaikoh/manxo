#!/usr/bin/env python3
"""
Max/MSPパッチデータでGraph Neural Networkを学習

複数のタスクに対応:
1. 接続予測 - 次に接続されるオブジェクトを予測
2. デバイスタイプ分類 - audio_effect, midi_effect等を分類
3. オブジェクト推薦 - コンテキストに基づいて次のオブジェクトを推薦
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import negative_sampling
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
from tqdm import tqdm

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MaxPatchGNN(nn.Module):
    """
    Max/MSPパッチ用のGraph Neural Network
    
    特徴:
    - 複数のGNNレイヤー（GCN, GAT, GraphSAGE）を選択可能
    - ノード特徴量とエッジ特徴量の両方を活用
    - 階層的なグラフ構造に対応
    - マルチタスク学習対応
    """
    
    def __init__(self, 
                 node_feature_dim: int,
                 edge_feature_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 num_object_types: int = 1598,
                 num_device_types: int = 7,
                 gnn_type: str = 'GraphSAGE',
                 dropout: float = 0.2):
        """
        初期化
        
        Args:
            node_feature_dim: ノード特徴量の次元
            edge_feature_dim: エッジ特徴量の次元
            hidden_dim: 隠れ層の次元
            num_layers: GNNレイヤー数
            num_object_types: オブジェクトタイプ数（分類用）
            num_device_types: デバイスタイプ数（分類用）
            gnn_type: 'GCN', 'GAT', 'GraphSAGE'のいずれか
            dropout: ドロップアウト率
        """
        super(MaxPatchGNN, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = dropout
        
        # 入力層
        self.input_transform = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # エッジ特徴量の変換（GATで使用）
        self.edge_transform = nn.Linear(edge_feature_dim, hidden_dim)
        
        # GNNレイヤー
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if gnn_type == 'GCN':
                layer = GCNConv(hidden_dim, hidden_dim)
            elif gnn_type == 'GAT':
                # GATは複数のアテンションヘッドを使用
                layer = GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout)
            elif gnn_type == 'GraphSAGE':
                layer = GraphSAGE(hidden_dim, hidden_dim, num_layers=1)
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
            
            self.gnn_layers.append(layer)
            
        # バッチ正規化層
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # タスク固有のヘッド
        
        # 1. 接続予測ヘッド（ノードペアの類似度を計算）
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # 2. オブジェクトタイプ分類ヘッド
        self.object_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_object_types)
        )
        
        # 3. デバイスタイプ分類ヘッド（グラフレベル）
        self.device_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # mean + max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_device_types)
        )
        
        # 4. ノード埋め込みヘッド（推薦用）
        self.node_embedder = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        順伝播
        
        Args:
            x: ノード特徴量 [num_nodes, node_feature_dim]
            edge_index: エッジインデックス [2, num_edges]
            edge_attr: エッジ特徴量 [num_edges, edge_feature_dim]
            batch: バッチインデックス [num_nodes]
            
        Returns:
            各タスクの出力を含む辞書
        """
        # 入力変換
        h = self.input_transform(x)
        
        # GNNレイヤーを適用
        for i, (gnn_layer, bn) in enumerate(zip(self.gnn_layers, self.batch_norms)):
            if self.gnn_type == 'GAT' and edge_attr is not None:
                # GATの場合はエッジ特徴量も使用可能（カスタム実装が必要）
                h_new = gnn_layer(h, edge_index)
            else:
                h_new = gnn_layer(h, edge_index)
            
            # 残差接続
            if i > 0:
                h = h + h_new
            else:
                h = h_new
                
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # ノード埋め込み
        node_embeddings = self.node_embedder(h)
        
        # オブジェクトタイプ分類
        object_logits = self.object_classifier(h)
        
        # グラフレベル表現（デバイスタイプ分類用）
        if batch is not None:
            # バッチ処理の場合
            graph_mean = global_mean_pool(h, batch)
            graph_max = global_max_pool(h, batch)
            graph_repr = torch.cat([graph_mean, graph_max], dim=1)
        else:
            # 単一グラフの場合
            graph_mean = h.mean(dim=0, keepdim=True)
            graph_max = h.max(dim=0)[0].unsqueeze(0)
            graph_repr = torch.cat([graph_mean, graph_max], dim=1)
            
        device_logits = self.device_classifier(graph_repr)
        
        return {
            'node_embeddings': node_embeddings,
            'object_logits': object_logits,
            'device_logits': device_logits,
            'hidden_states': h
        }
        
    def predict_link(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        エッジの存在確率を予測
        
        Args:
            z: ノード埋め込み
            edge_index: 予測するエッジ
            
        Returns:
            エッジの存在確率
        """
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        edge_features = torch.cat([src, dst], dim=1)
        return torch.sigmoid(self.link_predictor(edge_features)).squeeze()


class MaxPatchGNNTrainer:
    """GNNモデルの学習を管理するクラス"""
    
    def __init__(self, model: MaxPatchGNN, device: str = 'cuda'):
        """
        初期化
        
        Args:
            model: GNNモデル
            device: 'cuda' or 'cpu'
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 最適化器
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # 損失関数
        self.link_loss_fn = nn.BCELoss()
        self.object_loss_fn = nn.CrossEntropyLoss()
        self.device_loss_fn = nn.CrossEntropyLoss()
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        1エポックの学習
        
        Args:
            train_loader: 学習データローダー
            
        Returns:
            損失の辞書
        """
        self.model.train()
        total_losses = {'link': 0, 'object': 0, 'device': 0, 'total': 0}
        num_batches = 0
        
        for batch in tqdm(train_loader, desc='Training'):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # 順伝播
            outputs = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            # 損失計算
            losses = {}
            
            # 1. リンク予測損失（負例サンプリング）
            pos_edge_index = batch.edge_index
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=batch.x.size(0),
                num_neg_samples=pos_edge_index.size(1)
            )
            
            pos_pred = self.model.predict_link(outputs['node_embeddings'], pos_edge_index)
            neg_pred = self.model.predict_link(outputs['node_embeddings'], neg_edge_index)
            
            link_loss = self.link_loss_fn(pos_pred, torch.ones_like(pos_pred))
            link_loss += self.link_loss_fn(neg_pred, torch.zeros_like(neg_pred))
            losses['link'] = link_loss
            
            # 2. オブジェクトタイプ分類損失（ラベルがある場合）
            if hasattr(batch, 'y_object'):
                object_loss = self.object_loss_fn(
                    outputs['object_logits'], 
                    batch.y_object
                )
                losses['object'] = object_loss
            
            # 3. デバイスタイプ分類損失
            if hasattr(batch, 'y_device'):
                device_loss = self.device_loss_fn(
                    outputs['device_logits'],
                    batch.y_device
                )
                losses['device'] = device_loss
            
            # 総損失
            total_loss = sum(losses.values())
            losses['total'] = total_loss
            
            # 逆伝播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 損失を記録
            for key, value in losses.items():
                total_losses[key] += value.item()
            num_batches += 1
            
        # 平均損失を返す
        return {key: value / num_batches for key, value in total_losses.items()}
        
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        検証データで評価
        
        Args:
            val_loader: 検証データローダー
            
        Returns:
            評価メトリクスの辞書
        """
        self.model.eval()
        metrics = {'link_acc': 0, 'object_acc': 0, 'device_acc': 0}
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Evaluating'):
                batch = batch.to(self.device)
                
                # 順伝播
                outputs = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                
                # リンク予測精度
                pos_edge_index = batch.edge_index
                neg_edge_index = negative_sampling(
                    edge_index=pos_edge_index,
                    num_nodes=batch.x.size(0),
                    num_neg_samples=pos_edge_index.size(1)
                )
                
                pos_pred = self.model.predict_link(outputs['node_embeddings'], pos_edge_index)
                neg_pred = self.model.predict_link(outputs['node_embeddings'], neg_edge_index)
                
                link_acc = ((pos_pred > 0.5).float().mean() + 
                           (neg_pred < 0.5).float().mean()) / 2
                metrics['link_acc'] += link_acc.item()
                
                # オブジェクトタイプ分類精度
                if hasattr(batch, 'y_object'):
                    object_pred = outputs['object_logits'].argmax(dim=1)
                    object_acc = (object_pred == batch.y_object).float().mean()
                    metrics['object_acc'] += object_acc.item()
                
                # デバイスタイプ分類精度
                if hasattr(batch, 'y_device'):
                    device_pred = outputs['device_logits'].argmax(dim=1)
                    device_acc = (device_pred == batch.y_device).float().mean()
                    metrics['device_acc'] += device_acc.item()
                
                num_batches += 1
                
        # 平均メトリクスを返す
        return {key: value / num_batches for key, value in metrics.items()}
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 100, save_path: str = 'models/max_patch_gnn.pt'):
        """
        モデルを学習
        
        Args:
            train_loader: 学習データローダー
            val_loader: 検証データローダー
            num_epochs: エポック数
            save_path: モデル保存パス
        """
        best_val_loss = float('inf')
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(num_epochs):
            # 学習
            train_losses = self.train_epoch(train_loader)
            
            # 評価
            val_metrics = self.evaluate(val_loader)
            
            # 学習率調整
            val_loss = train_losses['total']  # 簡易的に学習損失を使用
            self.scheduler.step(val_loss)
            
            # ログ出力
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            logger.info(f"Train losses: {train_losses}")
            logger.info(f"Val metrics: {val_metrics}")
            logger.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
            
            # ベストモデルを保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics
                }, save_path)
                logger.info(f"Best model saved! Val loss: {val_loss:.4f}")


def load_dataset(dataset_path: str) -> Tuple[List[Data], Dict]:
    """
    保存されたデータセットを読み込む
    
    Args:
        dataset_path: データセットファイルのパス
        
    Returns:
        (graphs, metadata)のタプル
    """
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
        
    return data['graphs'], {
        'object_type_encoder': data['object_type_encoder'],
        'maxclass_encoder': data['maxclass_encoder'],
        'device_type_encoder': data['device_type_encoder'],
        'position_scaler': data['position_scaler']
    }


def prepare_dataloaders(graphs: List[Data], batch_size: int = 32, 
                       train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """
    データローダーを準備
    
    Args:
        graphs: グラフデータのリスト
        batch_size: バッチサイズ
        train_ratio: 学習データの割合
        
    Returns:
        (train_loader, val_loader)のタプル
    """
    # ラベルを追加（仮実装）
    for graph in graphs:
        # デバイスタイプラベル（簡易版）
        device_type_map = {
            'audio_effect': 0, 'midi_effect': 1, 'instrument': 2,
            'max_for_live': 3, 'midi_instrument': 4, 'unknown': 5
        }
        graph.y_device = torch.tensor([device_type_map.get(graph.device_type, 5)])
    
    # 学習/検証に分割
    train_graphs, val_graphs = train_test_split(graphs, train_size=train_ratio, random_state=42)
    
    # データローダー作成
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Max/MSPパッチGNNモデルの学習')
    parser.add_argument('--dataset', type=str, default='data/graphs/max_patch_graphs.pkl',
                       help='データセットファイル')
    parser.add_argument('--model-type', type=str, default='GraphSAGE',
                       choices=['GCN', 'GAT', 'GraphSAGE'],
                       help='GNNモデルタイプ')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='隠れ層の次元')
    parser.add_argument('--num-layers', type=int, default=3,
                       help='GNNレイヤー数')
    parser.add_argument('--epochs', type=int, default=100,
                       help='学習エポック数')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='バッチサイズ')
    parser.add_argument('--device', type=str, default='cuda',
                       help='cuda or cpu')
    parser.add_argument('--save-path', type=str, default='models/max_patch_gnn.pt',
                       help='モデル保存パス')
    
    args = parser.parse_args()
    
    # データセット読み込み
    logger.info(f"データセットを読み込み中: {args.dataset}")
    graphs, metadata = load_dataset(args.dataset)
    logger.info(f"グラフ数: {len(graphs)}")
    
    # データローダー準備
    train_loader, val_loader = prepare_dataloaders(graphs, batch_size=args.batch_size)
    
    # モデル作成
    sample_graph = graphs[0]
    node_feature_dim = sample_graph.x.shape[1]
    edge_feature_dim = sample_graph.edge_attr.shape[1] if sample_graph.edge_attr.numel() > 0 else 5
    
    model = MaxPatchGNN(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_object_types=len(metadata['object_type_encoder'].classes_),
        num_device_types=6,  # 仮の値
        gnn_type=args.model_type,
        dropout=0.2
    )
    
    logger.info(f"モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    # トレーナー作成
    trainer = MaxPatchGNNTrainer(model, device=args.device)
    
    # 学習実行
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_path=args.save_path
    )
    
    logger.info("学習完了！")


if __name__ == "__main__":
    main()