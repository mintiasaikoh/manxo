#!/usr/bin/env python3
"""
大規模データセット用のGNN学習スクリプト
メモリ効率とGPU利用を最適化
"""

import sys
sys.path.append('/Users/mymac/manxo')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphSAGE, global_mean_pool, global_max_pool
from torch_geometric.data import DataLoader
from torch_geometric.utils import negative_sampling
import numpy as np
from pathlib import Path
import pickle
import json
import logging
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score, f1_score
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EfficientMaxPatchGNN(nn.Module):
    """メモリ効率的なGNNモデル"""
    
    def __init__(self, node_feature_dim, hidden_dim=128, num_layers=2, 
                 num_object_types=4292, dropout=0.3):
        super().__init__()
        
        # より小さな隠れ層サイズで効率化
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # GCNレイヤー（メモリ効率的）
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            self.convs.append(GCNConv(in_dim, hidden_dim))
        
        self.dropout = dropout
        
        # タスクヘッド
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.device_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4)  # 4 device types
        )
        
        self.object_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_object_types)
        )
    
    def encode(self, x, edge_index):
        x = self.node_encoder(x)
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def forward(self, x, edge_index, batch=None, task='link'):
        z = self.encode(x, edge_index)
        
        if task == 'link':
            return z  # リンク予測用の埋め込み
        elif task == 'device':
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            # グラフレベル表現
            z_mean = global_mean_pool(z, batch)
            z_max = global_max_pool(z, batch)
            z_graph = torch.cat([z_mean, z_max], dim=1)
            return self.device_classifier(z_graph)
        elif task == 'object':
            return self.object_predictor(z)
        
        return z


def train_epoch(model, loader, optimizer, device, gradient_accumulation_steps=4):
    """勾配蓄積を使用した効率的な学習"""
    model.train()
    total_loss = 0
    total_examples = 0
    
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(loader, desc='Training')):
        batch = batch.to(device)
        
        # リンク予測
        z = model(batch.x, batch.edge_index, batch.batch, task='link')
        
        # 正例
        pos_edge_index = batch.edge_index
        
        # 負例生成
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=z.size(0),
            num_neg_samples=pos_edge_index.size(1)
        )
        
        # エッジ埋め込み
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        src, dst = edge_index
        edge_embeddings = torch.cat([z[src], z[dst]], dim=-1)
        
        # 予測
        link_pred = model.link_predictor(edge_embeddings).squeeze()
        
        # ラベル
        link_labels = torch.cat([
            torch.ones(pos_edge_index.size(1)),
            torch.zeros(neg_edge_index.size(1))
        ]).to(device)
        
        # 損失
        link_loss = F.binary_cross_entropy_with_logits(link_pred, link_labels)
        
        # デバイス分類
        device_pred = model(batch.x, batch.edge_index, batch.batch, task='device')
        device_loss = F.cross_entropy(device_pred, batch.y)
        
        # 合計損失
        loss = link_loss + 0.5 * device_loss
        loss = loss / gradient_accumulation_steps
        
        loss.backward()
        
        if (i + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        total_examples += 1
        
        # メモリクリア
        if i % 50 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
    
    return total_loss / total_examples


def evaluate(model, loader, device):
    """評価"""
    model.eval()
    link_preds = []
    link_labels = []
    device_preds = []
    device_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            batch = batch.to(device)
            
            # リンク予測評価
            z = model(batch.x, batch.edge_index, batch.batch, task='link')
            pos_edge_index = batch.edge_index
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=z.size(0),
                num_neg_samples=pos_edge_index.size(1) // 2  # 評価時は少なめ
            )
            
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            src, dst = edge_index
            edge_embeddings = torch.cat([z[src], z[dst]], dim=-1)
            
            link_pred = model.link_predictor(edge_embeddings).squeeze()
            link_label = torch.cat([
                torch.ones(pos_edge_index.size(1)),
                torch.zeros(neg_edge_index.size(1))
            ]).to(device)
            
            link_preds.extend((link_pred > 0).cpu().numpy())
            link_labels.extend(link_label.cpu().numpy())
            
            # デバイス分類評価
            device_pred = model(batch.x, batch.edge_index, batch.batch, task='device')
            device_preds.extend(device_pred.argmax(dim=1).cpu().numpy())
            device_labels.extend(batch.y.cpu().numpy())
    
    link_acc = accuracy_score(link_labels, link_preds)
    device_acc = accuracy_score(device_labels, device_preds)
    device_f1 = f1_score(device_labels, device_preds, average='macro')
    
    return {
        'link_acc': link_acc,
        'device_acc': device_acc,
        'device_f1': device_f1
    }


def main():
    parser = argparse.ArgumentParser(description='大規模データセットでGNN学習')
    parser.add_argument('--dataset', type=str, default='data/graph_dataset_full/max_patch_graphs.pkl',
                       help='データセットパス')
    parser.add_argument('--epochs', type=int, default=30, help='エポック数')
    parser.add_argument('--batch-size', type=int, default=64, help='バッチサイズ')
    parser.add_argument('--lr', type=float, default=0.001, help='学習率')
    parser.add_argument('--hidden-dim', type=int, default=128, help='隠れ層次元')
    parser.add_argument('--num-layers', type=int, default=2, help='GNN層数')
    parser.add_argument('--dropout', type=float, default=0.3, help='ドロップアウト率')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    logger.info(f"設定: {args}")
    
    # データセット読み込み
    logger.info(f"データセット読み込み中: {args.dataset}")
    with open(args.dataset, 'rb') as f:
        data = pickle.load(f)
    
    graphs = data['graphs']
    logger.info(f"グラフ数: {len(graphs)}")
    
    # データ分割
    n_graphs = len(graphs)
    n_train = int(0.8 * n_graphs)
    n_val = int(0.1 * n_graphs)
    
    train_graphs = graphs[:n_train]
    val_graphs = graphs[n_train:n_train + n_val]
    test_graphs = graphs[n_train + n_val:]
    
    logger.info(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    
    # データローダー
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size * 2)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size * 2)
    
    # モデル初期化
    sample_graph = train_graphs[0]
    node_feature_dim = sample_graph.x.shape[1]
    
    model = EfficientMaxPatchGNN(
        node_feature_dim=node_feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(args.device)
    
    logger.info(f"モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 最適化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # 学習ループ
    best_val_acc = 0
    best_model_path = Path('models/max_patch_gnn_large.pt')
    best_model_path.parent.mkdir(exist_ok=True)
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 学習
        train_loss = train_epoch(model, train_loader, optimizer, args.device)
        logger.info(f"Train Loss: {train_loss:.4f}")
        
        # 評価
        val_metrics = evaluate(model, val_loader, args.device)
        logger.info(f"Val Metrics: {val_metrics}")
        
        # スケジューラー更新
        scheduler.step(val_metrics['device_acc'])
        
        # ベストモデル保存
        if val_metrics['device_acc'] > best_val_acc:
            best_val_acc = val_metrics['device_acc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'args': args
            }, best_model_path)
            logger.info(f"ベストモデル保存! Device Acc: {best_val_acc:.4f}")
    
    # テスト評価
    logger.info("\n=== テスト評価 ===")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, args.device)
    logger.info(f"Test Metrics: {test_metrics}")
    
    # 最終モデル保存
    torch.save(model, Path('models/max_patch_gnn_large_final.pt'))
    logger.info("学習完了！")


if __name__ == "__main__":
    main()