#!/usr/bin/env python3
"""
全データセットでGNNモデルをトレーニング
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from datetime import datetime
import argparse
from tqdm import tqdm
import glob

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MaxPatchGNN(nn.Module):
    """Max/MSPパッチ用のGraph Neural Network"""
    
    def __init__(self, node_feature_dim: int, hidden_dim: int = 256, 
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # 入力層
        self.convs.append(SAGEConv(node_feature_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # 隠れ層
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # 出力層（リンク予測用）
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = dropout
        
        # リンク予測用のMLP
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, edge_index, batch=None):
        # GNN層を通す
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.bns[:-1])):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 最終層
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        
        return x
    
    def predict_link(self, x, edge_index, src, dst, batch=None):
        """リンク予測"""
        node_embeddings = self.forward(x, edge_index, batch)
        
        src_embeddings = node_embeddings[src]
        dst_embeddings = node_embeddings[dst]
        
        # ソースとターゲットの埋め込みを連結
        edge_embeddings = torch.cat([src_embeddings, dst_embeddings], dim=-1)
        
        # リンク存在確率を予測
        return self.link_predictor(edge_embeddings).squeeze()

def load_dataset(dataset_dir: str, max_batches: int = None):
    """データセットを読み込み"""
    print(f"Loading dataset from {dataset_dir}")
    
    # メタデータを読み込み
    with open(os.path.join(dataset_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    print(f"Dataset info:")
    print(f"- Total files: {metadata['total_files']}")
    print(f"- Processed files: {metadata['processed_files']}")
    print(f"- Node feature dim: {metadata['node_feature_dim']}")
    print(f"- Number of batches: {metadata['num_batches']}")
    
    # バッチファイルを読み込み
    all_graphs = []
    batch_files = sorted(glob.glob(os.path.join(dataset_dir, 'batch_*.pt')))
    
    if max_batches:
        batch_files = batch_files[:max_batches]
    
    for batch_file in tqdm(batch_files, desc="Loading batches"):
        graphs = torch.load(batch_file, weights_only=False)
        all_graphs.extend(graphs)
    
    print(f"Loaded {len(all_graphs)} graphs")
    return all_graphs, metadata

def create_link_prediction_data(graph):
    """リンク予測用のデータを作成"""
    # 既存のエッジを正例として使用
    num_edges = graph.edge_index.size(1)
    num_nodes = graph.x.size(0)
    
    if num_edges == 0 or num_nodes < 2:
        return None
    
    # 正例（既存のエッジ）
    pos_edge_index = graph.edge_index
    
    # 負例（存在しないエッジ）をサンプリング
    neg_edge_index = []
    existing_edges = set()
    
    for i in range(num_edges):
        src, dst = pos_edge_index[0, i].item(), pos_edge_index[1, i].item()
        existing_edges.add((src, dst))
    
    # 同数の負例を生成
    while len(neg_edge_index) < num_edges:
        src = torch.randint(0, num_nodes, (1,)).item()
        dst = torch.randint(0, num_nodes, (1,)).item()
        
        if src != dst and (src, dst) not in existing_edges:
            neg_edge_index.append([src, dst])
    
    neg_edge_index = torch.tensor(neg_edge_index).t()
    
    return pos_edge_index, neg_edge_index

def train_epoch(model, graphs, optimizer, device):
    """1エポックの学習"""
    model.train()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    for graph in tqdm(graphs, desc="Training"):
        # リンク予測データを作成
        data = create_link_prediction_data(graph)
        if data is None:
            continue
            
        pos_edge_index, neg_edge_index = data
        
        # デバイスに転送
        graph = graph.to(device)
        pos_edge_index = pos_edge_index.to(device)
        neg_edge_index = neg_edge_index.to(device)
        
        optimizer.zero_grad()
        
        # 正例の予測
        pos_pred = model.predict_link(
            graph.x, graph.edge_index,
            pos_edge_index[0], pos_edge_index[1]
        )
        
        # 負例の予測
        neg_pred = model.predict_link(
            graph.x, graph.edge_index,
            neg_edge_index[0], neg_edge_index[1]
        )
        
        # テンソルの次元を確認
        if pos_pred.dim() == 0:
            pos_pred = pos_pred.unsqueeze(0)
        if neg_pred.dim() == 0:
            neg_pred = neg_pred.unsqueeze(0)
        
        # ラベル
        pos_label = torch.ones_like(pos_pred)
        neg_label = torch.zeros_like(neg_pred)
        
        # 損失計算
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_pred, neg_pred]),
            torch.cat([pos_label, neg_label])
        )
        
        loss.backward()
        optimizer.step()
        
        # 精度計算
        pred = torch.cat([pos_pred, neg_pred])
        label = torch.cat([pos_label, neg_label])
        acc = ((pred > 0) == label).float().mean()
        
        total_loss += loss.item()
        total_acc += acc.item()
        num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches

def evaluate(model, graphs, device):
    """評価"""
    model.eval()
    total_acc = 0
    num_batches = 0
    
    with torch.no_grad():
        for graph in tqdm(graphs, desc="Evaluating"):
            data = create_link_prediction_data(graph)
            if data is None:
                continue
                
            pos_edge_index, neg_edge_index = data
            
            graph = graph.to(device)
            pos_edge_index = pos_edge_index.to(device)
            neg_edge_index = neg_edge_index.to(device)
            
            # 予測
            pos_pred = model.predict_link(
                graph.x, graph.edge_index,
                pos_edge_index[0], pos_edge_index[1]
            )
            neg_pred = model.predict_link(
                graph.x, graph.edge_index,
                neg_edge_index[0], neg_edge_index[1]
            )
            
            # 精度計算
            pos_acc = (pos_pred > 0).float().mean()
            neg_acc = (neg_pred <= 0).float().mean()
            acc = (pos_acc + neg_acc) / 2
            
            total_acc += acc.item()
            num_batches += 1
    
    return total_acc / num_batches

def main():
    parser = argparse.ArgumentParser(description='Train GNN on Max patch dataset')
    parser.add_argument('--dataset', type=str, default='data/graph_dataset_full',
                      help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=256,
                      help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=3,
                      help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate')
    parser.add_argument('--max-batches', type=int, default=10,
                      help='Maximum number of batch files to load (for testing)')
    # MacのMPSをサポート
    if torch.backends.mps.is_available():
        default_device = 'mps'
    elif torch.cuda.is_available():
        default_device = 'cuda'
    else:
        default_device = 'cpu'
    
    parser.add_argument('--device', type=str, default=default_device,
                      help='Device to use (mps/cuda/cpu)')
    
    args = parser.parse_args()
    
    # データセットを読み込み
    graphs, metadata = load_dataset(args.dataset, args.max_batches)
    
    # 訓練・検証データに分割
    train_graphs, val_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")
    
    # モデルを作成
    model = MaxPatchGNN(
        node_feature_dim=metadata['node_feature_dim'],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # オプティマイザ
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 学習ループ
    best_val_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 訓練
        train_loss, train_acc = train_epoch(model, train_graphs, optimizer, args.device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # 検証
        val_acc = evaluate(model, val_graphs, args.device)
        print(f"Val Acc: {val_acc:.4f}")
        
        # ベストモデルを保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/max_patch_gnn_full.pt')
            print(f"Saved best model (acc: {best_val_acc:.4f})")
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()