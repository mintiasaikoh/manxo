#!/usr/bin/env python3
"""
メモリ効率的な全データセットでのGNNモデルトレーニング
バッチファイルを逐次読み込みして処理
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
import numpy as np
from datetime import datetime
import argparse
from tqdm import tqdm
import glob
import random

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

def load_dataset_info(dataset_dir: str):
    """データセットのメタデータのみを読み込み"""
    print(f"Loading dataset info from {dataset_dir}")
    
    # メタデータを読み込み
    with open(os.path.join(dataset_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    print(f"Dataset info:")
    print(f"- Total files: {metadata['total_files']}")
    print(f"- Processed files: {metadata['processed_files']}")
    print(f"- Node feature dim: {metadata['node_feature_dim']}")
    print(f"- Number of batches: {metadata['num_batches']}")
    
    # バッチファイルのリストを取得
    batch_files = sorted(glob.glob(os.path.join(dataset_dir, 'batch_*.pt')))
    
    return metadata, batch_files

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

def process_batch_file(model, batch_file, optimizer, device, is_training=True):
    """単一のバッチファイルを処理"""
    # バッチファイルを読み込み
    graphs = torch.load(batch_file, weights_only=False)
    
    total_loss = 0
    total_acc = 0
    num_graphs = 0
    
    if is_training:
        model.train()
    else:
        model.eval()
    
    # グラフごとに処理
    for graph in graphs:
        # リンク予測データを作成
        data = create_link_prediction_data(graph)
        if data is None:
            continue
            
        pos_edge_index, neg_edge_index = data
        
        # デバイスに転送
        graph = graph.to(device)
        pos_edge_index = pos_edge_index.to(device)
        neg_edge_index = neg_edge_index.to(device)
        
        if is_training:
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
        
        if is_training:
            # 損失計算
            loss = F.binary_cross_entropy_with_logits(
                torch.cat([pos_pred, neg_pred]),
                torch.cat([pos_label, neg_label])
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 精度計算
        with torch.no_grad():
            pred = torch.cat([pos_pred, neg_pred])
            label = torch.cat([pos_label, neg_label])
            acc = ((pred > 0) == label).float().mean()
            total_acc += acc.item()
        
        num_graphs += 1
        
        # メモリ解放
        del graph, pos_edge_index, neg_edge_index
        if device == 'mps' or device == 'cuda':
            torch.mps.empty_cache() if device == 'mps' else torch.cuda.empty_cache()
    
    # バッチファイルのメモリも解放
    del graphs
    
    if num_graphs == 0:
        return 0, 0
    
    return total_loss / num_graphs, total_acc / num_graphs

def train_epoch(model, train_batch_files, optimizer, device):
    """1エポックの学習（バッチファイルを逐次処理）"""
    model.train()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    # バッチファイルをシャッフル
    batch_files_shuffled = train_batch_files.copy()
    random.shuffle(batch_files_shuffled)
    
    for batch_file in tqdm(batch_files_shuffled, desc="Training"):
        loss, acc = process_batch_file(model, batch_file, optimizer, device, is_training=True)
        if loss > 0:  # 有効なグラフがあった場合
            total_loss += loss
            total_acc += acc
            num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches

def evaluate(model, val_batch_files, device):
    """評価（バッチファイルを逐次処理）"""
    model.eval()
    total_acc = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_file in tqdm(val_batch_files, desc="Evaluating"):
            _, acc = process_batch_file(model, batch_file, None, device, is_training=False)
            if acc > 0:  # 有効なグラフがあった場合
                total_acc += acc
                num_batches += 1
    
    return total_acc / num_batches

def main():
    parser = argparse.ArgumentParser(description='Train GNN on Max patch dataset (memory efficient)')
    parser.add_argument('--dataset', type=str, default='data/graph_dataset_full',
                      help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=256,
                      help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=3,
                      help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate')
    parser.add_argument('--val-split', type=float, default=0.2,
                      help='Validation split ratio')
    
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
    
    # MPSのメモリ制限を解除（オプション）
    if args.device == 'mps':
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    
    # データセット情報を読み込み
    metadata, batch_files = load_dataset_info(args.dataset)
    
    # 訓練・検証用にバッチファイルを分割
    num_val = int(len(batch_files) * args.val_split)
    val_batch_files = batch_files[:num_val]
    train_batch_files = batch_files[num_val:]
    
    print(f"Train batches: {len(train_batch_files)}, Val batches: {len(val_batch_files)}")
    
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
    
    # モデル保存ディレクトリを作成
    os.makedirs('models', exist_ok=True)
    
    # 学習ループ
    best_val_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 訓練
        train_loss, train_acc = train_epoch(model, train_batch_files, optimizer, args.device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # 検証
        val_acc = evaluate(model, val_batch_files, args.device)
        print(f"Val Acc: {val_acc:.4f}")
        
        # ベストモデルを保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/max_patch_gnn_memory_efficient.pt')
            print(f"Saved best model (acc: {best_val_acc:.4f})")
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()