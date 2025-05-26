#!/usr/bin/env python3
"""
シンプルなグラフデータセット作成スクリプト
全データセット（11,894ファイル）を順次処理
"""

import os
import sys
import json
import torch
from torch_geometric.data import Data
from typing import Dict, Optional, Tuple
import argparse
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.db_connector import DatabaseConnector

class SimpleMaxPatchGraphDataset:
    def __init__(self, db_config_path: str):
        self.db_config_path = db_config_path
        self.object_type_encoder = None
        
    def _connect_db(self):
        """データベースに接続"""
        db = DatabaseConnector(self.db_config_path)
        db.connect()
        return db
        
    def _create_object_type_encoder(self, db) -> Dict[str, int]:
        """オブジェクトタイプのエンコーダーを作成"""
        print("Creating object type encoder...")
        result = db.execute_query("""
            SELECT DISTINCT object_type 
            FROM object_details 
            WHERE object_type IS NOT NULL
            ORDER BY object_type
        """)
        
        types = [row['object_type'] for row in result]
        encoder = {t: i for i, t in enumerate(types)}
        encoder['unknown'] = len(types)
        
        print(f"Found {len(encoder)} unique object types")
        return encoder
        
    def process_file(self, db, file_path: str) -> Optional[Data]:
        """単一ファイルを処理"""
        try:
            # オブジェクト情報を取得
            objects_result = db.execute_query("""
                SELECT object_id, object_type, position, inlet_types, outlet_types
                FROM object_details
                WHERE patch_file = %s
                ORDER BY object_id
            """, (file_path,))
            
            if not objects_result:
                return None
                
            # 接続情報を取得
            connections_result = db.execute_query("""
                SELECT source_object_type, source_port, target_object_type, target_port
                FROM object_connections
                WHERE patch_file = %s
            """, (file_path,))
            
            if not connections_result:
                return None
                
            # ノード特徴量を作成
            node_features = []
            obj_type_to_idx = {}
            
            for idx, obj in enumerate(objects_result):
                # オブジェクトタイプのワンホットエンコーディング
                obj_type = obj['object_type'] or 'unknown'
                type_idx = self.object_type_encoder.get(obj_type, self.object_type_encoder['unknown'])
                type_onehot = [0.0] * len(self.object_type_encoder)
                type_onehot[type_idx] = 1.0
                
                # 位置情報
                pos = obj['position'] or [0, 0, 100, 22]
                normalized_pos = [
                    pos[0] / 1000.0,
                    pos[1] / 1000.0,
                    pos[2] / 200.0,
                    pos[3] / 100.0
                ]
                
                # ポート数
                inlet_count = len(obj['inlet_types']) if obj['inlet_types'] else 0
                outlet_count = len(obj['outlet_types']) if obj['outlet_types'] else 0
                
                # 特徴量を結合
                features = type_onehot + normalized_pos + [inlet_count / 10.0, outlet_count / 10.0]
                node_features.append(features)
                
                # タイプからインデックスへのマップ
                if obj_type not in obj_type_to_idx:
                    obj_type_to_idx[obj_type] = []
                obj_type_to_idx[obj_type].append(idx)
            
            # エッジリストを作成
            edge_list = []
            for conn in connections_result:
                source_type = conn['source_object_type']
                target_type = conn['target_object_type']
                
                # マッチするオブジェクトを探す
                if source_type in obj_type_to_idx and target_type in obj_type_to_idx:
                    # 簡易的に最初のマッチを使用
                    source_idx = obj_type_to_idx[source_type][0]
                    target_idx = obj_type_to_idx[target_type][0]
                    edge_list.append([source_idx, target_idx])
            
            if not edge_list:
                return None
                
            # PyTorchテンソルに変換
            x = torch.tensor(node_features, dtype=torch.float32)
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            
            return Data(x=x, edge_index=edge_index, num_nodes=len(objects_result))
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
            
    def create_dataset(self, output_dir: str, batch_size: int = 100):
        """全データセットを作成"""
        os.makedirs(output_dir, exist_ok=True)
        
        # データベース接続
        db = self._connect_db()
        
        # エンコーダーを作成
        self.object_type_encoder = self._create_object_type_encoder(db)
        
        # 処理対象ファイルを取得
        print("Loading patch files...")
        result = db.execute_query("""
            SELECT DISTINCT patch_file 
            FROM object_connections
            ORDER BY patch_file
        """)
        file_paths = [row['patch_file'] for row in result]
        
        total_files = len(file_paths)
        print(f"Found {total_files} files to process")
        
        # 進捗管理
        progress_file = os.path.join(output_dir, 'progress.json')
        processed_files = set()
        
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                processed_files = set(progress_data.get('processed_files', []))
            print(f"Resuming from {len(processed_files)} processed files")
        
        # 未処理ファイル
        remaining_files = [f for f in file_paths if f not in processed_files]
        
        # バッチ処理
        graphs = []
        batch_idx = len(processed_files) // batch_size
        start_time = datetime.now()
        
        for file_path in tqdm(remaining_files, desc="Processing files"):
            graph = self.process_file(db, file_path)
            
            if graph is not None:
                graphs.append(graph)
                processed_files.add(file_path)
                
                # バッチ保存
                if len(graphs) >= batch_size:
                    batch_file = os.path.join(output_dir, f'batch_{batch_idx:04d}.pt')
                    torch.save(graphs, batch_file)
                    print(f"\nSaved batch {batch_idx} with {len(graphs)} graphs")
                    graphs = []
                    batch_idx += 1
                    
                    # 進捗保存
                    with open(progress_file, 'w') as f:
                        json.dump({
                            'processed_files': list(processed_files),
                            'total_files': total_files,
                            'last_update': datetime.now().isoformat()
                        }, f)
                    
                    # 統計表示
                    elapsed = (datetime.now() - start_time).total_seconds()
                    processed_count = len(processed_files)
                    rate = processed_count / elapsed
                    eta = (total_files - processed_count) / rate
                    
                    print(f"Progress: {processed_count}/{total_files} ({processed_count/total_files*100:.1f}%)")
                    print(f"Rate: {rate:.1f} files/sec, ETA: {eta/60:.1f} minutes")
        
        # 残りのグラフを保存
        if graphs:
            batch_file = os.path.join(output_dir, f'batch_{batch_idx:04d}.pt')
            torch.save(graphs, batch_file)
            print(f"Saved final batch with {len(graphs)} graphs")
        
        # メタデータ保存
        metadata = {
            'total_files': total_files,
            'processed_files': len(processed_files),
            'num_batches': batch_idx + 1,
            'node_feature_dim': len(self.object_type_encoder) + 6,
            'object_types': list(self.object_type_encoder.keys()),
            'creation_date': datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        db.disconnect()
        
        print(f"\nCompleted! Processed {len(processed_files)} files")
        print(f"Total time: {(datetime.now() - start_time).total_seconds()/60:.1f} minutes")

def main():
    parser = argparse.ArgumentParser(description='Create Max patch graph dataset')
    parser.add_argument('--output', type=str, default='data/graph_dataset_full',
                      help='Output directory')
    parser.add_argument('--batch-size', type=int, default=100,
                      help='Graphs per batch file')
    
    args = parser.parse_args()
    
    dataset = SimpleMaxPatchGraphDataset('scripts/db_settings.ini')
    dataset.create_dataset(args.output, args.batch_size)

if __name__ == "__main__":
    main()