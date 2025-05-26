#!/usr/bin/env python3
"""
最適化されたグラフデータセット作成スクリプト
全データセット（11,894ファイル）を効率的に処理
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import argparse
from datetime import datetime
import pickle
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.db_connector import DatabaseConnector

class OptimizedMaxPatchGraphDataset:
    def __init__(self, db_config_path: str):
        self.db = DatabaseConnector(db_config_path)
        self.db.connect()
        
        # オブジェクトタイプをエンコード
        self.object_type_encoder = self._create_object_type_encoder()
        
        # 属性マッピング
        self.attribute_columns = [
            'parameter_enable', 'parameter_mappable', 'saved_attribute_attributes',
            'size', 'style', 'fontsize', 'minimum', 'maximum'
        ]
        
    def _create_object_type_encoder(self) -> Dict[str, int]:
        """オブジェクトタイプのワンホットエンコーダーを作成"""
        print("Creating object type encoder...")
        result = self.db.execute_query("""
            SELECT DISTINCT object_type 
            FROM object_details 
            WHERE object_type IS NOT NULL
            ORDER BY object_type
        """)
        
        types = [row['object_type'] for row in result]
        encoder = {t: i for i, t in enumerate(types)}
        encoder['unknown'] = len(types)  # Unknown type
        
        print(f"Found {len(encoder)} unique object types")
        return encoder
        
    def process_single_file(self, file_path: str) -> Optional[Data]:
        """単一ファイルを処理"""
        try:
            # オブジェクト情報を取得
            objects_query = """
                SELECT 
                    object_id, maxclass, object_type, text_content,
                    position, inlet_types, outlet_types, saved_attributes
                FROM object_details
                WHERE patch_file = %s
                ORDER BY object_id
            """
            objects_result = self.db.execute_query(objects_query, (file_path,))
            
            if not objects_result:
                return None
                
            objects_df = pd.DataFrame(objects_result)
            
            # 接続情報を取得
            connections_query = """
                SELECT 
                    source_object_type, source_port, target_object_type, target_port,
                    source_value, target_value, source_outlet_types, target_inlet_types,
                    source_position, target_position, connection_order, hierarchy_depth
                FROM object_connections
                WHERE patch_file = %s
                ORDER BY connection_order
            """
            connections_result = self.db.execute_query(connections_query, (file_path,))
            
            if not connections_result:
                return None
                
            connections_df = pd.DataFrame(connections_result)
            
            # ノード特徴量を抽出
            node_features = self._extract_node_features(objects_df)
            
            # エッジ情報を抽出
            edge_index, edge_attr = self._extract_edge_features(connections_df, objects_df)
            
            # グラフデータを作成
            graph = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=len(objects_df),
                file_path=file_path
            )
            
            return graph
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None
            
    def _extract_node_features(self, objects_df: pd.DataFrame) -> torch.Tensor:
        """ノード特徴量を抽出（メモリ効率的）"""
        features = []
        
        for _, obj in objects_df.iterrows():
            feat = []
            
            # 1. オブジェクトタイプ（ワンホットエンコーディング）
            obj_type = obj['object_type'] or 'unknown'
            type_idx = self.object_type_encoder.get(obj_type, self.object_type_encoder['unknown'])
            type_onehot = np.zeros(len(self.object_type_encoder), dtype=np.float32)
            type_onehot[type_idx] = 1.0
            feat.extend(type_onehot)
            
            # 2. 位置情報（正規化）
            pos = obj['position'] or [0, 0, 100, 22]
            normalized_pos = [
                pos[0] / 1000.0,  # x
                pos[1] / 1000.0,  # y
                pos[2] / 200.0,   # width
                pos[3] / 100.0    # height
            ]
            feat.extend(normalized_pos)
            
            # 3. ポート数
            inlet_count = len(obj['inlet_types']) if obj['inlet_types'] else 0
            outlet_count = len(obj['outlet_types']) if obj['outlet_types'] else 0
            feat.extend([inlet_count / 10.0, outlet_count / 10.0])
            
            # 4. テキストコンテンツの長さ（正規化）
            text_len = len(obj['text_content']) if obj['text_content'] else 0
            feat.append(min(text_len / 100.0, 1.0))
            
            # 5. maxclassタイプ（カテゴリカル）
            maxclass_types = ['newobj', 'message', 'flonum', 'number', 'toggle', 
                             'button', 'comment', 'inlet', 'outlet']
            maxclass_feat = [1.0 if obj['maxclass'] == t else 0.0 for t in maxclass_types]
            feat.extend(maxclass_feat)
            
            features.append(feat)
            
        return torch.tensor(features, dtype=torch.float32)
        
    def _extract_edge_features(self, connections_df: pd.DataFrame, 
                              objects_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """エッジ特徴量を抽出（メモリ効率的）"""
        edge_list = []
        edge_features = []
        
        # オブジェクトインデックスマップを作成
        obj_id_to_idx = {obj_id: idx for idx, obj_id in enumerate(objects_df['object_id'])}
        
        for _, conn in connections_df.iterrows():
            # ソースとターゲットのマッチング（簡易版）
            source_matches = objects_df[objects_df['object_type'] == conn['source_object_type']]
            target_matches = objects_df[objects_df['object_type'] == conn['target_object_type']]
            
            if len(source_matches) == 0 or len(target_matches) == 0:
                continue
                
            # 位置情報を使った簡易マッチング
            source_idx = source_matches.index[0]
            target_idx = target_matches.index[0]
            
            edge_list.append([source_idx, target_idx])
            
            # エッジ特徴量（5次元）
            edge_feat = [
                conn['source_port'],
                conn['target_port'],
                conn['connection_order'] if conn['connection_order'] else 0,
                conn['hierarchy_depth'],
                1.0  # 互換性スコア（仮）
            ]
            edge_features.append(edge_feat)
            
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 5), dtype=torch.float32)
            
        return edge_index, edge_attr
        
    def create_dataset(self, output_dir: str, batch_size: int = 100, num_workers: int = 4):
        """全データセットを作成（バッチ処理）"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 処理対象のパッチファイルを取得
        print("Loading patch files from database...")
        patches_query = """
            SELECT DISTINCT patch_file as file_path
            FROM object_connections
            ORDER BY patch_file
        """
        patches_result = self.db.execute_query(patches_query)
        file_paths = [row['file_path'] for row in patches_result]
        
        total_files = len(file_paths)
        print(f"Found {total_files} patch files to process")
        
        # プログレストラッキング用ファイル
        progress_file = os.path.join(output_dir, 'progress.json')
        processed_files = set()
        
        # 既存の進捗を読み込み
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                processed_files = set(progress_data.get('processed_files', []))
            print(f"Resuming from {len(processed_files)} already processed files")
        
        # 未処理ファイルのみ
        remaining_files = [f for f in file_paths if f not in processed_files]
        print(f"Processing {len(remaining_files)} remaining files...")
        
        # バッチ処理
        batch_idx = 0
        start_time = datetime.now()
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i in range(0, len(remaining_files), batch_size):
                batch_files = remaining_files[i:i+batch_size]
                batch_graphs = []
                
                # 並列処理
                future_to_file = {
                    executor.submit(self.process_single_file, f): f 
                    for f in batch_files
                }
                
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        graph = future.result()
                        if graph is not None:
                            batch_graphs.append(graph)
                            processed_files.add(file_path)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                
                # バッチを保存
                if batch_graphs:
                    batch_file = os.path.join(output_dir, f'batch_{batch_idx:04d}.pt')
                    torch.save(batch_graphs, batch_file)
                    print(f"Saved batch {batch_idx} with {len(batch_graphs)} graphs")
                    batch_idx += 1
                
                # 進捗を保存
                progress_data = {
                    'processed_files': list(processed_files),
                    'total_files': total_files,
                    'last_update': datetime.now().isoformat()
                }
                with open(progress_file, 'w') as f:
                    json.dump(progress_data, f)
                
                # 進捗表示
                elapsed = (datetime.now() - start_time).total_seconds()
                processed_count = len(processed_files)
                rate = processed_count / elapsed if elapsed > 0 else 0
                eta = (total_files - processed_count) / rate if rate > 0 else 0
                
                print(f"\nProgress: {processed_count}/{total_files} files "
                      f"({processed_count/total_files*100:.1f}%)")
                print(f"Rate: {rate:.1f} files/second")
                print(f"ETA: {eta/60:.1f} minutes")
                
                # メモリクリーンアップ
                gc.collect()
        
        print(f"\nDataset creation completed!")
        print(f"Total time: {(datetime.now() - start_time).total_seconds()/60:.1f} minutes")
        print(f"Output directory: {output_dir}")
        
        # メタデータを保存
        metadata = {
            'total_files': total_files,
            'processed_files': len(processed_files),
            'num_batches': batch_idx,
            'object_types': list(self.object_type_encoder.keys()),
            'node_feature_dim': len(self.object_type_encoder) + 16,
            'edge_feature_dim': 5,
            'creation_date': datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        self.db.disconnect()

def main():
    parser = argparse.ArgumentParser(description='Create optimized Max patch graph dataset')
    parser.add_argument('--output', type=str, default='data/graph_dataset_optimized',
                      help='Output directory for dataset')
    parser.add_argument('--batch-size', type=int, default=100,
                      help='Batch size for processing')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count(),
                      help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # データセット作成
    dataset = OptimizedMaxPatchGraphDataset('scripts/db_settings.ini')
    dataset.create_dataset(args.output, args.batch_size, args.num_workers)

if __name__ == "__main__":
    main()