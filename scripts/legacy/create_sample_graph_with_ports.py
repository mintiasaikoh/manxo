#!/usr/bin/env python3
"""
サンプルのグラフデータセットを作成（ポートタイプ情報付き）
メモリとディスク容量を考慮して、小さなサンプルから開始
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from db_connector import DatabaseConnector
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SampleGraphCreator:
    def __init__(self, db_config_path: str):
        self.db = DatabaseConnector(db_config_path)
        self.object_type_encoder = LabelEncoder()
        self.maxclass_encoder = LabelEncoder()
        self.port_type_encoder = LabelEncoder()
        self.position_scaler = StandardScaler()
        
    def prepare_encoders(self):
        """エンコーダーを準備"""
        self.db.connect()
        
        try:
            # オブジェクトタイプ
            obj_types = self.db.execute_query(
                "SELECT DISTINCT object_type FROM object_details WHERE object_type IS NOT NULL LIMIT 1000"
            )
            self.object_type_encoder.fit([r['object_type'] for r in obj_types])
            
            # maxclass
            maxclasses = self.db.execute_query(
                "SELECT DISTINCT maxclass FROM object_details WHERE maxclass IS NOT NULL"
            )
            self.maxclass_encoder.fit([r['maxclass'] for r in maxclasses])
            
            # ポートタイプ（全ての配列要素を収集）
            port_types = set()
            inlet_results = self.db.execute_query(
                "SELECT DISTINCT unnest(inlet_types) as port_type FROM object_details WHERE inlet_types IS NOT NULL LIMIT 1000"
            )
            outlet_results = self.db.execute_query(
                "SELECT DISTINCT unnest(outlet_types) as port_type FROM object_details WHERE outlet_types IS NOT NULL LIMIT 1000"
            )
            
            for r in inlet_results + outlet_results:
                if r['port_type']:
                    port_types.add(r['port_type'])
                    
            if port_types:
                self.port_type_encoder.fit(list(port_types))
                logger.info(f"ユニークなポートタイプ数: {len(port_types)}")
            
            # 位置情報
            positions = []
            pos_results = self.db.execute_query(
                "SELECT position FROM object_details WHERE position IS NOT NULL LIMIT 5000"
            )
            for r in pos_results:
                if r['position'] and len(r['position']) >= 4:
                    positions.append(r['position'][:4])
            if positions:
                self.position_scaler.fit(positions)
                
        finally:
            self.db.disconnect()
            
    def create_sample_dataset(self, num_patches: int = 100):
        """サンプルデータセットを作成"""
        self.prepare_encoders()
        
        self.db.connect()
        
        try:
            # サンプルパッチを取得
            patch_files = self.db.execute_query(
                f"SELECT DISTINCT patch_file FROM object_connections LIMIT {num_patches}"
            )
            patch_files = [r['patch_file'] for r in patch_files]
            
            logger.info(f"処理するパッチ数: {len(patch_files)}")
            
            graphs = []
            
            for patch_file in tqdm(patch_files, desc="グラフ作成中"):
                # オブジェクト詳細を取得
                objects = pd.DataFrame(self.db.execute_query(
                    "SELECT * FROM object_details WHERE patch_file = %s",
                    (patch_file,)
                ))
                
                # 接続情報を取得（ポートタイプ情報付き）
                connections = pd.DataFrame(self.db.execute_query(
                    """
                    SELECT *, 
                           source_outlet_types, 
                           target_inlet_types,
                           source_position,
                           target_position
                    FROM object_connections 
                    WHERE patch_file = %s
                    """,
                    (patch_file,)
                ))
                
                if not objects.empty and not connections.empty:
                    graph = self.create_graph_with_ports(patch_file, objects, connections)
                    if graph is not None:
                        graphs.append(graph)
                        
            logger.info(f"作成されたグラフ数: {len(graphs)}")
            
            # 保存
            os.makedirs('data/graphs_sample', exist_ok=True)
            output_path = 'data/graphs_sample/sample_with_ports.pkl'
            
            with open(output_path, 'wb') as f:
                pickle.dump(graphs, f, protocol=4)
                
            logger.info(f"サンプルデータセット保存完了: {output_path}")
            
            # 最初のグラフの詳細を表示
            if graphs:
                g = graphs[0]
                logger.info(f"\n最初のグラフの詳細:")
                logger.info(f"  ノード数: {g.num_nodes}")
                logger.info(f"  エッジ数: {g.edge_index.shape[1]}")
                logger.info(f"  ノード特徴量次元: {g.x.shape[1]}")
                if hasattr(g, 'port_type_features'):
                    logger.info(f"  ポートタイプ特徴量: {g.port_type_features.shape}")
                if hasattr(g, 'edge_port_types'):
                    logger.info(f"  エッジポートタイプ: {len(g.edge_port_types)} pairs")
                    
            return graphs
            
        finally:
            self.db.disconnect()
            
    def create_graph_with_ports(self, patch_file: str, objects_df: pd.DataFrame, connections_df: pd.DataFrame):
        """ポートタイプ情報付きのグラフを作成"""
        try:
            num_objects = len(objects_df)
            
            # 基本特徴量
            num_object_types = len(self.object_type_encoder.classes_)
            num_maxclasses = len(self.maxclass_encoder.classes_)
            feature_dim = num_object_types + num_maxclasses + 4 + 2 + 3
            
            features = np.zeros((num_objects, feature_dim), dtype=np.float32)
            
            # ポートタイプ特徴量用
            max_ports = 10  # 最大ポート数
            num_port_types = len(self.port_type_encoder.classes_) if self.port_type_encoder.classes_.size > 0 else 1
            port_features = np.zeros((num_objects, max_ports * 2, num_port_types), dtype=np.float32)
            
            # object_idからインデックスへのマッピング
            object_id_to_idx = {row['object_id']: idx for idx, row in objects_df.iterrows()}
            
            for idx, (_, obj) in enumerate(objects_df.iterrows()):
                feat_idx = 0
                
                # 基本特徴量（既存のコードと同じ）
                if obj['object_type'] and obj['object_type'] in self.object_type_encoder.classes_:
                    obj_type_idx = self.object_type_encoder.transform([obj['object_type']])[0]
                    features[idx, feat_idx + obj_type_idx] = 1
                feat_idx += num_object_types
                
                if obj['maxclass'] and obj['maxclass'] in self.maxclass_encoder.classes_:
                    maxclass_idx = self.maxclass_encoder.transform([obj['maxclass']])[0]
                    features[idx, feat_idx + maxclass_idx] = 1
                feat_idx += num_maxclasses
                
                # 位置
                position = obj['position'] if obj['position'] else [0, 0, 0, 0]
                if len(position) >= 4:
                    features[idx, feat_idx:feat_idx+4] = self.position_scaler.transform([position[:4]])[0]
                feat_idx += 4
                
                # ポート数
                features[idx, feat_idx] = len(obj['inlet_types']) if obj['inlet_types'] else 0
                features[idx, feat_idx+1] = len(obj['outlet_types']) if obj['outlet_types'] else 0
                feat_idx += 2
                
                # テキスト特徴
                if obj['text_content']:
                    features[idx, feat_idx] = len(obj['text_content'])
                    features[idx, feat_idx+1] = 1 if '$' in obj['text_content'] else 0
                    features[idx, feat_idx+2] = 1 if ' ' in obj['text_content'] else 0
                    
                # ポートタイプ特徴量
                if self.port_type_encoder.classes_.size > 0:
                    # インレット
                    if obj['inlet_types']:
                        for i, port_type in enumerate(obj['inlet_types'][:max_ports]):
                            if port_type in self.port_type_encoder.classes_:
                                port_idx = self.port_type_encoder.transform([port_type])[0]
                                port_features[idx, i, port_idx] = 1
                                
                    # アウトレット
                    if obj['outlet_types']:
                        for i, port_type in enumerate(obj['outlet_types'][:max_ports]):
                            if port_type in self.port_type_encoder.classes_:
                                port_idx = self.port_type_encoder.transform([port_type])[0]
                                port_features[idx, max_ports + i, port_idx] = 1
                                
            # エッジ処理
            edge_list = []
            edge_features = []
            edge_port_types = []
            
            for _, conn in connections_df.iterrows():
                # オブジェクトタイプでマッチング（簡易版）
                source_candidates = objects_df[objects_df['object_type'] == conn['source_object_type']]
                target_candidates = objects_df[objects_df['object_type'] == conn['target_object_type']]
                
                if not source_candidates.empty and not target_candidates.empty:
                    source_idx = source_candidates.index[0]
                    target_idx = target_candidates.index[0]
                    
                    edge_list.append([source_idx, target_idx])
                    
                    # エッジ特徴量
                    edge_feat = [
                        conn['source_port'],
                        conn['target_port'],
                        conn['connection_order'] if conn['connection_order'] else 0,
                        conn['hierarchy_depth'],
                        1.0  # 互換性スコア
                    ]
                    edge_features.append(edge_feat)
                    
                    # エッジポートタイプ
                    source_outlet_type = 'signal' if conn['source_outlet_types'] else 'unknown'
                    target_inlet_type = 'signal' if conn['target_inlet_types'] else 'unknown'
                    
                    if conn['source_outlet_types'] and len(conn['source_outlet_types']) > conn['source_port']:
                        source_outlet_type = conn['source_outlet_types'][conn['source_port']]
                    if conn['target_inlet_types'] and len(conn['target_inlet_types']) > conn['target_port']:
                        target_inlet_type = conn['target_inlet_types'][conn['target_port']]
                        
                    edge_port_types.append([source_outlet_type, target_inlet_type])
                    
            # PyTorch Geometricデータ作成
            data = Data(
                x=torch.tensor(features, dtype=torch.float32),
                edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.tensor(edge_features, dtype=torch.float32) if edge_features else torch.empty((0, 5), dtype=torch.float32),
                num_nodes=num_objects,
                file_path=patch_file
            )
            
            # ポートタイプ情報を追加
            data.port_type_features = torch.tensor(port_features, dtype=torch.float32)
            if edge_port_types:
                data.edge_port_types = edge_port_types  # 文字列のまま保存（後で必要に応じてエンコード）
                
            return data
            
        except Exception as e:
            logger.error(f"グラフ作成エラー ({patch_file}): {e}")
            return None


def main():
    # サンプルデータセット作成
    creator = SampleGraphCreator('scripts/db_settings.ini')
    graphs = creator.create_sample_dataset(num_patches=100)
    
    logger.info("\n✅ サンプルデータセット作成完了！")
    logger.info("次のステップ: このデータでGNNモデルのトレーニングをテストできます")


if __name__ == "__main__":
    main()