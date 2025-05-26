#!/usr/bin/env python3
"""
Max/MSPパッチからPyTorch Geometric用のグラフデータセットを作成

このスクリプトは以下を実行します：
1. PostgreSQLからパッチデータを読み込み
2. 特徴量を一つも損なわずにグラフ構造に変換
3. PyTorch Geometric形式で保存
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
import networkx as nx
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from db_connector import DatabaseConnector
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MaxPatchGraphDataset:
    """Max/MSPパッチをグラフデータセットに変換するクラス"""
    
    def __init__(self, db_config_path: str):
        """
        初期化
        
        Args:
            db_config_path: データベース設定ファイルのパス
        """
        self.db = DatabaseConnector(db_config_path)
        self.object_type_encoder = LabelEncoder()
        self.maxclass_encoder = LabelEncoder()
        self.device_type_encoder = LabelEncoder()
        self.port_type_encoder = LabelEncoder()
        self.position_scaler = StandardScaler()
        self.text_vocab = {}  # テキスト内容の語彙
        self.graphs = []
        
    def connect(self):
        """データベースに接続"""
        self.db.connect()
        logger.info("データベースに接続しました")
        
    def disconnect(self):
        """データベース接続を切断"""
        self.db.disconnect()
        logger.info("データベース接続を切断しました")
        
    def load_patch_data(self, patch_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        特定のパッチファイルのデータを読み込む
        
        Args:
            patch_file: パッチファイルパス
            
        Returns:
            (object_details, object_connections)のタプル
        """
        # オブジェクト詳細を取得
        objects_query = """
        SELECT * FROM object_details 
        WHERE patch_file = %s
        ORDER BY id
        """
        objects_df = pd.DataFrame(self.db.execute_query(objects_query, (patch_file,)))
        
        # 接続情報を取得
        connections_query = """
        SELECT * FROM object_connections 
        WHERE patch_file = %s
        ORDER BY id
        """
        connections_df = pd.DataFrame(self.db.execute_query(connections_query, (patch_file,)))
        
        return objects_df, connections_df
        
    def extract_node_features(self, objects_df: pd.DataFrame) -> torch.Tensor:
        """
        オブジェクトからノード特徴量を抽出
        
        特徴量:
        1. object_type (one-hot)
        2. maxclass (one-hot)
        3. position (x, y, width, height) - 正規化
        4. inlet/outlet数
        5. text_content埋め込み
        6. saved_attributes特徴量
        7. 階層深度
        """
        features_list = []
        
        for _, obj in objects_df.iterrows():
            feature_vec = []
            
            # 1. Object type (one-hot encoding)
            if hasattr(self.object_type_encoder, 'classes_'):
                obj_type_vec = np.zeros(len(self.object_type_encoder.classes_))
                if obj['object_type'] in self.object_type_encoder.classes_:
                    idx = list(self.object_type_encoder.classes_).index(obj['object_type'])
                    obj_type_vec[idx] = 1
            else:
                obj_type_vec = [0]  # 初回は仮の値
            feature_vec.extend(obj_type_vec)
            
            # 2. Maxclass (one-hot encoding)
            if hasattr(self.maxclass_encoder, 'classes_'):
                maxclass_vec = np.zeros(len(self.maxclass_encoder.classes_))
                if obj['maxclass'] in self.maxclass_encoder.classes_:
                    idx = list(self.maxclass_encoder.classes_).index(obj['maxclass'])
                    maxclass_vec[idx] = 1
            else:
                maxclass_vec = [0]  # 初回は仮の値
            feature_vec.extend(maxclass_vec)
            
            # 3. Position (normalized)
            position = obj['position'] if obj['position'] else [0, 0, 0, 0]
            if isinstance(position, str):
                position = json.loads(position)
            feature_vec.extend(position[:4])  # x, y, width, height
            
            # 4. Port counts
            inlet_count = obj['numinlets'] if obj['numinlets'] else 0
            outlet_count = obj['numoutlets'] if obj['numoutlets'] else 0
            feature_vec.extend([inlet_count, outlet_count])
            
            # 5. Text content embedding (簡易版 - 後で改善)
            text_len = len(obj['text_content']) if obj['text_content'] else 0
            has_dollar = 1 if obj['text_content'] and '$' in obj['text_content'] else 0
            has_space = 1 if obj['text_content'] and ' ' in obj['text_content'] else 0
            feature_vec.extend([text_len, has_dollar, has_space])
            
            # 6. Saved attributes features
            if obj['saved_attributes']:
                attrs = obj['saved_attributes'] if isinstance(obj['saved_attributes'], dict) else {}
                # 重要な属性の存在チェック
                has_parameter = 1 if 'parameter_enable' in attrs else 0
                has_fontsize = 1 if 'fontsize' in attrs else 0
                has_presentation = 1 if 'presentation' in attrs else 0
                feature_vec.extend([has_parameter, has_fontsize, has_presentation])
            else:
                feature_vec.extend([0, 0, 0])
            
            # 7. 階層深度（full_object_idから推定）
            hierarchy_depth = obj['full_object_id'].count(':') if obj['full_object_id'] else 0
            feature_vec.append(hierarchy_depth)
            
            features_list.append(feature_vec)
            
        return torch.tensor(features_list, dtype=torch.float32)
        
    def extract_edge_features(self, connections_df: pd.DataFrame, 
                            objects_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        接続からエッジ特徴量を抽出
        
        Returns:
            (edge_index, edge_attr)のタプル
        """
        # オブジェクトIDからインデックスへのマッピング
        obj_id_to_idx = {row['object_id']: idx for idx, row in objects_df.iterrows()}
        
        edge_list = []
        edge_features = []
        
        for _, conn in connections_df.iterrows():
            # ソースとターゲットのオブジェクトを特定
            source_matches = objects_df[objects_df['object_type'] == conn['source_object_type']]
            target_matches = objects_df[objects_df['object_type'] == conn['target_object_type']]
            
            if len(source_matches) == 0 or len(target_matches) == 0:
                continue
                
            # 位置情報を使って最も近いオブジェクトペアを見つける（簡易版）
            # TODO: より正確なマッチング方法の実装
            source_idx = source_matches.index[0]
            target_idx = target_matches.index[0]
            
            edge_list.append([source_idx, target_idx])
            
            # エッジ特徴量
            edge_feat = []
            
            # 1. ポート番号
            edge_feat.extend([conn['source_port'], conn['target_port']])
            
            # 2. 接続順序
            edge_feat.append(conn['connection_order'] if conn['connection_order'] else 0)
            
            # 3. 階層深度
            edge_feat.append(conn['hierarchy_depth'])
            
            # 4. ポートタイプ互換性（簡易版）
            # TODO: 実際のポートタイプマッチングの実装
            edge_feat.append(1.0)  # 仮の互換性スコア
            
            edge_features.append(edge_feat)
            
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 5), dtype=torch.float32)
            
        return edge_index, edge_attr
        
    def create_graph(self, patch_file: str) -> Optional[Data]:
        """
        パッチファイルからグラフを作成
        
        Args:
            patch_file: パッチファイルパス
            
        Returns:
            PyTorch GeometricのDataオブジェクト
        """
        try:
            # データ読み込み
            objects_df, connections_df = self.load_patch_data(patch_file)
            
            if objects_df.empty:
                logger.warning(f"オブジェクトが見つかりません: {patch_file}")
                return None
                
            # 特徴量抽出
            node_features = self.extract_node_features(objects_df)
            edge_index, edge_attr = self.extract_edge_features(connections_df, objects_df)
            
            # グラフレベル特徴量
            device_type = connections_df.iloc[0]['device_type'] if not connections_df.empty else 'unknown'
            file_type = connections_df.iloc[0]['file_type'] if not connections_df.empty else 'unknown'
            
            # PyTorch GeometricのDataオブジェクト作成
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=len(objects_df),
                patch_file=patch_file,
                device_type=device_type,
                file_type=file_type
            )
            
            # メタデータを保存（復元用）
            data.object_ids = objects_df['object_id'].tolist()
            data.object_types = objects_df['object_type'].tolist()
            data.positions = objects_df['position'].tolist()
            data.text_contents = objects_df['text_content'].tolist()
            
            return data
            
        except Exception as e:
            logger.error(f"グラフ作成エラー ({patch_file}): {e}")
            return None
            
    def fit_encoders(self):
        """
        全データを使ってエンコーダーを学習
        """
        logger.info("エンコーダーの学習を開始...")
        
        # 全オブジェクトタイプを取得
        query = "SELECT DISTINCT object_type FROM object_details WHERE object_type IS NOT NULL"
        object_types = [row['object_type'] for row in self.db.execute_query(query)]
        self.object_type_encoder.fit(object_types)
        logger.info(f"オブジェクトタイプ数: {len(object_types)}")
        
        # 全maxclassを取得
        query = "SELECT DISTINCT maxclass FROM object_details WHERE maxclass IS NOT NULL"
        maxclasses = [row['maxclass'] for row in self.db.execute_query(query)]
        self.maxclass_encoder.fit(maxclasses)
        logger.info(f"maxclass数: {len(maxclasses)}")
        
        # デバイスタイプ
        query = "SELECT DISTINCT device_type FROM object_connections WHERE device_type IS NOT NULL"
        device_types = [row['device_type'] for row in self.db.execute_query(query)]
        self.device_type_encoder.fit(device_types)
        logger.info(f"デバイスタイプ数: {len(device_types)}")
        
        # 位置情報の正規化パラメータを計算
        query = "SELECT position FROM object_details WHERE position IS NOT NULL LIMIT 10000"
        positions = []
        for row in self.db.execute_query(query):
            if row['position']:
                pos = row['position'] if isinstance(row['position'], list) else json.loads(row['position'])
                positions.append(pos[:4])
        if positions:
            self.position_scaler.fit(positions)
            
        logger.info("エンコーダーの学習完了")
        
    def create_dataset(self, limit: Optional[int] = None, 
                      output_dir: str = "data/graphs") -> List[Data]:
        """
        データセット全体を作成
        
        Args:
            limit: 処理するパッチファイルの最大数
            output_dir: 出力ディレクトリ
            
        Returns:
            グラフデータのリスト
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # エンコーダーを学習
        self.fit_encoders()
        
        # 全パッチファイルを取得
        query = """
        SELECT DISTINCT patch_file 
        FROM object_connections 
        ORDER BY patch_file
        """
        if limit:
            query += f" LIMIT {limit}"
            
        patch_files = [row['patch_file'] for row in self.db.execute_query(query)]
        logger.info(f"処理するパッチファイル数: {len(patch_files)}")
        
        graphs = []
        for i, patch_file in enumerate(patch_files):
            if i % 100 == 0:
                logger.info(f"進捗: {i}/{len(patch_files)} ({i/len(patch_files)*100:.1f}%)")
                
            graph = self.create_graph(patch_file)
            if graph is not None:
                graphs.append(graph)
                
        logger.info(f"作成されたグラフ数: {len(graphs)}")
        
        # データセットを保存
        dataset_path = output_path / "max_patch_graphs.pkl"
        with open(dataset_path, 'wb') as f:
            pickle.dump({
                'graphs': graphs,
                'object_type_encoder': self.object_type_encoder,
                'maxclass_encoder': self.maxclass_encoder,
                'device_type_encoder': self.device_type_encoder,
                'position_scaler': self.position_scaler
            }, f)
            
        logger.info(f"データセットを保存しました: {dataset_path}")
        
        # 統計情報を出力
        self.print_dataset_stats(graphs)
        
        return graphs
        
    def print_dataset_stats(self, graphs: List[Data]):
        """データセットの統計情報を出力"""
        
        node_counts = [g.num_nodes for g in graphs]
        edge_counts = [g.edge_index.shape[1] for g in graphs]
        
        stats = {
            'total_graphs': len(graphs),
            'total_nodes': sum(node_counts),
            'total_edges': sum(edge_counts),
            'avg_nodes_per_graph': np.mean(node_counts),
            'std_nodes_per_graph': np.std(node_counts),
            'min_nodes': min(node_counts),
            'max_nodes': max(node_counts),
            'avg_edges_per_graph': np.mean(edge_counts),
            'std_edges_per_graph': np.std(edge_counts),
            'min_edges': min(edge_counts),
            'max_edges': max(edge_counts),
            'node_feature_dim': graphs[0].x.shape[1] if graphs else 0
        }
        
        print("\n=== データセット統計 ===")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
                
        # デバイスタイプ別統計
        device_type_counts = defaultdict(int)
        for g in graphs:
            device_type_counts[g.device_type] += 1
            
        print("\n=== デバイスタイプ別分布 ===")
        for device_type, count in sorted(device_type_counts.items(), 
                                       key=lambda x: x[1], reverse=True):
            print(f"{device_type}: {count} ({count/len(graphs)*100:.1f}%)")


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Max/MSPパッチからグラフデータセットを作成')
    parser.add_argument('--config', type=str, default='scripts/db_settings.ini',
                       help='データベース設定ファイル')
    parser.add_argument('--limit', type=int, default=None,
                       help='処理するパッチファイルの最大数')
    parser.add_argument('--output', type=str, default='data/graphs',
                       help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    # データセット作成
    dataset_creator = MaxPatchGraphDataset(args.config)
    
    try:
        dataset_creator.connect()
        graphs = dataset_creator.create_dataset(limit=args.limit, output_dir=args.output)
        
        # サンプルグラフの詳細を表示
        if graphs:
            print("\n=== サンプルグラフの詳細 ===")
            sample_graph = graphs[0]
            print(f"パッチファイル: {sample_graph.patch_file}")
            print(f"ノード数: {sample_graph.num_nodes}")
            print(f"エッジ数: {sample_graph.edge_index.shape[1]}")
            print(f"ノード特徴量次元: {sample_graph.x.shape[1]}")
            print(f"エッジ特徴量次元: {sample_graph.edge_attr.shape[1] if sample_graph.edge_attr.numel() > 0 else 0}")
            print(f"最初の5つのオブジェクトタイプ: {sample_graph.object_types[:5]}")
            
    finally:
        dataset_creator.disconnect()


if __name__ == "__main__":
    main()