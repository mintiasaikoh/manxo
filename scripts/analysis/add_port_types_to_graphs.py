#!/usr/bin/env python3
"""
既存のグラフデータセットにポートタイプ情報を追加する

19GBのデータを再生成するのは時間がかかるので、
既存のグラフにポートタイプ情報を追加する方法を採用！
"""

import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any
from db_connector import DatabaseConnector
from tqdm import tqdm
import logging
import gc

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PortTypeEnhancer:
    """グラフデータにポートタイプ情報を追加するクラス"""
    
    def __init__(self, db_config_path: str):
        self.db = DatabaseConnector(db_config_path)
        self.port_type_cache = {}
        
    def connect(self):
        self.db.connect()
        logger.info("データベースに接続しました")
        
    def disconnect(self):
        self.db.disconnect()
        logger.info("データベース接続を切断しました")
        
    def load_port_types_batch(self, patch_files: List[str]) -> Dict[str, Dict]:
        """
        パッチファイルのバッチからポートタイプ情報を取得
        """
        placeholders = ','.join(['%s'] * len(patch_files))
        
        # object_detailsからポートタイプ情報を取得
        query = f"""
        SELECT 
            patch_file,
            object_id,
            object_type,
            inlet_types,
            outlet_types,
            position
        FROM object_details
        WHERE patch_file IN ({placeholders})
        ORDER BY patch_file, id
        """
        
        results = self.db.execute_query(query, tuple(patch_files))
        
        # パッチファイルごとに整理
        port_info_by_patch = {}
        for row in results:
            patch_file = row['patch_file']
            if patch_file not in port_info_by_patch:
                port_info_by_patch[patch_file] = {}
                
            port_info_by_patch[patch_file][row['object_id']] = {
                'object_type': row['object_type'],
                'inlet_types': row['inlet_types'] or [],
                'outlet_types': row['outlet_types'] or [],
                'position': row['position']
            }
            
        return port_info_by_patch
        
    def enhance_graph_with_port_types(self, graph, port_info: Dict) -> None:
        """
        グラフにポートタイプ情報を追加（in-place）
        """
        # ノードのポートタイプ情報を追加
        inlet_types_list = []
        outlet_types_list = []
        
        # object_idsを使ってポートタイプを取得
        if hasattr(graph, 'object_ids'):
            for obj_id in graph.object_ids:
                if obj_id in port_info:
                    info = port_info[obj_id]
                    inlet_types_list.append(info['inlet_types'])
                    outlet_types_list.append(info['outlet_types'])
                else:
                    # デフォルト値
                    inlet_types_list.append([])
                    outlet_types_list.append([])
        else:
            # object_idsがない場合はobject_typesで推測
            for i, obj_type in enumerate(graph.object_types):
                found = False
                for obj_id, info in port_info.items():
                    if info['object_type'] == obj_type:
                        inlet_types_list.append(info['inlet_types'])
                        outlet_types_list.append(info['outlet_types'])
                        found = True
                        break
                if not found:
                    inlet_types_list.append([])
                    outlet_types_list.append([])
                    
        # グラフに属性を追加
        graph.inlet_types = inlet_types_list
        graph.outlet_types = outlet_types_list
        
        # エッジのポートタイプ互換性を計算
        if graph.edge_index.shape[1] > 0:
            edge_port_compatibility = []
            
            for i in range(graph.edge_index.shape[1]):
                src_idx = graph.edge_index[0, i].item()
                dst_idx = graph.edge_index[1, i].item()
                
                # ポート番号を取得（edge_attrから）
                if graph.edge_attr is not None and graph.edge_attr.shape[0] > i:
                    src_port = int(graph.edge_attr[i, 0].item())
                    dst_port = int(graph.edge_attr[i, 1].item())
                else:
                    src_port = 0
                    dst_port = 0
                
                # ポートタイプを取得
                src_outlet_types = outlet_types_list[src_idx] if src_idx < len(outlet_types_list) else []
                dst_inlet_types = inlet_types_list[dst_idx] if dst_idx < len(inlet_types_list) else []
                
                # 互換性スコアを計算
                compatibility = 1.0  # デフォルト
                
                if (src_outlet_types and src_port < len(src_outlet_types) and 
                    dst_inlet_types and dst_port < len(dst_inlet_types)):
                    src_type = src_outlet_types[src_port]
                    dst_type = dst_inlet_types[dst_port]
                    
                    # 簡易的な互換性チェック
                    if src_type == dst_type:
                        compatibility = 1.0
                    elif 'signal' in src_type and 'signal' in dst_type:
                        compatibility = 0.9
                    elif src_type == '' or dst_type == '':
                        compatibility = 0.8
                    else:
                        compatibility = 0.5
                        
                edge_port_compatibility.append(compatibility)
                
            # エッジ特徴量に互換性スコアを追加
            if graph.edge_attr is not None:
                # 既存のedge_attrの最後の列（互換性スコア）を更新
                graph.edge_attr[:, -1] = torch.tensor(edge_port_compatibility, dtype=torch.float32)
                
    def process_dataset(self, input_path: str, output_path: str, batch_size: int = 100):
        """
        データセット全体を処理してポートタイプ情報を追加
        """
        logger.info(f"データセットを読み込み中: {input_path}")
        
        # 既存のデータセットを読み込み
        with open(input_path, 'rb') as f:
            dataset = pickle.load(f)
            
        graphs = dataset['graphs']
        logger.info(f"グラフ数: {len(graphs)}")
        
        # パッチファイルごとにグループ化
        graphs_by_patch = {}
        for i, graph in enumerate(graphs):
            patch_file = graph.patch_file
            if patch_file not in graphs_by_patch:
                graphs_by_patch[patch_file] = []
            graphs_by_patch[patch_file].append((i, graph))
            
        # バッチ処理
        all_patch_files = list(graphs_by_patch.keys())
        
        for batch_start in tqdm(range(0, len(all_patch_files), batch_size), 
                               desc="ポートタイプ情報を追加中"):
            batch_files = all_patch_files[batch_start:batch_start + batch_size]
            
            # バッチでポート情報を取得
            port_info_batch = self.load_port_types_batch(batch_files)
            
            # 各グラフを更新
            for patch_file in batch_files:
                if patch_file in port_info_batch and patch_file in graphs_by_patch:
                    port_info = port_info_batch[patch_file]
                    
                    for idx, graph in graphs_by_patch[patch_file]:
                        self.enhance_graph_with_port_types(graph, port_info)
                        
            # メモリクリア
            if batch_start % (batch_size * 10) == 0:
                gc.collect()
                
        # ポートタイプエンコーダーを作成
        all_port_types = set()
        for graph in graphs:
            if hasattr(graph, 'inlet_types'):
                for types_list in graph.inlet_types:
                    all_port_types.update(types_list)
            if hasattr(graph, 'outlet_types'):
                for types_list in graph.outlet_types:
                    all_port_types.update(types_list)
                    
        logger.info(f"ユニークなポートタイプ数: {len(all_port_types)}")
        
        # 更新されたデータセットを保存
        dataset['port_types'] = sorted(list(all_port_types))
        dataset['enhanced_with_port_types'] = True
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"拡張データセットを保存中: {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        logger.info("完了！")
        
        # 統計情報を表示
        self.print_enhancement_stats(graphs)
        
    def print_enhancement_stats(self, graphs: List):
        """拡張の統計情報を表示"""
        
        graphs_with_port_types = 0
        total_inlet_types = 0
        total_outlet_types = 0
        
        for graph in graphs:
            if hasattr(graph, 'inlet_types') and hasattr(graph, 'outlet_types'):
                graphs_with_port_types += 1
                for types in graph.inlet_types:
                    total_inlet_types += len(types)
                for types in graph.outlet_types:
                    total_outlet_types += len(types)
                    
        print(f"\n=== ポートタイプ拡張統計 ===")
        print(f"ポートタイプ情報を持つグラフ: {graphs_with_port_types}/{len(graphs)}")
        print(f"総インレットタイプ数: {total_inlet_types}")
        print(f"総アウトレットタイプ数: {total_outlet_types}")
        print(f"平均インレットタイプ/グラフ: {total_inlet_types/len(graphs):.2f}")
        print(f"平均アウトレットタイプ/グラフ: {total_outlet_types/len(graphs):.2f}")


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description='グラフデータセットにポートタイプ情報を追加')
    parser.add_argument('--input', type=str, default='data/graphs_full/max_patch_graphs.pkl',
                       help='入力データセットファイル')
    parser.add_argument('--output', type=str, default='data/graphs_full/max_patch_graphs_enhanced.pkl',
                       help='出力データセットファイル')
    parser.add_argument('--config', type=str, default='scripts/db_settings.ini',
                       help='データベース設定ファイル')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='バッチサイズ')
    
    args = parser.parse_args()
    
    # エンハンサー作成
    enhancer = PortTypeEnhancer(args.config)
    
    try:
        enhancer.connect()
        enhancer.process_dataset(
            input_path=args.input,
            output_path=args.output,
            batch_size=args.batch_size
        )
    finally:
        enhancer.disconnect()


if __name__ == "__main__":
    main()