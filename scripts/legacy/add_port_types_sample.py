#!/usr/bin/env python3
"""
サンプルデータでポートタイプ追加をテスト
"""

import pickle
import torch
from db_connector import DatabaseConnector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_port_type_addition():
    # データベース接続
    db = DatabaseConnector('scripts/db_settings.ini')
    db.connect()
    
    # 元データセットの最初の100グラフを読み込み
    logger.info("元データセットを読み込み中...")
    with open('data/graphs_full/max_patch_graphs.pkl', 'rb') as f:
        dataset = pickle.load(f)
    
    graphs = dataset['graphs'][:100]  # 最初の100個だけ
    logger.info(f"テスト用グラフ数: {len(graphs)}")
    
    # 各グラフにポートタイプ情報を追加
    for i, graph in enumerate(graphs):
        patch_file = graph.patch_file
        
        # object_detailsからポートタイプ情報を取得
        query = """
        SELECT 
            object_id,
            object_type,
            inlet_types,
            outlet_types,
            position
        FROM object_details
        WHERE patch_file = %s
        ORDER BY id
        """
        
        results = db.execute_query(query, (patch_file,))
        
        # ポート情報をマッピング
        port_info = {}
        for row in results:
            port_info[row['object_id']] = {
                'object_type': row['object_type'],
                'inlet_types': row['inlet_types'] or [],
                'outlet_types': row['outlet_types'] or [],
                'position': row['position']
            }
        
        # グラフにポートタイプ情報を追加
        inlet_types_list = []
        outlet_types_list = []
        
        if hasattr(graph, 'object_ids'):
            for obj_id in graph.object_ids:
                if obj_id in port_info:
                    info = port_info[obj_id]
                    inlet_types_list.append(info['inlet_types'])
                    outlet_types_list.append(info['outlet_types'])
                else:
                    inlet_types_list.append([])
                    outlet_types_list.append([])
                    
        # 属性を追加
        graph.inlet_types = inlet_types_list
        graph.outlet_types = outlet_types_list
        
        if i == 0:
            logger.info(f"サンプルグラフのポートタイプ:")
            logger.info(f"  inlet_types[0:3]: {inlet_types_list[:3]}")
            logger.info(f"  outlet_types[0:3]: {outlet_types_list[:3]}")
    
    # 拡張されたサンプルデータセットを保存
    sample_dataset = {
        'graphs': graphs,
        'object_type_encoder': dataset['object_type_encoder'],
        'maxclass_encoder': dataset['maxclass_encoder'],
        'device_type_encoder': dataset['device_type_encoder'],
        'position_scaler': dataset['position_scaler'],
        'enhanced_with_port_types': True
    }
    
    output_path = 'data/graphs_full/max_patch_graphs_sample_enhanced.pkl'
    logger.info(f"サンプルデータセットを保存: {output_path}")
    
    with open(output_path, 'wb') as f:
        pickle.dump(sample_dataset, f)
    
    # 統計情報
    total_inlet_types = sum(len(types) for g in graphs for types in g.inlet_types)
    total_outlet_types = sum(len(types) for g in graphs for types in g.outlet_types)
    
    print(f"\n=== ポートタイプ追加完了 ===")
    print(f"グラフ数: {len(graphs)}")
    print(f"総インレットタイプ数: {total_inlet_types}")
    print(f"総アウトレットタイプ数: {total_outlet_types}")
    print(f"平均インレットタイプ/グラフ: {total_inlet_types/len(graphs):.2f}")
    print(f"平均アウトレットタイプ/グラフ: {total_outlet_types/len(graphs):.2f}")
    
    db.disconnect()

if __name__ == "__main__":
    test_port_type_addition()