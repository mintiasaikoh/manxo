#!/usr/bin/env python3
"""
大規模データセットから効率的にサンプリングしてグラフデータセットを作成
"""

import sys
sys.path.append('/Users/mymac/manxo')
from scripts.db_connector import DatabaseConnector
from scripts.create_graph_dataset import MaxPatchGraphDataset
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # パラメータ
    SAMPLE_SIZE = 5000  # 5000パッチをサンプリング
    MIN_CONNECTIONS = 10
    MAX_CONNECTIONS = 1000
    OUTPUT_DIR = 'data/graph_dataset_5k'
    
    logger.info(f"サンプリングサイズ: {SAMPLE_SIZE}")
    logger.info(f"接続数範囲: {MIN_CONNECTIONS} - {MAX_CONNECTIONS}")
    
    # データセット作成器を初期化
    dataset_creator = MaxPatchGraphDataset('scripts/db_settings.ini')
    
    try:
        dataset_creator.connect()
        
        # まずエンコーダーを学習
        dataset_creator.fit_encoders()
        
        # パッチファイルを取得（接続数でフィルタリング）
        query = f"""
        SELECT 
            patch_file,
            file_type,
            device_type,
            COUNT(*) as connection_count
        FROM object_connections
        GROUP BY patch_file, file_type, device_type
        HAVING COUNT(*) >= {MIN_CONNECTIONS} 
           AND COUNT(*) <= {MAX_CONNECTIONS}
        ORDER BY RANDOM()
        LIMIT {SAMPLE_SIZE}
        """
        
        patch_files = []
        for row in dataset_creator.db.execute_query(query):
            patch_files.append(row['patch_file'])
        
        logger.info(f"取得したパッチファイル数: {len(patch_files)}")
        
        # グラフを作成
        graphs = []
        for i, patch_file in enumerate(patch_files):
            if i % 100 == 0:
                logger.info(f"進捗: {i}/{len(patch_files)} ({i/len(patch_files)*100:.1f}%)")
            
            try:
                graph = dataset_creator.create_graph(patch_file)
                if graph is not None:
                    graphs.append(graph)
            except Exception as e:
                logger.warning(f"スキップ: {patch_file} - {str(e)}")
        
        logger.info(f"作成されたグラフ数: {len(graphs)}")
        
        # 保存
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(parents=True, exist_ok=True)
        
        import pickle
        with open(output_path / 'max_patch_graphs.pkl', 'wb') as f:
            pickle.dump({
                'graphs': graphs,
                'object_type_encoder': dataset_creator.object_type_encoder,
                'maxclass_encoder': dataset_creator.maxclass_encoder,
                'device_type_encoder': dataset_creator.device_type_encoder,
                'position_scaler': dataset_creator.position_scaler
            }, f)
        
        logger.info(f"保存完了: {output_path}")
        
        # 統計を表示
        dataset_creator.print_dataset_stats(graphs)
        
    finally:
        dataset_creator.disconnect()


if __name__ == "__main__":
    main()