#!/usr/bin/env python3
"""
自然言語からMax/MSPパッチを生成するスクリプト

このスクリプトは以下を実行します：
1. 自然言語の説明を受け取る
2. 学習済みGNNモデルを使ってパッチ構造を生成
3. .maxpat/.amxdファイルとして保存
"""

import os
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from db_connector import DatabaseConnector
from train_gnn_model import MaxPatchGNN
import logging

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MaxPatchGenerator:
    """自然言語からMax/MSPパッチを生成するクラス"""
    
    def __init__(self, model_path: str, dataset_path: str, db_config_path: str):
        """
        初期化
        
        Args:
            model_path: 学習済みモデルのパス
            dataset_path: データセットファイルのパス（エンコーダー情報）
            db_config_path: データベース設定ファイルのパス
        """
        self.db = DatabaseConnector(db_config_path)
        self.db.connect()
        
        # データセットとエンコーダーを読み込み
        with open(dataset_path, 'rb') as f:
            dataset_info = pickle.load(f)
            self.object_type_encoder = dataset_info['object_type_encoder']
            self.maxclass_encoder = dataset_info['maxclass_encoder']
            self.device_type_encoder = dataset_info['device_type_encoder']
            self.position_scaler = dataset_info['position_scaler']
            sample_graph = dataset_info['graphs'][0]
            
        # モデルを読み込み
        checkpoint = torch.load(model_path, map_location='cpu')
        node_feature_dim = sample_graph.x.shape[1]
        edge_feature_dim = sample_graph.edge_attr.shape[1] if sample_graph.edge_attr.numel() > 0 else 5
        
        self.model = MaxPatchGNN(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=256,
            num_layers=3,
            num_object_types=len(self.object_type_encoder.classes_),
            num_device_types=6,
            gnn_type='GCN'
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # テンプレート情報を読み込み
        self.load_templates()
        
    def load_templates(self):
        """パッチテンプレートを読み込み"""
        # 基本的なエフェクトテンプレート
        self.templates = {
            'reverb': {
                'objects': ['inlet~', 'reverb~', 'outlet~'],
                'connections': [(0, 0, 1, 0), (1, 0, 2, 0), (1, 1, 2, 1)],
                'device_type': 'audio_effect'
            },
            'delay': {
                'objects': ['inlet~', 'delay~', '*~', 'outlet~'],
                'connections': [(0, 0, 1, 0), (1, 0, 2, 0), (2, 0, 3, 0)],
                'device_type': 'audio_effect'
            },
            'synthesizer': {
                'objects': ['notein', 'mtof', 'cycle~', 'gain~', 'dac~'],
                'connections': [(0, 0, 1, 0), (1, 0, 2, 0), (2, 0, 3, 0), (3, 0, 4, 0), (3, 0, 4, 1)],
                'device_type': 'instrument'
            },
            'filter': {
                'objects': ['inlet~', 'svf~', 'outlet~'],
                'connections': [(0, 0, 1, 0), (1, 0, 2, 0)],
                'device_type': 'audio_effect'
            }
        }
        
    def parse_description(self, description: str) -> Dict[str, Any]:
        """
        自然言語の説明を解析
        
        Args:
            description: パッチの説明文
            
        Returns:
            解析結果の辞書
        """
        description_lower = description.lower()
        
        # キーワードベースの簡易解析（後でLLMに置き換え）
        detected_features = {
            'effects': [],
            'device_type': 'audio_effect',
            'has_midi': False,
            'has_ui': False
        }
        
        # エフェクト検出
        effect_keywords = {
            'reverb': ['reverb', 'リバーブ', 'hall', 'room'],
            'delay': ['delay', 'ディレイ', 'echo'],
            'filter': ['filter', 'フィルター', 'lowpass', 'highpass'],
            'distortion': ['distortion', 'ディストーション', 'drive', 'overdrive']
        }
        
        for effect, keywords in effect_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                detected_features['effects'].append(effect)
                
        # デバイスタイプ検出
        if any(keyword in description_lower for keyword in ['synth', 'シンセ', 'synthesizer', '楽器']):
            detected_features['device_type'] = 'instrument'
            detected_features['has_midi'] = True
        elif any(keyword in description_lower for keyword in ['midi', 'note', 'ノート']):
            detected_features['device_type'] = 'midi_effect'
            detected_features['has_midi'] = True
            
        # UI要素検出
        if any(keyword in description_lower for keyword in ['slider', 'スライダー', 'knob', 'ノブ', 'button']):
            detected_features['has_ui'] = True
            
        return detected_features
        
    def generate_patch_structure(self, features: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
        """
        特徴量からパッチ構造を生成
        
        Args:
            features: 解析された特徴
            
        Returns:
            (objects, connections)のタプル
        """
        objects = []
        connections = []
        obj_id_counter = 1
        
        # デバイスタイプに基づいて基本構造を決定
        if features['device_type'] == 'instrument':
            # シンセサイザーの基本構造
            if features['has_midi']:
                # MIDI入力
                objects.append({
                    'id': f'obj-{obj_id_counter}',
                    'maxclass': 'newobj',
                    'text': 'notein',
                    'position': [50, 50, 60, 22]
                })
                midi_in_id = obj_id_counter
                obj_id_counter += 1
                
                # MIDI to Frequency
                objects.append({
                    'id': f'obj-{obj_id_counter}',
                    'maxclass': 'newobj',
                    'text': 'mtof',
                    'position': [50, 100, 35, 22]
                })
                mtof_id = obj_id_counter
                obj_id_counter += 1
                
                connections.append({
                    'source': [f'obj-{midi_in_id}', 0],
                    'dest': [f'obj-{mtof_id}', 0]
                })
                
                # オシレーター
                objects.append({
                    'id': f'obj-{obj_id_counter}',
                    'maxclass': 'newobj',
                    'text': 'cycle~',
                    'position': [50, 150, 50, 22]
                })
                osc_id = obj_id_counter
                obj_id_counter += 1
                
                connections.append({
                    'source': [f'obj-{mtof_id}', 0],
                    'dest': [f'obj-{osc_id}', 0]
                })
                
                current_signal_id = osc_id
            else:
                # 基本的なオシレーター
                objects.append({
                    'id': f'obj-{obj_id_counter}',
                    'maxclass': 'newobj',
                    'text': 'cycle~ 440',
                    'position': [50, 50, 70, 22]
                })
                current_signal_id = obj_id_counter
                obj_id_counter += 1
                
        else:
            # エフェクトの基本構造（入力）
            objects.append({
                'id': f'obj-{obj_id_counter}',
                'maxclass': 'newobj',
                'text': 'inlet~',
                'position': [50, 50, 40, 22]
            })
            current_signal_id = obj_id_counter
            obj_id_counter += 1
            
        # エフェクトを追加
        y_pos = 200
        for effect in features['effects']:
            if effect in self.templates:
                template = self.templates[effect]
                # メインエフェクトオブジェクトのみ追加（簡易版）
                if 'reverb' in effect:
                    objects.append({
                        'id': f'obj-{obj_id_counter}',
                        'maxclass': 'newobj',
                        'text': 'reverb~',
                        'position': [50, y_pos, 60, 22]
                    })
                elif 'delay' in effect:
                    objects.append({
                        'id': f'obj-{obj_id_counter}',
                        'maxclass': 'newobj',
                        'text': 'delay~ 1000',
                        'position': [50, y_pos, 80, 22]
                    })
                elif 'filter' in effect:
                    objects.append({
                        'id': f'obj-{obj_id_counter}',
                        'maxclass': 'newobj',
                        'text': 'svf~ 1000 0.5',
                        'position': [50, y_pos, 90, 22]
                    })
                    
                # 接続を追加
                connections.append({
                    'source': [f'obj-{current_signal_id}', 0],
                    'dest': [f'obj-{obj_id_counter}', 0]
                })
                
                current_signal_id = obj_id_counter
                obj_id_counter += 1
                y_pos += 50
                
        # UI要素を追加
        if features['has_ui']:
            # スライダー追加
            objects.append({
                'id': f'obj-{obj_id_counter}',
                'maxclass': 'slider',
                'position': [200, 100, 20, 140],
                'parameter_enable': 1
            })
            slider_id = obj_id_counter
            obj_id_counter += 1
            
            # スライダーをゲインに接続
            objects.append({
                'id': f'obj-{obj_id_counter}',
                'maxclass': 'newobj',
                'text': 'gain~',
                'position': [50, y_pos, 45, 140]
            })
            gain_id = obj_id_counter
            obj_id_counter += 1
            
            connections.append({
                'source': [f'obj-{current_signal_id}', 0],
                'dest': [f'obj-{gain_id}', 0]
            })
            connections.append({
                'source': [f'obj-{slider_id}', 0],
                'dest': [f'obj-{gain_id}', 0]
            })
            
            current_signal_id = gain_id
            y_pos += 50
            
        # 出力
        if features['device_type'] == 'instrument':
            # ステレオ出力
            objects.append({
                'id': f'obj-{obj_id_counter}',
                'maxclass': 'newobj',
                'text': 'dac~',
                'position': [50, y_pos + 50, 35, 22]
            })
            dac_id = obj_id_counter
            obj_id_counter += 1
            
            connections.append({
                'source': [f'obj-{current_signal_id}', 0],
                'dest': [f'obj-{dac_id}', 0]
            })
            connections.append({
                'source': [f'obj-{current_signal_id}', 0],
                'dest': [f'obj-{dac_id}', 1]
            })
        else:
            # エフェクト出力
            objects.append({
                'id': f'obj-{obj_id_counter}',
                'maxclass': 'newobj',
                'text': 'outlet~',
                'position': [50, y_pos + 50, 45, 22]
            })
            outlet_id = obj_id_counter
            
            connections.append({
                'source': [f'obj-{current_signal_id}', 0],
                'dest': [f'obj-{outlet_id}', 0]
            })
            
        return objects, connections
        
    def create_from_text(self, description: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        テキスト説明からパッチを生成
        
        Args:
            description: パッチの説明
            output_path: 出力ファイルパス（オプション）
            
        Returns:
            生成されたパッチデータ
        """
        logger.info(f"パッチ生成開始: {description}")
        
        # 説明を解析
        features = self.parse_description(description)
        logger.info(f"検出された特徴: {features}")
        
        # パッチ構造を生成
        objects, connections = self.generate_patch_structure(features)
        
        # Max/MSPパッチフォーマットに変換
        patch_data = {
            "patcher": {
                "fileversion": 1,
                "appversion": {
                    "major": 8,
                    "minor": 5,
                    "revision": 0,
                    "architecture": "x64",
                    "modernui": 1
                },
                "classnamespace": "box",
                "rect": [100, 100, 640, 480],
                "bglocked": 0,
                "openinpresentation": 0,
                "default_fontsize": 12.0,
                "default_fontface": 0,
                "default_fontname": "Arial",
                "gridonopen": 1,
                "gridsize": [15.0, 15.0],
                "gridsnaponopen": 1,
                "objectsnaponopen": 1,
                "statusbarvisible": 2,
                "toolbarvisible": 1,
                "lefttoolbarpinned": 0,
                "toptoolbarpinned": 0,
                "righttoolbarpinned": 0,
                "bottomtoolbarpinned": 0,
                "toolbars_unpinned_last_save": 0,
                "tallnewobj": 0,
                "boxanimatetime": 200,
                "enablehscroll": 1,
                "enablevscroll": 1,
                "devicewidth": 0.0,
                "description": description,
                "digest": "",
                "tags": "",
                "style": "",
                "subpatcher_template": "",
                "assistshowspatchername": 0,
                "boxes": objects,
                "lines": connections,
                "parameters": {},
                "dependency_cache": [],
                "autosave": 0
            }
        }
        
        # ファイルに保存
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(patch_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"パッチを保存しました: {output_file}")
            
        return patch_data
        
    def save_as_amxd(self, patch_data: Dict[str, Any], output_path: str):
        """
        Max for Liveデバイスとして保存
        
        Args:
            patch_data: パッチデータ
            output_path: 出力ファイルパス
        """
        # amxdヘッダーを追加
        features = self.parse_description(patch_data['patcher'].get('description', ''))
        
        if features['device_type'] == 'audio_effect':
            header = b'ampf\x00\x00\x00\x00aaaameta\x00\x00\x00\x00'
        elif features['device_type'] == 'midi_effect':
            header = b'ampf\x00\x00\x00\x00mmmmmeta\x00\x00\x00\x00'
        else:  # instrument
            header = b'ampf\x00\x00\x00\x00iiiimeta\x00\x00\x00\x00'
            
        # JSONデータをバイトに変換
        json_bytes = json.dumps(patch_data, separators=(',', ':')).encode('utf-8')
        
        # ptchセクションを追加
        ptch_header = b'ptch' + len(json_bytes).to_bytes(4, 'big')
        
        # 完全なamxdファイルを作成
        amxd_data = header + ptch_header + json_bytes
        
        # ファイルに保存
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'wb') as f:
            f.write(amxd_data)
            
        logger.info(f"Max for Liveデバイスを保存しました: {output_file}")


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description='自然言語からMax/MSPパッチを生成')
    parser.add_argument('description', type=str, help='パッチの説明')
    parser.add_argument('--model', type=str, default='models/max_patch_gnn.pt',
                       help='学習済みモデルのパス')
    parser.add_argument('--dataset', type=str, default='data/graphs/max_patch_graphs.pkl',
                       help='データセットファイル')
    parser.add_argument('--config', type=str, default='scripts/db_settings.ini',
                       help='データベース設定ファイル')
    parser.add_argument('--output', type=str, default=None,
                       help='出力ファイルパス')
    parser.add_argument('--format', type=str, default='maxpat',
                       choices=['maxpat', 'amxd'],
                       help='出力フォーマット')
    
    args = parser.parse_args()
    
    # ジェネレーター作成
    generator = MaxPatchGenerator(
        model_path=args.model,
        dataset_path=args.dataset,
        db_config_path=args.config
    )
    
    # 出力パスを決定
    if args.output:
        output_path = args.output
    else:
        # 説明から自動的にファイル名を生成
        safe_name = ''.join(c for c in args.description[:30] if c.isalnum() or c in ' -_')
        safe_name = safe_name.strip().replace(' ', '_')
        output_path = f"generated_patches/{safe_name}.{args.format}"
    
    # パッチ生成
    if args.format == 'maxpat':
        patch_data = generator.create_from_text(args.description, output_path)
    else:
        # amxd形式
        patch_data = generator.create_from_text(args.description)
        generator.save_as_amxd(patch_data, output_path)
    
    print(f"\nパッチが正常に生成されました！")
    print(f"ファイル: {output_path}")
    print(f"オブジェクト数: {len(patch_data['patcher']['boxes'])}")
    print(f"接続数: {len(patch_data['patcher']['lines'])}")


if __name__ == "__main__":
    main()