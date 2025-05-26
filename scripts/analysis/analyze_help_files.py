#!/usr/bin/env python3
"""
.maxhelpファイルから実際の接続パターンを分析してポートタイプを推測
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import logging
from db_connector import DatabaseConnector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HelpFileAnalyzer:
    def __init__(self, db_config_path: str):
        self.db = DatabaseConnector(db_config_path)
        self.help_base_path = "/Applications/Ableton Live 12 Suite.app/Contents/App-Resources/Max/Max.app/Contents/Resources/C74/help"
        self.connection_patterns = defaultdict(lambda: defaultdict(set))
        
    def analyze_help_files(self):
        """すべてのヘルプファイルを分析"""
        help_files = []
        
        # .maxhelpファイルを収集
        for subdir in ['max', 'msp', 'jit']:
            dir_path = os.path.join(self.help_base_path, subdir)
            if os.path.exists(dir_path):
                help_files.extend(Path(dir_path).glob("*.maxhelp"))
                
        logger.info(f"見つかったヘルプファイル: {len(help_files)}個")
        
        # 各ヘルプファイルを分析
        for help_file in help_files[:100]:  # まず100個で試す
            self.analyze_single_help(help_file)
            
        # 結果を表示
        self.show_patterns()
        
    def analyze_single_help(self, help_file):
        """単一のヘルプファイルを分析"""
        try:
            with open(help_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if 'patcher' not in data:
                return
                
            # ヘルプ対象のオブジェクト名を取得
            target_object = help_file.stem  # 例: cycle~.maxhelp → cycle~
            
            # パッチ内のオブジェクトを収集
            objects = {}
            if 'boxes' in data['patcher']:
                for box in data['patcher']['boxes']:
                    if 'box' in box:
                        box_data = box['box']
                        if 'id' in box_data:
                            obj_id = box_data['id']
                            maxclass = box_data.get('maxclass', '')
                            text = box_data.get('text', '')
                            
                            # オブジェクトタイプを判定
                            if maxclass == 'newobj' and text:
                                obj_type = text.split()[0]
                            else:
                                obj_type = maxclass
                                
                            objects[obj_id] = {
                                'type': obj_type,
                                'maxclass': maxclass,
                                'text': text
                            }
                            
            # 接続を分析
            if 'lines' in data['patcher']:
                for line in data['patcher']['lines']:
                    if 'patchline' in line:
                        patchline = line['patchline']
                        source = patchline.get('source', ['', 0])
                        dest = patchline.get('destination', ['', 0])
                        
                        if len(source) >= 2 and len(dest) >= 2:
                            source_id, source_port = source[0], source[1]
                            dest_id, dest_port = dest[0], dest[1]
                            
                            if source_id in objects and dest_id in objects:
                                source_obj = objects[source_id]
                                dest_obj = objects[dest_id]
                                
                                # ターゲットオブジェクトへの接続を記録
                                if dest_obj['type'] == target_object:
                                    # どのタイプのオブジェクトがどのポートに接続されているか
                                    self.connection_patterns[target_object][f'inlet_{dest_port}'].add(source_obj['type'])
                                    
                                # ターゲットオブジェクトからの接続を記録
                                if source_obj['type'] == target_object:
                                    # どのポートからどのタイプのオブジェクトへ接続されているか
                                    self.connection_patterns[target_object][f'outlet_{source_port}_to'].add(dest_obj['type'])
                                    
        except Exception as e:
            logger.error(f"ヘルプファイル分析エラー ({help_file}): {e}")
            
    def show_patterns(self):
        """発見されたパターンを表示"""
        logger.info("\n=== ヘルプファイルから推測されたポートタイプ ===")
        
        # 特定のオブジェクトについて詳細を表示
        for obj_name in ['cycle~', 'dac~', '+', 'metro', 'button', 'flonum']:
            if obj_name in self.connection_patterns:
                logger.info(f"\n{obj_name}:")
                patterns = self.connection_patterns[obj_name]
                
                # インレット
                inlet_patterns = {k: v for k, v in patterns.items() if k.startswith('inlet_')}
                if inlet_patterns:
                    logger.info("  インレット:")
                    for port, connected_types in sorted(inlet_patterns.items()):
                        port_num = port.split('_')[1]
                        types_str = ', '.join(sorted(connected_types))
                        logger.info(f"    inlet {port_num}: ← {types_str}")
                        
                # アウトレット
                outlet_patterns = {k: v for k, v in patterns.items() if k.startswith('outlet_')}
                if outlet_patterns:
                    logger.info("  アウトレット:")
                    for port, connected_types in sorted(outlet_patterns.items()):
                        port_num = port.split('_')[1]
                        types_str = ', '.join(sorted(connected_types))
                        logger.info(f"    outlet {port_num}: → {types_str}")
                        
    def infer_port_types(self):
        """接続パターンからポートタイプを推測"""
        type_inference = {}
        
        # 音声系オブジェクトのパターン
        signal_objects = {'dac~', 'adc~', 'cycle~', 'noise~', 'saw~', 'rect~', 'tri~', 
                         'phasor~', 'oscilloscope~', 'spectroscope~', 'meter~', 'gain~',
                         'live.gain~', '*~', '+~', '-~', '/~'}
        
        # 数値系オブジェクト
        number_objects = {'number', 'flonum', 'int', 'float'}
        
        # メッセージ系
        message_objects = {'message', 'prepend', 'append', 'sprintf'}
        
        for obj_name, patterns in self.connection_patterns.items():
            inlets = []
            outlets = []
            
            # インレットタイプを推測
            for port, connected_types in patterns.items():
                if port.startswith('inlet_'):
                    # 接続元のタイプから推測
                    if any(t in signal_objects for t in connected_types):
                        inlets.append('signal')
                    elif any(t in number_objects for t in connected_types):
                        if obj_name.endswith('~'):
                            inlets.append('signal/float')
                        else:
                            inlets.append('float/int')
                    elif any(t in message_objects for t in connected_types):
                        inlets.append('anything')
                    else:
                        inlets.append('anything')
                        
            # アウトレットタイプを推測
            for port, connected_types in patterns.items():
                if port.startswith('outlet_') and port.endswith('_to'):
                    # 接続先のタイプから推測
                    if any(t in signal_objects for t in connected_types):
                        outlets.append('signal')
                    elif any(t in number_objects for t in connected_types):
                        if obj_name.endswith('~'):
                            outlets.append('signal')
                        else:
                            outlets.append('float/int')
                    else:
                        outlets.append('anything')
                        
            if inlets or outlets:
                type_inference[obj_name] = {
                    'inlet_types': inlets,
                    'outlet_types': outlets
                }
                
        return type_inference


def main():
    analyzer = HelpFileAnalyzer('scripts/db_settings.ini')
    
    # ヘルプファイルを分析
    analyzer.analyze_help_files()
    
    # ポートタイプを推測
    inferred_types = analyzer.infer_port_types()
    
    logger.info("\n=== 推測されたポートタイプ ===")
    for obj_name, types in sorted(inferred_types.items())[:20]:
        logger.info(f"{obj_name}:")
        logger.info(f"  inlets: {types['inlet_types']}")
        logger.info(f"  outlets: {types['outlet_types']}")


if __name__ == "__main__":
    main()