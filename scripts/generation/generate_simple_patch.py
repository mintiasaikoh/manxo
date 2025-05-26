#!/usr/bin/env python3
"""
シンプルなルールベースのMax/MSPパッチ生成
まずは基本的なパッチを生成できることを確認
"""

import json
from pathlib import Path
import argparse
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplePatchGenerator:
    """シンプルなパッチ生成クラス"""
    
    def __init__(self):
        self.box_id_counter = 1
        
    def create_empty_patch(self) -> Dict[str, Any]:
        """空のパッチ構造を作成"""
        return {
            "patcher": {
                "fileversion": 1,
                "appversion": {
                    "major": 8,
                    "minor": 5,
                    "revision": 0,
                    "architecture": "x64"
                },
                "classnamespace": "box",
                "rect": [100, 100, 800, 600],
                "bglocked": 0,
                "openinpresentation": 0,
                "default_fontsize": 12.0,
                "default_fontface": 0,
                "default_fontname": "Arial",
                "gridonopen": 1,
                "gridsize": [15.0, 15.0],
                "gridsnaponopen": 1,
                "toolbarvisible": 1,
                "boxes": [],
                "lines": [],
                "dependency_cache": [],
                "autosave": 0
            }
        }
    
    def create_box(self, maxclass: str, text: str = "", position: List[float] = None, 
                   numinlets: int = 1, numoutlets: int = 1, **kwargs) -> Dict[str, Any]:
        """オブジェクトボックスを作成"""
        if position is None:
            position = [100, 100, 100, 22]
        
        box = {
            "box": {
                "id": f"obj-{self.box_id_counter}",
                "maxclass": maxclass,
                "numinlets": numinlets,
                "numoutlets": numoutlets,
                "patching_rect": position
            }
        }
        
        if text:
            box["box"]["text"] = text
            
        # 追加のパラメータ
        for key, value in kwargs.items():
            box["box"][key] = value
            
        self.box_id_counter += 1
        return box
    
    def create_connection(self, source_id: str, source_outlet: int, 
                         dest_id: str, dest_inlet: int) -> Dict[str, Any]:
        """接続（パッチライン）を作成"""
        return {
            "patchline": {
                "destination": [dest_id, dest_inlet],
                "source": [source_id, source_outlet]
            }
        }
    
    def generate_filter_patch(self) -> Dict[str, Any]:
        """フィルターパッチを生成"""
        patch = self.create_empty_patch()
        boxes = []
        lines = []
        
        # ADC（オーディオ入力）
        adc = self.create_box("ezdac~", "", [50, 50, 61, 61], 1, 2)
        boxes.append(adc)
        adc_id = adc["box"]["id"]
        
        # スライダー（カットオフ周波数コントロール）
        slider = self.create_box("slider", "", [200, 50, 22, 140], 1, 1,
                                floatoutput=1, size=20000.0)
        boxes.append(slider)
        slider_id = slider["box"]["id"]
        
        # Number box（周波数表示）
        flonum = self.create_box("flonum", "", [200, 200, 50, 22], 1, 2,
                                format=6, numdecimalplaces=2)
        boxes.append(flonum)
        flonum_id = flonum["box"]["id"]
        
        # ローパスフィルター
        lores = self.create_box("newobj", "lores~ 1000 0.5", [100, 250, 100, 22], 3, 1)
        boxes.append(lores)
        lores_id = lores["box"]["id"]
        
        # ゲイン調整
        gain = self.create_box("gain~", "", [100, 350, 48, 140], 2, 2,
                              orientation=2, parameter_enable=1)
        boxes.append(gain)
        gain_id = gain["box"]["id"]
        
        # DAC（オーディオ出力）
        dac = self.create_box("ezdac~", "", [100, 500, 61, 61], 2, 0)
        boxes.append(dac)
        dac_id = dac["box"]["id"]
        
        # 接続を作成
        # スライダー → 数値表示
        lines.append(self.create_connection(slider_id, 0, flonum_id, 0))
        # 数値表示 → フィルター周波数
        lines.append(self.create_connection(flonum_id, 0, lores_id, 1))
        # オーディオ入力 → フィルター
        lines.append(self.create_connection(adc_id, 0, lores_id, 0))
        # フィルター → ゲイン
        lines.append(self.create_connection(lores_id, 0, gain_id, 0))
        # ゲイン → 出力（左）
        lines.append(self.create_connection(gain_id, 0, dac_id, 0))
        # ゲイン → 出力（右）
        lines.append(self.create_connection(gain_id, 0, dac_id, 1))
        
        patch["patcher"]["boxes"] = boxes
        patch["patcher"]["lines"] = lines
        
        return patch
    
    def generate_reverb_delay_synth(self) -> Dict[str, Any]:
        """リバーブ＋ディレイ付きシンセサイザーを生成"""
        patch = self.create_empty_patch()
        boxes = []
        lines = []
        
        # MIDIノート入力
        notein = self.create_box("notein", "", [50, 50, 50, 22], 1, 3)
        boxes.append(notein)
        notein_id = notein["box"]["id"]
        
        # mtof（MIDI→周波数変換）
        mtof = self.create_box("newobj", "mtof", [50, 100, 34, 22], 1, 1)
        boxes.append(mtof)
        mtof_id = mtof["box"]["id"]
        
        # オシレーター
        cycle = self.create_box("newobj", "cycle~", [50, 150, 43, 22], 2, 1)
        boxes.append(cycle)
        cycle_id = cycle["box"]["id"]
        
        # ベロシティ→ゲイン変換
        scale = self.create_box("newobj", "scale 0 127 0. 0.8", [150, 100, 117, 22], 6, 1)
        boxes.append(scale)
        scale_id = scale["box"]["id"]
        
        # 乗算（ゲイン調整）
        mult = self.create_box("newobj", "*~ 0.5", [100, 200, 39, 22], 2, 1)
        boxes.append(mult)
        mult_id = mult["box"]["id"]
        
        # ディレイライン
        tapin = self.create_box("newobj", "tapin~ 1000", [100, 250, 72, 22], 1, 1)
        boxes.append(tapin)
        tapin_id = tapin["box"]["id"]
        
        tapout = self.create_box("newobj", "tapout~ 250 500", [100, 300, 97, 22], 1, 2)
        boxes.append(tapout)
        tapout_id = tapout["box"]["id"]
        
        # フィードバック用の乗算
        feedback = self.create_box("newobj", "*~ 0.4", [200, 350, 39, 22], 2, 1)
        boxes.append(feedback)
        feedback_id = feedback["box"]["id"]
        
        # ミックス
        mix = self.create_box("newobj", "+~", [100, 400, 29, 22], 2, 1)
        boxes.append(mix)
        mix_id = mix["box"]["id"]
        
        # 出力ゲイン
        gain = self.create_box("gain~", "", [100, 450, 48, 140], 2, 2)
        boxes.append(gain)
        gain_id = gain["box"]["id"]
        
        # DAC
        dac = self.create_box("ezdac~", "", [100, 600, 61, 61], 2, 0)
        boxes.append(dac)
        dac_id = dac["box"]["id"]
        
        # 接続を作成
        # MIDI → 周波数
        lines.append(self.create_connection(notein_id, 0, mtof_id, 0))
        lines.append(self.create_connection(mtof_id, 0, cycle_id, 0))
        
        # ベロシティ → ゲイン
        lines.append(self.create_connection(notein_id, 1, scale_id, 0))
        lines.append(self.create_connection(scale_id, 0, mult_id, 1))
        
        # オシレーター → ゲイン調整
        lines.append(self.create_connection(cycle_id, 0, mult_id, 0))
        
        # ディレイ処理
        lines.append(self.create_connection(mult_id, 0, tapin_id, 0))
        lines.append(self.create_connection(tapin_id, 0, tapout_id, 0))
        lines.append(self.create_connection(tapout_id, 0, feedback_id, 0))
        lines.append(self.create_connection(feedback_id, 0, tapin_id, 0))
        
        # ミックス
        lines.append(self.create_connection(mult_id, 0, mix_id, 0))
        lines.append(self.create_connection(tapout_id, 1, mix_id, 1))
        
        # 出力
        lines.append(self.create_connection(mix_id, 0, gain_id, 0))
        lines.append(self.create_connection(gain_id, 0, dac_id, 0))
        lines.append(self.create_connection(gain_id, 0, dac_id, 1))
        
        patch["patcher"]["boxes"] = boxes
        patch["patcher"]["lines"] = lines
        
        return patch
    
    def generate_patch(self, description: str) -> Dict[str, Any]:
        """説明に基づいてパッチを生成"""
        description_lower = description.lower()
        
        if "filter" in description_lower:
            logger.info("フィルターパッチを生成します")
            return self.generate_filter_patch()
        elif "synth" in description_lower or "reverb" in description_lower:
            logger.info("リバーブ付きシンセサイザーを生成します")
            return self.generate_reverb_delay_synth()
        else:
            # デフォルトはフィルター
            logger.info("デフォルトパッチ（フィルター）を生成します")
            return self.generate_filter_patch()


def main():
    parser = argparse.ArgumentParser(description='シンプルなMax/MSPパッチ生成')
    parser.add_argument('description', type=str, 
                       help='パッチの説明')
    parser.add_argument('--output', type=str, required=True,
                       help='出力ファイルパス')
    
    args = parser.parse_args()
    
    # ジェネレーターを初期化
    generator = SimplePatchGenerator()
    
    # パッチを生成
    patch = generator.generate_patch(args.description)
    
    # ファイルに保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(patch, f, indent=2)
    
    logger.info(f"パッチを生成しました: {output_path}")
    logger.info(f"オブジェクト数: {len(patch['patcher']['boxes'])}")
    logger.info(f"接続数: {len(patch['patcher']['lines'])}")


if __name__ == "__main__":
    main()