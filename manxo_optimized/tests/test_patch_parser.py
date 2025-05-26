#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_patch_parser.py - パッチパーサーのテスト
"""

import sys
import os
import unittest
import json
import tempfile
from typing import Dict, Any

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.patch_parser import EnhancedPatchParser, parse_maxpat_file
from src.box_types import (
    BoxType, MaxBox, ObjectBox, MessageBox, NumberBox, FloatBox, CommentBox,
    UIControlBox, LiveUIBox, MSPBox, JitterBox, JavaScriptBox, 
    PatcherBox, GenBox, BPatcherBox
)

# テスト用のサンプルパッチデータ
# このデータには様々なボックスタイプと接続が含まれています
# パーサーの基本機能とボックスタイプ、接続タイプの推論をテストするのに使用します
SAMPLE_PATCH = {
    "patcher": {
        "fileversion": 1,
        "appversion": {"major": 8, "minor": 3, "revision": 1, "architecture": "x64"},
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
        "description": "",
        "digest": "",
        "tags": "",
        "style": "",
        "subpatcher_template": "",
        "assistshowspatchername": 0,
        "boxes": {
            "obj-1": {
                "box": {
                    "maxclass": "newobj",
                    "text": "metro 500",
                    "patching_rect": [100.0, 100.0, 66.0, 22.0],
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": ["bang"],
                    "id": "obj-1"
                }
            },
            "obj-2": {
                "box": {
                    "maxclass": "message",
                    "text": "bang",
                    "patching_rect": [100.0, 150.0, 35.0, 22.0],
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": [""],
                    "id": "obj-2"
                }
            },
            "obj-3": {
                "box": {
                    "maxclass": "number",
                    "patching_rect": [150.0, 150.0, 50.0, 22.0],
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": ["int", "bang"],
                    "parameter_enable": 0,
                    "id": "obj-3"
                }
            },
            "obj-4": {
                "box": {
                    "maxclass": "flonum",
                    "patching_rect": [200.0, 150.0, 50.0, 22.0],
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": ["float", "bang"],
                    "parameter_enable": 0,
                    "id": "obj-4"
                }
            },
            "obj-5": {
                "box": {
                    "maxclass": "comment",
                    "text": "Test comment",
                    "patching_rect": [100.0, 50.0, 150.0, 20.0],
                    "id": "obj-5"
                }
            },
            "obj-6": {
                "box": {
                    "maxclass": "newobj",
                    "text": "p testSubpatch",
                    "patching_rect": [250.0, 150.0, 85.0, 22.0],
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": ["bang"],
                    "id": "obj-6",
                    "patcher": {
                        "fileversion": 1,
                        "appversion": {"major": 8, "minor": 3, "revision": 1, "architecture": "x64"},
                        "classnamespace": "box",
                        "rect": [0, 0, 300, 300],
                        "bglocked": 0,
                        "openinpresentation": 0,
                        "boxes": {
                            "obj-1": {
                                "box": {
                                    "maxclass": "inlet",
                                    "patching_rect": [100.0, 50.0, 30.0, 30.0],
                                    "numinlets": 0,
                                    "numoutlets": 1,
                                    "outlettype": ["bang"],
                                    "id": "obj-1"
                                }
                            },
                            "obj-2": {
                                "box": {
                                    "maxclass": "outlet",
                                    "patching_rect": [100.0, 250.0, 30.0, 30.0],
                                    "numinlets": 1,
                                    "numoutlets": 0,
                                    "id": "obj-2"
                                }
                            },
                            "obj-3": {
                                "box": {
                                    "maxclass": "newobj",
                                    "text": "delay 100",
                                    "patching_rect": [100.0, 150.0, 63.0, 22.0],
                                    "numinlets": 2,
                                    "numoutlets": 1,
                                    "outlettype": ["bang"],
                                    "id": "obj-3"
                                }
                            }
                        },
                        "lines": {
                            "line-1": {
                                "patchline": {
                                    "source": ["obj-1", 0],
                                    "destination": ["obj-3", 0],
                                    "id": "line-1"
                                }
                            },
                            "line-2": {
                                "patchline": {
                                    "source": ["obj-3", 0],
                                    "destination": ["obj-2", 0],
                                    "id": "line-2"
                                }
                            }
                        }
                    }
                }
            },
            "obj-7": {
                "box": {
                    "maxclass": "newobj",
                    "text": "cycle~ 440",
                    "patching_rect": [350.0, 150.0, 68.0, 22.0],
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": ["signal"],
                    "id": "obj-7"
                }
            },
            "obj-8": {
                "box": {
                    "maxclass": "toggle",
                    "patching_rect": [100.0, 200.0, 24.0, 24.0],
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": ["int"],
                    "parameter_enable": 0,
                    "id": "obj-8"
                }
            }
        },
        "lines": {
            "line-1": {
                "patchline": {
                    "source": ["obj-1", 0],
                    "destination": ["obj-2", 0],
                    "id": "line-1"
                }
            },
            "line-2": {
                "patchline": {
                    "source": ["obj-8", 0],
                    "destination": ["obj-1", 0],
                    "id": "line-2"
                }
            },
            "line-3": {
                "patchline": {
                    "source": ["obj-3", 0],
                    "destination": ["obj-7", 0],
                    "id": "line-3"
                }
            }
        }
    }
}


class TestPatchParser(unittest.TestCase):
    """パッチパーサーのテストケース"""
    
    def setUp(self):
        """テスト前の準備"""
        self.parser = EnhancedPatchParser()
        
        # テスト用の一時ファイルを作成
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.maxpat')
        with open(self.temp_file.name, 'w', encoding='utf-8') as f:
            json.dump(SAMPLE_PATCH, f)
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        # 一時ファイルを削除
        os.unlink(self.temp_file.name)
    
    def test_parse_patch_data(self):
        """パッチデータのパース処理テスト"""
        boxes, connections, metadata = self.parser.parse_patch_data(SAMPLE_PATCH)
        
        # ボックス数の確認
        self.assertEqual(len(boxes), 8)
        
        # 接続数の確認
        self.assertEqual(len(connections), 3)
        
        # メタデータの確認
        self.assertEqual(metadata["box_count"], 8)
        self.assertEqual(metadata["connection_count"], 3)
        
        # ボックスタイプの統計確認
        self.assertEqual(metadata["box_types"]["newobj"], 3)
        self.assertEqual(metadata["box_types"]["message"], 1)
        self.assertEqual(metadata["box_types"]["number"], 1)
        self.assertEqual(metadata["box_types"]["flonum"], 1)
        self.assertEqual(metadata["box_types"]["comment"], 1)
        self.assertEqual(metadata["box_types"]["toggle"], 1)
    
    def test_parse_file(self):
        """ファイルからのパース処理テスト"""
        boxes, connections, metadata = self.parser.parse_file(self.temp_file.name)
        
        # ボックス数の確認
        self.assertEqual(len(boxes), 8)
        
        # 接続数の確認
        self.assertEqual(len(connections), 3)
        
        # メタデータの確認
        self.assertEqual(metadata["source_file"], self.temp_file.name)
    
    def test_box_type_parsing(self):
        """ボックスタイプの正しいパースを確認"""
        boxes, _, _ = self.parser.parse_patch_data(SAMPLE_PATCH)
        
        # 代表的なボックスタイプの確認
        self.assertIsInstance(boxes["obj-1"], ObjectBox)
        self.assertIsInstance(boxes["obj-2"], MessageBox)
        self.assertIsInstance(boxes["obj-3"], NumberBox)
        self.assertIsInstance(boxes["obj-4"], FloatBox)
        self.assertIsInstance(boxes["obj-5"], CommentBox)
        self.assertIsInstance(boxes["obj-8"], UIControlBox)
        
        # ObjectBoxの属性確認
        metro_box = boxes["obj-1"]
        self.assertEqual(metro_box.object_name, "metro")
        self.assertEqual(metro_box.arguments, ["500"])
        
        # MSPオブジェクト確認
        cycle_box = boxes["obj-7"]
        self.assertTrue(cycle_box.is_msp_object)
        self.assertEqual(cycle_box.object_name, "cycle~")
        
        # UIControlBoxの属性確認
        toggle_box = boxes["obj-8"]
        self.assertEqual(toggle_box.box_type, BoxType.TOGGLE)
        self.assertTrue(toggle_box.box_type.is_ui_element)
        
        # ポート情報の確認
        self.assertEqual(toggle_box.numinlets, 1)
        self.assertEqual(toggle_box.numoutlets, 1)
        self.assertEqual(toggle_box.outlets[0]["type"], "int")
    
    def test_subpatcher_parsing(self):
        """サブパッチャーの解析テスト"""
        boxes, _, _ = self.parser.parse_patch_data(SAMPLE_PATCH)
        
        # サブパッチャーボックスの確認
        subpatcher_box = boxes["obj-6"]
        self.assertTrue(subpatcher_box.is_subpatcher)
        self.assertTrue(subpatcher_box.has_subpatch)
        
        # サブパッチャー内のボックス確認
        self.assertIsNotNone(subpatcher_box.subpatch)
        self.assertEqual(len(subpatcher_box.subpatch), 3)
        
        # サブパッチャー内の特定のボックスを確認
        sub_boxes = subpatcher_box.subpatch
        self.assertTrue("obj-1" in sub_boxes)  # inlet
        self.assertTrue("obj-3" in sub_boxes)  # delay
        
        # 親子関係の確認
        for sub_box in sub_boxes.values():
            self.assertEqual(sub_box.parent_id, "obj-6")
        
        # サブパッチャーのポート設定確認
        # パッチャー内のinlet/outletオブジェクトに基づいて適切なポート数が設定されている
        self.assertEqual(subpatcher_box.numinlets, 1)  # inlet 1つ
        self.assertEqual(subpatcher_box.numoutlets, 1)  # outlet 1つ
        
        # inlet/outletオブジェクトの識別確認
        inlet_box = sub_boxes["obj-1"]
        self.assertEqual(inlet_box.object_name, "inlet")
        self.assertEqual(inlet_box.numinlets, 0)
        self.assertEqual(inlet_box.numoutlets, 1)
    
    def test_hierarchy_levels(self):
        """階層レベルの設定テスト"""
        boxes, _, _ = self.parser.parse_patch_data(SAMPLE_PATCH)
        
        # メインパッチのボックスは階層レベル0
        self.assertEqual(boxes["obj-1"].hierarchy_level, 0)
        
        # サブパッチャー内のボックスは階層レベル1
        subpatcher_box = boxes["obj-6"]
        for sub_box in subpatcher_box.subpatch.values():
            self.assertEqual(sub_box.hierarchy_level, 1)
    
    def test_connection_type_inference(self):
        """接続タイプの推論テスト"""
        boxes, connections, _ = self.parser.parse_patch_data(SAMPLE_PATCH)
        
        # control接続の確認
        toggle_to_metro = next(conn for conn in connections 
                             if conn["source_id"] == "obj-8" and conn["destination_id"] == "obj-1")
        self.assertEqual(toggle_to_metro["type"], "control")
        
        # signal接続の確認（number -> cycle~は制御接続だがcycle~がMSPオブジェクト）
        num_to_cycle = next(conn for conn in connections 
                           if conn["source_id"] == "obj-3" and conn["destination_id"] == "obj-7")
        self.assertEqual(num_to_cycle["type"], "control")  # cycle~への周波数設定は制御接続
        
        # メッセージボックスのテスト
        message_box = boxes["obj-2"]
        self.assertEqual(message_box.message_text, "bang")
        
        # ボックスタイププロパティのテスト
        cycle_box = boxes["obj-7"]
        self.assertTrue(cycle_box.is_msp_object)  # ObjectBoxのプロパティ
        
        # 入出力ポート数のテスト
        self.assertEqual(boxes["obj-1"].numinlets, 1)  # metro
        self.assertEqual(boxes["obj-1"].numoutlets, 1)
        
        self.assertEqual(boxes["obj-2"].numinlets, 2)  # message
        self.assertEqual(boxes["obj-2"].numoutlets, 1)
        
        self.assertEqual(boxes["obj-3"].numinlets, 1)  # number
        self.assertEqual(boxes["obj-3"].numoutlets, 2)
        
        self.assertEqual(boxes["obj-4"].numinlets, 1)  # flonum
        self.assertEqual(boxes["obj-4"].numoutlets, 2)
    
    def test_utility_function(self):
        """ユーティリティ関数のテスト"""
        boxes, connections, metadata = parse_maxpat_file(self.temp_file.name)
        
        # 基本的な結果確認
        self.assertEqual(len(boxes), 8)
        self.assertEqual(len(connections), 3)
        self.assertEqual(metadata["source_file"], self.temp_file.name)
    
    def test_message_box_features(self):
        """メッセージボックスの特徴的な機能テスト"""
        boxes, _, _ = self.parser.parse_patch_data(SAMPLE_PATCH)
        
        message_box = boxes["obj-2"]
        self.assertIsInstance(message_box, MessageBox)
        
        # 基本的なプロパティ
        self.assertEqual(message_box.message_text, "bang")
        
        # メッセージ解析機能テスト
        # bangは変数を含まないので空のはず
        self.assertEqual(len(message_box.variables), 0)
        
        # 高度なメッセージ解析機能のテスト
        # テスト用に複雑なメッセージテキストを持つオブジェクトを作成
        complex_message = MessageBox(
            "obj-test", [100, 100], [100, 22],
            "set $1 $2; clear; append $1-$2, $i3, $f4; test $s5"
        )
        
        # 変数の検出
        self.assertEqual(set(complex_message.variables), {1, 2, 3, 4, 5})
        
        # 変数型の検出
        self.assertEqual(complex_message.variable_types[1], "any")
        self.assertEqual(complex_message.variable_types[2], "any")
        self.assertEqual(complex_message.variable_types[3], "int")
        self.assertEqual(complex_message.variable_types[4], "float")
        self.assertEqual(complex_message.variable_types[5], "symbol")
        
        # 転送先の検出
        self.assertTrue(complex_message.has_forward)
        self.assertEqual(len(complex_message.forwarding_targets), 2)
        self.assertEqual(complex_message.forwarding_targets[0]["target"], "clear")
        
        # 特殊メッセージの検出
        self.assertIn("set", complex_message.special_messages)
        self.assertIn("clear", complex_message.special_messages)
        self.assertIn("append", complex_message.special_messages)
        
        # メッセージ解決機能のテスト
        args = [10, 20, 30, 40.5, "hello"]
        resolved = complex_message.resolve_message(args)
        self.assertIn("set 10 20", resolved)
        self.assertIn("append 10-20, 30, 40.5", resolved)
        self.assertIn("test hello", resolved)


if __name__ == "__main__":
    unittest.main()