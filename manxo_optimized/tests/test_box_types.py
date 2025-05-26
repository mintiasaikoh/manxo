#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_box_types.py - ボックスタイプクラスのテスト
"""

import sys
import os
import unittest
import json
from typing import Dict, Any

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.box_types import (
    BoxType, MaxBox, ObjectBox, MessageBox, NumberBox, 
    FloatBox, CommentBox, UIControlBox, BPatcherBox, GenericBox
)

class TestBoxTypes(unittest.TestCase):
    """ボックスタイプクラスのテストケース"""
    
    def test_box_type_enum(self):
        """BoxType列挙型のテスト"""
        # 列挙型の値が正しく設定されているか
        self.assertEqual(BoxType.OBJECT_BOX.value, "newobj")
        self.assertEqual(BoxType.MESSAGE_BOX.value, "message")
        self.assertEqual(BoxType.NUMBER_BOX.value, "number")
        
        # from_maxclassメソッドのテスト
        self.assertEqual(BoxType.from_maxclass("newobj"), BoxType.OBJECT_BOX)
        self.assertEqual(BoxType.from_maxclass("message"), BoxType.MESSAGE_BOX)
        
        # 不明なmaxclassの場合のデフォルト値
        self.assertEqual(BoxType.from_maxclass("unknown"), BoxType.OBJECT_BOX)
    
    def test_max_box_base(self):
        """MaxBox基底クラスのテスト"""
        box = MaxBox("obj-1", BoxType.OBJECT_BOX, [10, 20], [30, 40])
        
        # 基本的なプロパティの初期化を検証
        self.assertEqual(box.id, "obj-1")
        self.assertEqual(box.box_type, BoxType.OBJECT_BOX)
        self.assertEqual(box.position, [10, 20])
        self.assertEqual(box.size, [30, 40])
        self.assertEqual(box.inlets, [])
        self.assertEqual(box.outlets, [])
        self.assertEqual(box.numinlets, 0)
        self.assertEqual(box.numoutlets, 0)
        
        # インレット/アウトレット追加メソッドのテスト
        box.add_inlet(0, "float", "第1インレット", False)
        box.add_outlet(0, "float", "出力", False)
        
        self.assertEqual(len(box.inlets), 1)
        self.assertEqual(len(box.outlets), 1)
        self.assertEqual(box.numinlets, 1)
        self.assertEqual(box.numoutlets, 1)
        
        self.assertEqual(box.inlets[0]["type"], "float")
        self.assertEqual(box.outlets[0]["description"], "出力")
    
    def test_object_box(self):
        """ObjectBoxクラスのテスト"""
        # 基本的なオブジェクトのテスト
        obj_box = ObjectBox("obj-1", [10, 20], [30, 40], "metro 500")
        
        self.assertEqual(obj_box.box_type, BoxType.OBJECT_BOX)
        self.assertEqual(obj_box.object_name, "metro")
        self.assertEqual(obj_box.arguments_text, "500")
        self.assertEqual(obj_box.arguments, ["500"])
        
        # MSPオブジェクト判定のテスト
        msp_box = ObjectBox("obj-2", [10, 20], [30, 40], "cycle~ 440")
        
        self.assertTrue(msp_box.is_msp_object)
        self.assertFalse(msp_box.is_jitter_object)
        
        # Jitterオブジェクト判定のテスト
        jit_box = ObjectBox("obj-3", [10, 20], [30, 40], "jit.matrix 4 char 320 240")
        
        self.assertTrue(jit_box.is_jitter_object)
        self.assertFalse(jit_box.is_msp_object)
        
        # サブパッチャー判定のテスト
        sub_box = ObjectBox("obj-4", [10, 20], [30, 40], "p mySubpatch")
        
        self.assertTrue(sub_box.is_subpatcher)
        
        # 引数なしのオブジェクトのテスト
        empty_box = ObjectBox("obj-5", [10, 20], [30, 40], "print")
        
        self.assertEqual(empty_box.object_name, "print")
        self.assertEqual(empty_box.arguments, [])
    
    def test_message_box(self):
        """MessageBoxクラスのテスト"""
        # 基本的なメッセージボックスのテスト
        msg_box = MessageBox("obj-1", [10, 20], [30, 40], "bang")
        
        self.assertEqual(msg_box.box_type, BoxType.MESSAGE_BOX)
        self.assertEqual(msg_box.message_text, "bang")
        self.assertEqual(msg_box.variables, [])
        self.assertEqual(msg_box.numinlets, 2)  # メッセージボックスは常に2インレット
        self.assertEqual(msg_box.numoutlets, 1)
        
        # 変数を含むメッセージボックスのテスト
        var_msg_box = MessageBox("obj-2", [10, 20], [30, 40], "set $1 $2")
        
        self.assertEqual(var_msg_box.variables, [1, 2])
        
        # 送信先指定を含むメッセージボックスのテスト
        forward_msg_box = MessageBox("obj-3", [10, 20], [30, 40], "hello; world bang")
        
        self.assertTrue(forward_msg_box.has_forward)
        self.assertEqual(forward_msg_box.forwarding_destinations, ["world"])
    
    def test_number_boxes(self):
        """数値ボックスクラスのテスト"""
        # 整数ボックスのテスト
        int_box = NumberBox("obj-1", [10, 20], [30, 40])
        
        self.assertEqual(int_box.box_type, BoxType.NUMBER_BOX)
        self.assertEqual(int_box.value, 0)
        self.assertEqual(int_box.numinlets, 1)
        self.assertEqual(int_box.numoutlets, 2)  # 整数出力 + bang出力
        
        # 浮動小数点ボックスのテスト
        float_box = FloatBox("obj-2", [10, 20], [30, 40])
        
        self.assertEqual(float_box.box_type, BoxType.FLOAT_BOX)
        self.assertEqual(float_box.value, 0.0)
        self.assertEqual(float_box.numinlets, 1)
        self.assertEqual(float_box.numoutlets, 2)  # 浮動小数点出力 + bang出力
    
    def test_comment_box(self):
        """CommentBoxクラスのテスト"""
        comment = CommentBox("obj-1", [10, 20], [100, 40], "This is a test comment")
        
        self.assertEqual(comment.box_type, BoxType.COMMENT_BOX)
        self.assertEqual(comment.comment_text, "This is a test comment")
        self.assertEqual(comment.numinlets, 0)  # コメントはインレット/アウトレットを持たない
        self.assertEqual(comment.numoutlets, 0)
    
    def test_ui_control_box(self):
        """UIControlBoxクラスのテスト"""
        # トグルボックスのテスト
        toggle = UIControlBox("obj-1", BoxType.TOGGLE, [10, 20], [20, 20])
        
        self.assertEqual(toggle.box_type, BoxType.TOGGLE)
        self.assertEqual(toggle.numinlets, 1)
        self.assertEqual(toggle.numoutlets, 1)
        self.assertEqual(toggle.outlets[0]["type"], "int")
        
        # スライダーボックスのテスト
        slider = UIControlBox("obj-2", BoxType.SLIDER, [10, 50], [20, 100])
        
        self.assertEqual(slider.box_type, BoxType.SLIDER)
        self.assertEqual(slider.numinlets, 1)
        self.assertEqual(slider.numoutlets, 1)
        self.assertEqual(slider.outlets[0]["type"], "float")
        
        # KSliderボックスのテスト
        kslider = UIControlBox("obj-3", BoxType.KSLIDER, [10, 160], [100, 40])
        
        self.assertEqual(kslider.box_type, BoxType.KSLIDER)
        self.assertEqual(kslider.numinlets, 1)
        self.assertEqual(kslider.numoutlets, 2)  # KSliderは音程と速度の2つのアウトレット
    
    def test_bpatcher_box(self):
        """BPatcherBoxクラスのテスト"""
        bpatcher = BPatcherBox("obj-1", [10, 20], [200, 100], "myPatch.maxpat")
        
        self.assertEqual(bpatcher.box_type, BoxType.BPATCHER)
        self.assertEqual(bpatcher.patch_name, "myPatch.maxpat")
        self.assertEqual(bpatcher.bgmode, 0)
        self.assertEqual(bpatcher.border, 0)
    
    def test_generic_box(self):
        """GenericBoxクラスのテスト"""
        raw_data = {
            "maxclass": "umenu",
            "numinlets": 1,
            "numoutlets": 3,
            "outlettype": ["int", "bang", "clear"]
        }
        
        generic = GenericBox("obj-1", BoxType.UMENU, [10, 20], [100, 20], raw_data)
        
        self.assertEqual(generic.box_type, BoxType.UMENU)
        self.assertEqual(generic.numinlets, 1)
        self.assertEqual(generic.numoutlets, 3)
        self.assertEqual(len(generic.outlets), 3)
        self.assertEqual(generic.outlets[0]["type"], "int")
        self.assertEqual(generic.outlets[1]["type"], "bang")
        self.assertEqual(generic.outlets[2]["type"], "clear")
    
    def test_from_json_box(self):
        """from_json_boxスタティックメソッドのテスト"""
        # オブジェクトボックスのJSONデータ
        obj_data = {
            "maxclass": "newobj",
            "patching_rect": [100.0, 100.0, 66.0, 22.0],
            "text": "metro 500",
            "numinlets": 1,
            "numoutlets": 1,
            "outlettype": ["bang"]
        }
        
        box = MaxBox.from_json_box("obj-12", obj_data)
        
        self.assertIsInstance(box, ObjectBox)
        self.assertEqual(box.object_name, "metro")
        self.assertEqual(box.arguments, ["500"])
        
        # メッセージボックスのJSONデータ
        msg_data = {
            "maxclass": "message",
            "patching_rect": [50.0, 100.0, 35.0, 22.0],
            "text": "start",
            "numinlets": 2,
            "numoutlets": 1
        }
        
        box = MaxBox.from_json_box("obj-5", msg_data)
        
        self.assertIsInstance(box, MessageBox)
        self.assertEqual(box.message_text, "start")
        
        # コメントボックスのJSONデータ
        comment_data = {
            "maxclass": "comment",
            "patching_rect": [10.0, 10.0, 200.0, 20.0],
            "text": "This is a comment"
        }
        
        box = MaxBox.from_json_box("obj-2", comment_data)
        
        self.assertIsInstance(box, CommentBox)
        self.assertEqual(box.comment_text, "This is a comment")
    
    def test_to_dict(self):
        """to_dictメソッドのテスト"""
        # ObjectBoxのto_dictテスト
        obj_box = ObjectBox("obj-1", [10, 20], [30, 40], "metro 500")
        obj_dict = obj_box.to_dict()
        
        self.assertEqual(obj_dict["id"], "obj-1")
        self.assertEqual(obj_dict["type"], "newobj")
        self.assertEqual(obj_dict["object_name"], "metro")
        self.assertEqual(obj_dict["arguments"], ["500"])
        
        # MessageBoxのto_dictテスト
        msg_box = MessageBox("obj-2", [10, 20], [30, 40], "bang")
        msg_dict = msg_box.to_dict()
        
        self.assertEqual(msg_dict["id"], "obj-2")
        self.assertEqual(msg_dict["type"], "message")
        self.assertEqual(msg_dict["message_text"], "bang")
        
        # 辞書がJSON化可能かチェック
        json_str = json.dumps(obj_dict)
        self.assertTrue(isinstance(json_str, str))
        
        json_str = json.dumps(msg_dict)
        self.assertTrue(isinstance(json_str, str))


if __name__ == "__main__":
    unittest.main()