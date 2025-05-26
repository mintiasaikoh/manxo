#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_parser.py - Max/MSPパッチファイルの強化パーサー

このモジュールは、Max/MSPパッチファイル（.maxpat）を解析し、
ボックスタイプを適切に処理するための機能を提供します。
従来のパーサーと異なり、各ボックスタイプの特性に応じた
最適な処理を行います。
"""

import json
import os
from typing import Dict, List, Any, Optional, Union, Tuple
import logging

from src.box_types import (
    BoxType, MaxBox, ObjectBox, MessageBox, NumberBox, 
    FloatBox, CommentBox, UIControlBox, BPatcherBox, GenericBox
)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedPatchParser:
    """強化されたMax/MSPパッチパーサー
    
    ボックスタイプを考慮した高度なパッチ解析を提供します。
    """
    
    def __init__(self):
        """パーサーの初期化"""
        self.parsed_boxes = {}  # ID → ボックスオブジェクト
        self.connections = []   # 接続情報
        self.metadata = {}      # メタデータ
    
    def parse_file(self, file_path: str) -> Tuple[Dict[str, MaxBox], List[Dict[str, Any]], Dict[str, Any]]:
        """ファイルからパッチを解析
        
        Args:
            file_path: .maxpatファイルのパス
            
        Returns:
            (boxes, connections, metadata): ボックス、接続、メタデータの情報
        """
        if not os.path.exists(file_path):
            logger.error(f"ファイルが存在しません: {file_path}")
            return {}, [], {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                patch_data = json.load(f)
            
            return self.parse_patch_data(patch_data, file_path)
            
        except json.JSONDecodeError:
            logger.error(f"JSONの解析に失敗しました: {file_path}")
        except Exception as e:
            logger.error(f"パッチの解析中にエラーが発生しました: {e}")
        
        return {}, [], {}
    
    def parse_patch_data(
        self, 
        patch_data: Dict[str, Any], 
        source_path: Optional[str] = None
    ) -> Tuple[Dict[str, MaxBox], List[Dict[str, Any]], Dict[str, Any]]:
        """パッチデータを解析
        
        Args:
            patch_data: パッチデータの辞書
            source_path: データのソースパス（省略可）
            
        Returns:
            (boxes, connections, metadata): ボックス、接続、メタデータの情報
        """
        # 初期化
        self.parsed_boxes = {}
        self.connections = []
        self.metadata = {}
        
        # メタデータの抽出
        self._extract_metadata(patch_data, source_path)
        
        # ボックスの抽出
        self._extract_boxes(patch_data)
        
        # 接続情報の抽出
        self._extract_connections(patch_data)
        
        # サブパッチャーの処理
        self._process_subpatchers(patch_data)
        
        # 階層情報の設定
        self._set_hierarchy_levels()
        
        return self.parsed_boxes, self.connections, self.metadata
    
    def _extract_metadata(self, patch_data: Dict[str, Any], source_path: Optional[str] = None) -> None:
        """パッチからメタデータを抽出"""
        patcher = patch_data.get("patcher", {})
        
        self.metadata = {
            "source_file": source_path,
            "max_version": patcher.get("maxversion", "unknown"),
            "appversion": patcher.get("appversion", "unknown"),
            "rect": patcher.get("rect", [0, 0, 0, 0]),
            "openrect": patcher.get("openrect", [0, 0, 0, 0]),
            "box_count": 0,
            "connection_count": 0,
            "box_types": {}
        }
    
    def _extract_boxes(self, patch_data: Dict[str, Any]) -> None:
        """パッチからボックスを抽出"""
        patcher = patch_data.get("patcher", {})
        boxes_data = patcher.get("boxes", {})
        
        # バージョンに応じた処理
        if isinstance(boxes_data, list):
            # 古い形式: リスト形式
            for box in boxes_data:
                if "box" in box:
                    box_id = box["box"].get("id", f"obj-{len(self.parsed_boxes)}")
                    self._process_box(box_id, box["box"])
        else:
            # 新しい形式: 辞書形式
            for box_id, box_data in boxes_data.items():
                if "box" in box_data:
                    self._process_box(box_id, box_data["box"])
                else:
                    # 直接boxデータがある場合
                    self._process_box(box_id, box_data)
        
        # メタデータの更新
        self.metadata["box_count"] = len(self.parsed_boxes)
        
        # ボックスタイプの統計
        box_types = {}
        for box in self.parsed_boxes.values():
            box_type = box.box_type.value
            if box_type not in box_types:
                box_types[box_type] = 0
            box_types[box_type] += 1
        
        self.metadata["box_types"] = box_types
    
    def _process_box(self, box_id: str, box_data: Dict[str, Any]) -> None:
        """個別のボックスを処理"""
        # MaxBox.from_json_boxメソッドを使用して適切なボックスオブジェクトを作成
        box = MaxBox.from_json_box(box_id, box_data)
        
        # 作成したボックスを辞書に保存
        self.parsed_boxes[box_id] = box
    
    def _extract_connections(self, patch_data: Dict[str, Any]) -> None:
        """パッチから接続情報を抽出"""
        patcher = patch_data.get("patcher", {})
        lines_data = patcher.get("lines", {})
        
        # バージョンに応じた処理
        if isinstance(lines_data, list):
            # 古い形式: リスト形式
            for line in lines_data:
                if "patchline" in line:
                    self._process_connection(line["patchline"])
        else:
            # 新しい形式: 辞書形式
            for line_id, line_data in lines_data.items():
                if "patchline" in line_data:
                    self._process_connection(line_data["patchline"])
                else:
                    # 直接patchlineデータがある場合
                    self._process_connection(line_data)
        
        # メタデータの更新
        self.metadata["connection_count"] = len(self.connections)
    
    def _process_connection(self, connection_data: Dict[str, Any]) -> None:
        """個別の接続情報を処理"""
        # 接続の基本情報を抽出
        source = connection_data.get("source", "unknown")
        destination = connection_data.get("destination", "unknown")
        
        # source/destinationがリスト形式の場合（Max 8以降）
        if isinstance(source, list) and len(source) >= 1:
            source_id = source[0]
            source_port = source[1] if len(source) > 1 else 0
        else:
            # 旧形式
            source_id = source
            source_port = connection_data.get("sourceoutletindex", 0)
            if isinstance(source_port, list) and len(source_port) > 0:
                source_port = source_port[0]
        
        if isinstance(destination, list) and len(destination) >= 1:
            dest_id = destination[0]
            dest_port = destination[1] if len(destination) > 1 else 0
        else:
            # 旧形式
            dest_id = destination
            dest_port = connection_data.get("destinationinletindex", 0)
            if isinstance(dest_port, list) and len(dest_port) > 0:
                dest_port = dest_port[0]
        
        # 隠し接続フラグ
        hidden = connection_data.get("hidden", 0)
        
        # 接続情報の辞書を作成
        connection = {
            "source_id": source_id,
            "source_port": int(source_port),
            "destination_id": dest_id,
            "destination_port": int(dest_port),
            "hidden": int(hidden) == 1
        }
        
        # 接続タイプの推定
        connection["type"] = self._infer_connection_type(
            connection["source_id"], 
            connection["source_port"],
            connection["destination_id"], 
            connection["destination_port"]
        )
        
        # 接続リストに追加
        self.connections.append(connection)
    
    def _infer_connection_type(
        self, 
        source_id: str, 
        source_port: int,
        dest_id: str, 
        dest_port: int
    ) -> str:
        """接続タイプを推定"""
        source_box = self.parsed_boxes.get(source_id)
        dest_box = self.parsed_boxes.get(dest_id)
        
        # オブジェクトが見つからない場合はデフォルト
        if not source_box or not dest_box:
            return "control"
        
        # MSP（信号）接続の検出
        if hasattr(source_box, "box_type") and source_box.box_type.is_msp_element:
            if hasattr(dest_box, "box_type") and dest_box.box_type.is_msp_element:
                return "signal"
        
        # ObjectBoxのチルダ判定（従来の方法）
        if isinstance(source_box, ObjectBox) and source_box.is_msp_object:
            if isinstance(dest_box, ObjectBox) and dest_box.is_msp_object:
                return "signal"
        
        # Jitter接続の検出
        if hasattr(source_box, "box_type") and source_box.box_type.is_jitter_element:
            if hasattr(dest_box, "box_type") and dest_box.box_type.is_jitter_element:
                return "matrix"
        
        # ObjectBoxのJitter判定（従来の方法）
        if isinstance(source_box, ObjectBox) and source_box.is_jitter_object:
            if isinstance(dest_box, ObjectBox) and dest_box.is_jitter_object:
                return "matrix"
        
        # Gen~関連の接続
        if isinstance(source_box, GenBox) or isinstance(dest_box, GenBox):
            return "signal"
        
        # ポート情報からの接続タイプ推定
        # アウトレットタイプの確認（あれば）
        if hasattr(source_box, "outlets") and source_box.outlets:
            for outlet in source_box.outlets:
                if outlet.get("index", -1) == source_port:
                    outlet_type = outlet.get("type", "").lower()
                    if outlet_type in ["signal", "audio", "msp"]:
                        return "signal"
                    elif outlet_type in ["matrix", "jit_matrix", "jitter"]:
                        return "matrix"
                    break
        
        # インレットタイプの確認（あれば）
        if hasattr(dest_box, "inlets") and dest_box.inlets:
            for inlet in dest_box.inlets:
                if inlet.get("index", -1) == dest_port:
                    inlet_type = inlet.get("type", "").lower()
                    if inlet_type in ["signal", "audio", "msp"]:
                        return "signal"
                    elif inlet_type in ["matrix", "jit_matrix", "jitter"]:
                        return "matrix"
                    break
        
        # オブジェクト名に基づく推測
        if isinstance(source_box, ObjectBox):
            obj_name = source_box.object_name.lower()
            # MSP信号オブジェクト
            if (obj_name.endswith("~") or 
                obj_name.startswith("gen~") or 
                obj_name in ["sig", "adc", "dac", "ezdac", "ezadc"]):
                return "signal"
            
            # Jitterオブジェクト
            if (obj_name.startswith("jit.") or 
                obj_name.startswith("gl.") or 
                obj_name in ["jit", "jitter"]):
                return "matrix"
        
        # デフォルトは制御接続
        return "control"
    
    def _process_subpatchers(self, patch_data: Dict[str, Any]) -> None:
        """サブパッチャーの処理"""
        # まずは全ボックスを走査してサブパッチャーを特定
        for box_id, box in self.parsed_boxes.items():
            # サブパッチャーを含む可能性のあるボックスタイプを検出
            is_subpatcher = False
            
            # ObjectBoxのサブパッチャー
            if isinstance(box, ObjectBox) and box.is_subpatcher:
                is_subpatcher = True
            
            # PatcherBoxはサブパッチャー
            elif isinstance(box, PatcherBox):
                is_subpatcher = True
            
            # GenBoxのサブパッチャー
            elif isinstance(box, GenBox) and box.has_subpatch:
                is_subpatcher = True
            
            # サブパッチャーがある場合は処理
            if is_subpatcher:
                # パッチデータからサブパッチャーのデータを検索
                subpatcher_data = self._find_subpatcher_data(patch_data, box_id)
                if subpatcher_data:
                    # サブパッチャーを解析
                    self._parse_subpatcher(box, subpatcher_data)
    
    def _find_subpatcher_data(self, patch_data: Dict[str, Any], box_id: str) -> Optional[Dict[str, Any]]:
        """指定されたボックスIDに対応するサブパッチャーデータを検索"""
        patcher = patch_data.get("patcher", {})
        boxes_data = patcher.get("boxes", {})
        
        # バージョンに応じた処理
        if isinstance(boxes_data, list):
            # 古い形式: リスト形式
            for box in boxes_data:
                if "box" in box and box["box"].get("id") == box_id:
                    # patcherキーの場所はバージョンによって異なる
                    if "patcher" in box["box"]:
                        return box["box"]["patcher"]
                    elif "subpatcher" in box["box"]:
                        return box["box"]["subpatcher"]
        else:
            # 新しい形式: 辞書形式
            if box_id in boxes_data:
                box_data = boxes_data[box_id]
                # patcherキーの場所はバージョンによって異なる
                if "box" in box_data:
                    if "patcher" in box_data["box"]:
                        return box_data["box"]["patcher"]
                    elif "subpatcher" in box_data["box"]:
                        return box_data["box"]["subpatcher"]
                else:
                    if "patcher" in box_data:
                        return box_data["patcher"]
                    elif "subpatcher" in box_data:
                        return box_data["subpatcher"]
        
        return None
    
    def _parse_subpatcher(self, parent_box: MaxBox, subpatcher_data: Dict[str, Any]) -> None:
        """サブパッチャーを解析"""
        # サブパッチャーフラグを設定（必要なら）
        if hasattr(parent_box, "has_subpatch"):
            parent_box.has_subpatch = True
        
        # サブパッチャーのパース（別のインスタンスを使用）
        sub_parser = EnhancedPatchParser()
        sub_boxes, sub_connections, sub_metadata = sub_parser.parse_patch_data(
            {"patcher": subpatcher_data}
        )
        
        # 親参照の設定
        for sub_box in sub_boxes.values():
            sub_box.parent_id = parent_box.id
        
        # サブパッチを親ボックスに保存
        if hasattr(parent_box, "subpatch"):
            parent_box.subpatch = sub_boxes
        
        # 入出力ポートマッピングの処理
        self._process_subpatcher_ports(parent_box, sub_boxes, sub_connections)
    
    def _process_subpatcher_ports(
        self, 
        parent_box: MaxBox, 
        sub_boxes: Dict[str, MaxBox], 
        sub_connections: List[Dict[str, Any]]
    ) -> None:
        """サブパッチャーのポートマッピングを処理
        
        サブパッチャー内のinlet/outletオブジェクトを検出し、
        親ボックスのインレット/アウトレットと適切にマッピングする。
        """
        # inlet/outletオブジェクトの収集
        inlets = []
        outlets = []
        
        for sub_box in sub_boxes.values():
            if isinstance(sub_box, ObjectBox):
                if sub_box.object_name == "inlet":
                    # inlet引数の処理（インデックス）
                    inlet_idx = 0
                    if sub_box.arguments and len(sub_box.arguments) > 0:
                        try:
                            inlet_idx = int(sub_box.arguments[0]) - 1
                        except (ValueError, TypeError):
                            inlet_idx = 0
                    
                    inlets.append({
                        "box": sub_box,
                        "index": inlet_idx
                    })
                
                elif sub_box.object_name == "outlet":
                    # outlet引数の処理（インデックス）
                    outlet_idx = 0
                    if sub_box.arguments and len(sub_box.arguments) > 0:
                        try:
                            outlet_idx = int(sub_box.arguments[0]) - 1
                        except (ValueError, TypeError):
                            outlet_idx = 0
                    
                    outlets.append({
                        "box": sub_box,
                        "index": outlet_idx
                    })
        
        # インレット/アウトレットが実際に何個あるか設定
        if hasattr(parent_box, "numinlets"):
            parent_box.numinlets = max(len(inlets), 1)
        
        if hasattr(parent_box, "numoutlets"):
            parent_box.numoutlets = max(len(outlets), 1)
        
        # インレット/アウトレットのクリア（再設定）
        if hasattr(parent_box, "inlets") and hasattr(parent_box, "outlets"):
            parent_box.inlets = []
            parent_box.outlets = []
            
            # サブパッチャーのinletに基づいてインレット設定
            for inlet in sorted(inlets, key=lambda x: x["index"]):
                inlet_box = inlet["box"]
                idx = inlet["index"]
                
                # 型情報の取得（可能なら）
                data_type = "any"
                comments = []
                
                # inletに接続されているオブジェクトの型情報収集
                for conn in sub_connections:
                    if conn["destination_id"] == inlet_box.id:
                        source_id = conn["source_id"]
                        source_box = sub_boxes.get(source_id)
                        if source_box and isinstance(source_box, ObjectBox):
                            # 信号オブジェクトかどうかチェック
                            if source_box.is_msp_object:
                                data_type = "signal"
                                break
                            # オブジェクト名に基づくヒント
                            obj_name = source_box.object_name.lower()
                            if "signal" in obj_name or obj_name.endswith("~"):
                                data_type = "signal"
                            elif "matrix" in obj_name or obj_name.startswith("jit."):
                                data_type = "jit_matrix"
                
                # inlet引数から型情報を取得（例: inlet~ -> signal）
                if inlet_box.object_name.endswith("~"):
                    data_type = "signal"
                
                # インレット情報を追加
                parent_box.add_inlet(idx, data_type, f"Inlet {idx+1}", False)
            
            # サブパッチャーのoutletに基づいてアウトレット設定
            for outlet in sorted(outlets, key=lambda x: x["index"]):
                outlet_box = outlet["box"]
                idx = outlet["index"]
                
                # 型情報の取得（可能なら）
                data_type = "any"
                comments = []
                
                # outletに接続されているオブジェクトの型情報収集
                for conn in sub_connections:
                    if conn["source_id"] == outlet_box.id:
                        dest_id = conn["destination_id"]
                        dest_box = sub_boxes.get(dest_id)
                        if dest_box and isinstance(dest_box, ObjectBox):
                            # 信号オブジェクトかどうかチェック
                            if dest_box.is_msp_object:
                                data_type = "signal"
                                break
                            # オブジェクト名に基づくヒント
                            obj_name = dest_box.object_name.lower()
                            if "signal" in obj_name or obj_name.endswith("~"):
                                data_type = "signal"
                            elif "matrix" in obj_name or obj_name.startswith("jit."):
                                data_type = "jit_matrix"
                
                # outlet引数から型情報を取得（例: outlet~ -> signal）
                if outlet_box.object_name.endswith("~"):
                    data_type = "signal"
                
                # アウトレット情報を追加
                parent_box.add_outlet(idx, data_type, f"Outlet {idx+1}", False)
    
    def _set_hierarchy_levels(self) -> None:
        """パッチの階層レベルを設定"""
        # 親ボックスがないボックスは階層0
        for box in self.parsed_boxes.values():
            if not box.parent_id:
                box.hierarchy_level = 0
                # 子ボックスがある場合は再帰的に処理
                if isinstance(box, ObjectBox) and box.has_subpatch and box.subpatch:
                    self._set_child_hierarchy_levels(box.subpatch, 1)
    
    def _set_child_hierarchy_levels(self, boxes: Dict[str, MaxBox], level: int) -> None:
        """子ボックスの階層レベルを再帰的に設定"""
        for box in boxes.values():
            box.hierarchy_level = level
            # さらに子がある場合は再帰
            if isinstance(box, ObjectBox) and box.has_subpatch and box.subpatch:
                self._set_child_hierarchy_levels(box.subpatch, level + 1)


def parse_maxpat_file(file_path: str) -> Tuple[Dict[str, MaxBox], List[Dict[str, Any]], Dict[str, Any]]:
    """Max/MSPパッチファイルを解析するユーティリティ関数
    
    Args:
        file_path: .maxpatファイルのパス
        
    Returns:
        (boxes, connections, metadata): ボックス、接続、メタデータの情報
    """
    parser = EnhancedPatchParser()
    return parser.parse_file(file_path)