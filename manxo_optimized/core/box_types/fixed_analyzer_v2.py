#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修正版 Max/MSP ボックスタイプ分析ツール (V2)
Max/MSPのパッチファイルを分析し、各ボックスのタイプを正確に識別します。
特に、「オブジェクトボックス」と「オブジェクト」の重要な区別を行います。
"""

import os
import json
import re
import sys
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import Counter

class FixedBoxTypeAnalyzerV2:
    """Max/MSPボックスタイプの高精度分析クラス（修正版V2）"""
    
    # maxclassの基本カテゴリ
    BOX_CATEGORIES = {
        "newobj": "オブジェクトボックス",
        "message": "メッセージボックス",
        "comment": "コメントボックス",
        "number": "数値ボックス (整数)",
        "flonum": "数値ボックス (小数)",
        "toggle": "トグル",
        "button": "ボタン",
        "slider": "スライダー",
        "dial": "ダイアル",
        "multislider": "マルチスライダー",
        "gain~": "ゲインフェーダー",
        "kslider": "キーボード",
        "ezdac~": "オーディオ出力",
        "ezadc~": "オーディオ入力",
        "bpatcher": "埋め込みパッチャー",
        "panel": "パネル",
        "radiogroup": "ラジオグループ",
        "tab": "タブ",
        "umenu": "メニュー",
        "attrui": "属性UI",
        "spectroscope~": "スペクトロスコープ",
        "live.slider": "Live.スライダー",
        "live.dial": "Live.ダイアル",
        "live.numbox": "Live.数値ボックス",
        "live.toggle": "Live.トグル",
        "live.text": "Live.テキスト",
        "live.menu": "Live.メニュー",
        "live.tab": "Live.タブ",
        "live.grid": "Live.グリッド",
    }
    
    # MSP信号オブジェクト（チルダ～付き）のパターン
    MSP_PATTERN = re.compile(r'^(.+)~$')
    
    # サブパッチャーオブジェクトの種類
    SUBPATCHER_TYPES = ["p", "patcher", "poly~", "bpatcher", "pattrstorage"]
    
    def __init__(self, debug: bool = False):
        """
        初期化
        
        Args:
            debug: デバッグモードフラグ (詳細ログを出力)
        """
        self.debug = debug
        # 分析結果の保存用
        self.maxclass_stats = Counter()
        self.object_stats = Counter()
        self.box_details = []
    
    def analyze_patch_file(self, file_path: str) -> Dict[str, Any]:
        """
        Max/MSPパッチファイルを分析
        
        Args:
            file_path: 分析するパッチファイルのパス
            
        Returns:
            分析結果の辞書
        """
        if not os.path.exists(file_path):
            print(f"エラー: ファイル {file_path} が存在しません")
            return {}
            
        try:
            # パッチファイルを読み込み
            with open(file_path, 'r', encoding='utf-8') as f:
                patch_data = json.load(f)
                
            # 分析を実行
            result = self._analyze_patch_data(patch_data, os.path.basename(file_path))
            
            if self.debug:
                print(f"パッチファイル '{file_path}' の分析が完了しました")
                
            return result
            
        except json.JSONDecodeError:
            print(f"エラー: '{file_path}' は有効なJSONファイルではありません")
            return {}
        except Exception as e:
            print(f"エラー: '{file_path}' の分析中にエラーが発生しました: {str(e)}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return {}
    
    def _analyze_patch_data(self, patch_data: Dict[str, Any], patch_name: str) -> Dict[str, Any]:
        """
        パッチデータを分析し、ボックスタイプ情報を抽出
        
        Args:
            patch_data: パッチのJSONデータ
            patch_name: パッチ名（ファイル名）
            
        Returns:
            分析結果の辞書
        """
        result = {
            "patch_name": patch_name,
            "box_count": 0,
            "maxclass_counts": {},
            "object_type_counts": {},
            "boxes": []
        }
        
        # パッチャーオブジェクトを確認
        patcher = patch_data.get("patcher", {})
        
        # ボックスを分析
        boxes = patcher.get("boxes", [])
        
        # boxesがリストかどうかを確認
        if isinstance(boxes, list):
            # リスト形式の場合の処理
            result["box_count"] = len(boxes)
            for box_data in boxes:
                self._analyze_box(box_data, result)
        else:
            # 辞書形式の場合の処理
            result["box_count"] = len(boxes)
            for box_id, box_data in boxes.items():
                # box_idを直接使用できる
                self._analyze_box(box_data, result, box_id)
        
        # 統計情報を集計
        result["maxclass_counts"] = dict(self.maxclass_stats)
        result["object_type_counts"] = dict(self.object_stats)
        
        return result
    
    def _analyze_box(self, box_data: Dict[str, Any], result: Dict[str, Any], box_id: str = None) -> None:
        """
        単一のボックスを分析
        
        Args:
            box_data: ボックスデータ
            result: 結果辞書（更新される）
            box_id: ボックスID（辞書形式の場合）
        """
        # box形式の確認（patcherバージョンによって異なる）
        box_content = box_data.get("box", box_data)
        
        # IDの取得
        if box_id is None and "id" in box_content:
            box_id = box_content["id"]
        elif box_id is None:
            box_id = f"box-{len(result['boxes'])}"
        
        # 基本的なオブジェクト情報
        maxclass = box_content.get("maxclass", "")
        self.maxclass_stats[maxclass] += 1
        
        # box_detailsにmaxclassを追加
        box_detail = {
            "id": box_id,
            "maxclass": maxclass,
            "box_type": self.BOX_CATEGORIES.get(maxclass, "その他"),
        }
        
        # テキスト（引数）の抽出
        obj_text = box_content.get("text", "")
        box_detail["text"] = obj_text
        
        # maxclassがnewobjの場合、テキストの最初の単語が実際のオブジェクトタイプ
        if maxclass == "newobj" and obj_text:
            parts = obj_text.split(None, 1)
            if parts:
                real_type = parts[0]
                args = parts[1] if len(parts) > 1 else ""
                
                # オブジェクトの種類をカウント
                self.object_stats[real_type] += 1
                
                # オブジェクトの詳細情報を追加
                box_detail["object_type"] = real_type
                box_detail["object_args"] = args
                
                # オブジェクトカテゴリを判定
                if real_type in self.SUBPATCHER_TYPES or real_type.startswith("p "):
                    box_detail["object_category"] = "サブパッチャー"
                elif self.MSP_PATTERN.match(real_type):
                    box_detail["object_category"] = "MSP信号オブジェクト"
                elif real_type.startswith("jit."):
                    box_detail["object_category"] = "Jitterオブジェクト"
                elif real_type.startswith("live."):
                    box_detail["object_category"] = "Liveオブジェクト"
                else:
                    box_detail["object_category"] = "標準オブジェクト"
        
        # インレット・アウトレット情報
        box_detail["numinlets"] = box_content.get("numinlets", 0)
        box_detail["numoutlets"] = box_content.get("numoutlets", 0)
        box_detail["outlettype"] = box_content.get("outlettype", [])
        
        # 位置情報
        box_detail["patching_rect"] = box_content.get("patching_rect", [])
        
        # サブパッチャーの場合、埋め込まれたパッチャーを再帰的に処理
        if "patcher" in box_content:
            subpatch_data = box_content["patcher"]
            subpatch_name = f"{result['patch_name']}_{box_id}"
            box_detail["subpatcher"] = self._analyze_patch_data(
                {"patcher": subpatch_data}, subpatch_name
            )
        
        # 結果に追加
        self.box_details.append(box_detail)
        result["boxes"].append(box_detail)
    
    def generate_report(self, results: List[Dict[str, Any]], output_dir: str = None) -> str:
        """
        分析結果からレポートを生成
        
        Args:
            results: 分析結果のリスト
            output_dir: 出力ディレクトリ (Noneの場合は生成しない)
            
        Returns:
            生成されたレポートファイルパス (出力された場合)
        """
        # 全てのボックスを集計
        all_boxes = []
        for result in results:
            all_boxes.extend(result.get("boxes", []))
        
        if not all_boxes:
            print("警告: 分析結果が空です")
            return ""
        
        # レポート文字列の初期化
        report = "# Max/MSP ボックスタイプ分析レポート\n\n"
        
        # 基本統計
        total_patches = len(results)
        total_boxes = len(all_boxes)
        report += f"## 基本統計\n\n"
        report += f"- 分析したパッチ数: {total_patches}\n"
        report += f"- 合計ボックス数: {total_boxes}\n\n"
        
        # maxclass統計
        report += f"## maxclassタイプ分布\n\n"
        maxclass_counts = {}
        for box in all_boxes:
            mc = box.get('maxclass', '')
            maxclass_counts[mc] = maxclass_counts.get(mc, 0) + 1
            
        if maxclass_counts:
            report += "| maxclass | カウント | パーセント |\n"
            report += "|----------|----------|------------|\n"
            for mc, count in sorted(maxclass_counts.items(), key=lambda x: x[1], reverse=True):
                percent = (count / total_boxes) * 100
                report += f"| {mc} | {count} | {percent:.2f}% |\n"
            report += "\n"
        
        # オブジェクトタイプ統計 (newobjの内訳)
        report += f"## オブジェクトタイプ分布 (newobj内)\n\n"
        object_types = {}
        for box in all_boxes:
            if box.get('maxclass') == 'newobj' and 'object_type' in box:
                obj_type = box.get('object_type', '')
                object_types[obj_type] = object_types.get(obj_type, 0) + 1
                
        if object_types:
            report += "| オブジェクトタイプ | カウント | パーセント |\n"
            report += "|------------------|----------|------------|\n"
            newobj_total = sum(object_types.values())
            for obj, count in sorted(object_types.items(), key=lambda x: x[1], reverse=True)[:30]:
                percent = (count / newobj_total) * 100
                report += f"| {obj} | {count} | {percent:.2f}% |\n"
            if len(object_types) > 30:
                report += f"| ... (他 {len(object_types)-30} 種類) | - | - |\n"
            report += "\n"
        
        # オブジェクトカテゴリ統計
        report += f"## オブジェクトカテゴリ分布\n\n"
        categories = {}
        for box in all_boxes:
            if 'object_category' in box:
                cat = box.get('object_category', '')
                categories[cat] = categories.get(cat, 0) + 1
                
        if categories:
            report += "| カテゴリ | カウント | パーセント |\n"
            report += "|----------|----------|------------|\n"
            category_total = sum(categories.values())
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                percent = (count / category_total) * 100
                report += f"| {cat} | {count} | {percent:.2f}% |\n"
            report += "\n"
        
        # ファイルに出力
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, "box_type_analysis_report.md")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            # JSONでも詳細データを出力
            json_path = os.path.join(output_dir, "box_type_analysis_data.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_boxes, f, ensure_ascii=False, indent=2)
                
            print(f"レポートを保存しました: {report_path}")
            print(f"詳細データを保存しました: {json_path}")
            return report_path
        
        return report