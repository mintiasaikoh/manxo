#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
box_type_analyzer.py - Max/MSPのボックスタイプとオブジェクトタイプを正確に解析するモジュール

このスクリプトは、Max/MSPパッチの構造を解析し、オブジェクトボックスとその中に含まれる
実際のオブジェクトを区別して処理します。maxclassによるボックスタイプの分類と、
textプロパティによるオブジェクトタイプの識別を明確に分離して行います。
"""

import os
import sys
import json
import logging
import argparse
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set, Union

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("box_type_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("box_type_analyzer")

class BoxTypeAnalyzer:
    """Max/MSPのボックスタイプとオブジェクトタイプを解析するクラス"""
    
    def __init__(self, xml_port_data_file=None):
        """
        初期化
        
        Args:
            xml_port_data_file: XMLから抽出したポート情報ファイル（オプション）
        """
        # XMLポート情報をロード（存在する場合）
        self.xml_port_data = {}
        if xml_port_data_file and os.path.exists(xml_port_data_file):
            try:
                with open(xml_port_data_file, 'r', encoding='utf-8') as f:
                    self.xml_port_data = json.load(f)
                logger.info(f"{len(self.xml_port_data)}個のXMLオブジェクト情報をロードしました")
            except Exception as e:
                logger.error(f"XMLポート情報ロードエラー: {e}")
        
        # maxclass分類定義
        self.maxclass_categories = {
            # 標準オブジェクト
            "newobj": "container",    # コンテナ - 実際のオブジェクトはtextフィールドで定義
            
            # メッセージ系
            "message": "message",     # メッセージボックス
            "comment": "ui",          # コメント
            
            # 数値系UI
            "number": "ui",           # 整数ボックス
            "flonum": "ui",           # 小数ボックス
            "incdec": "ui",           # インクリメント/デクリメントボタン
            
            # コントロール系UI
            "toggle": "ui",           # トグルスイッチ
            "button": "ui",           # バンボタン
            "slider": "ui",           # スライダー
            "dial": "ui",             # ダイヤル
            "rslider": "ui",          # 範囲スライダー 
            "multislider": "ui",      # マルチスライダー
            "kslider": "ui",          # キーボードスライダー
            "gain~": "ui",            # ゲインスライダー
            "live.slider": "ui",      # Liveスライダー
            "live.dial": "ui",        # Liveダイヤル
            "live.toggle": "ui",      # Liveトグル
            "live.button": "ui",      # Liveボタン
            
            # 表示系UI
            "panel": "ui",            # パネル
            "lcd": "ui",              # LCD表示
            "fpic": "ui",             # 画像表示
            "scope~": "ui",           # オシロスコープ
            "meter~": "ui",           # レベルメーター
            "spectroscope~": "ui",    # スペクトルアナライザー
            
            # オーディオ系
            "ezdac~": "audio",        # 簡易オーディオ出力
            "ezadc~": "audio",        # 簡易オーディオ入力
            "gain~": "audio",         # ゲイン調整
            
            # ファイル系
            "playlist~": "file",      # プレイリスト
            "buffer~": "file",        # バッファー管理
            
            # その他特殊
            "patcher": "patcher",     # パッチャー
            "bpatcher": "patcher",    # Bパッチャー
            "jpatcher": "patcher",    # Jパッチャー
            "inlet": "patcher",       # インレット
            "outlet": "patcher",      # アウトレット
            "patchercontextview": "patcher",  # パッチャーコンテキストビュー
            
            # プリセット系
            "preset": "preset",       # プリセット
            "pattrstorage": "preset", # Pattr ストレージ
            
            # テキスト系
            "textedit": "text",       # テキスト編集
            
            # データ構造
            "dict": "data",           # 辞書
            "table": "data",          # テーブル
            "coll": "data",           # コレクション
        }
        
        # newobjのカテゴリ分類定義
        self.object_categories = {
            # 基本カテゴリー
            "audio": [
                "~", "dsp", "adc", "dac", "sig", "audio", "sound", "jit.catch", 
                "buffer~", "wave~", "groove~", "play~", "record~"
            ],
            "control": [
                "metro", "counter", "select", "sel", "route", "gate", "trigger", "t ", "p ", 
                "patcher", "pak", "unpack", "print", "int", "float"
            ],
            "math": [
                "+", "-", "*", "/", "maximum", "minimum", "avg", "expr", "scale", "pow", "log",
                "sin", "cos", "tan", "abs", "sqrt", "random"
            ],
            "midi": [
                "notein", "noteout", "ctlin", "ctlout", "makenote", "midi", "midiformat", 
                "midiin", "midiout", "pgmin", "pgmout"
            ],
            "visual": [
                "jit.", "lcd", "swatch", "fpic", "pict", "panel", "jit.matrix", "jit.window",
                "jit.pwindow", "jit.gl."
            ],
            "storage": [
                "buffer~", "table", "coll", "dict", "store", "pattrstorage", "pattr"
            ],
            "utility": [
                "loadbang", "loadmess", "deferlow", "defer", "print", "snapshot~", "forward",
                "pipe", "delay", "qlim"
            ],
            "m4l": [
                "live.", "plugout~", "plugin~", "plugsend~", "plugreceive~", "live.slider",
                "live.dial", "live.button", "live.toggle"
            ],
            "subpatcher": [
                "p ", "poly~", "patcher", "bpatcher", "polybuffer~"
            ],
            "js": [
                "js ", "js."
            ]
        }
        
        # キャッシュ
        self.object_cache = {}
        self.port_info_cache = {}
        
        # オブジェクト解析用の特殊パターン
        self.special_patterns = {
            r'^p\s+': "patcher",
            r'^poly~\s+': "poly~",
            r'^gen~\s+': "gen~",
            r'^js\s+': "js",
            r'^bpatcher\s+': "bpatcher",
            r'^if\s+': "if",
            r'^expr\s+': "expr",
            r'^jit\.matrix\s+': "jit.matrix",
            r'^jit\.gl\.model\s+': "jit.gl.model"
        }
        
        # 正規化パターン
        self.normalization_patterns = [
            (r'([^\\])"', r'\1'),  # 引用符除去
            (r'\\(.)', r'\1'),     # エスケープ文字除去
            (r'\s+', ' '),         # 複数空白を単一に
            (r'^\s+|\s+$', '')     # 先頭末尾の空白除去
        ]
        
        # オブジェクト名のエイリアス
        self.object_aliases = {
            "sel": "select",
            "t": "trigger",
            "pak": "pack",
            "r": "receive",
            "s": "send",
            "f": "float",
            "i": "int",
            "b": "bang",
            "r~": "receive~",
            "s~": "send~"
        }
        
        # Max/MSPオブジェクトの出力タイプマッピング
        self.outlettype_mapping = {
            "metro": ["bang"],
            "counter": ["int", "bang", "bang"],
            "select": ["bang", "any"],  # デフォルト（引数に応じて変化）
            "route": ["any", "any"],   # デフォルト（引数に応じて変化）
            "pack": ["list"],
            "unpack": ["any", "any"],  # デフォルト（引数に応じて変化）
            "trigger": ["any", "any"],  # デフォルト（引数に応じて変化）
            "cycle~": ["signal"],
            "dac~": [],
            "adc~": ["signal", "signal"],
            "+": ["int"],
            "+~": ["signal"],
            "print": [],
            "patcher": ["any"],  # デフォルト
            "message": ["any"],
            "number": ["int", "bang"],
            "flonum": ["float", "bang"],
            "toggle": ["int"],
            "button": ["bang"]
        }
    
    def analyze_box(self, box_data):
        """
        ボックスタイプとオブジェクトタイプを解析
        
        Args:
            box_data: ボックスのデータ辞書
            
        Returns:
            拡張された属性を含むボックスデータ
        """
        # 入力チェック
        if not box_data or not isinstance(box_data, dict):
            return box_data
        
        # 必要なプロパティがあるか確認
        if 'maxclass' not in box_data:
            return box_data
        
        # 結果を格納する新しい辞書
        result = box_data.copy()
        
        # ボックスタイプ（maxclass）
        maxclass = box_data['maxclass']
        
        # ボックスカテゴリを取得
        box_category = self.maxclass_categories.get(maxclass, 'unknown')
        result['box_category'] = box_category
        
        # newobjタイプの場合はオブジェクトを解析
        if maxclass == 'newobj':
            # textフィールドからオブジェクトを取得
            text = box_data.get('text', '')
            
            # オブジェクトとその引数を解析
            obj_type, args = self._parse_object_text(text)
            
            # 結果を格納
            result['object_type'] = obj_type
            result['object_args'] = args
            
            # オブジェクトカテゴリを取得
            obj_category = self._determine_object_category(obj_type, text)
            result['object_category'] = obj_category
            
            # ポート情報を取得
            inlets, outlets = self._get_port_info(obj_type, args)
            result['calculated_inlets'] = inlets
            result['calculated_outlets'] = outlets
            
            # アウトレットタイプを取得
            outlet_types = self._get_outlet_types(obj_type, args, outlets)
            if outlet_types:
                result['calculated_outlettype'] = outlet_types
            
            # サブパッチャーかどうかを確認
            is_subpatcher = self._is_subpatcher(obj_type, text)
            if is_subpatcher:
                result['is_subpatcher'] = True
                result['subpatcher_type'] = obj_type
        else:
            # newobjでない場合は標準的なポート情報を追加
            inlets = box_data.get('numinlets', 1)
            outlets = box_data.get('numoutlets', 0)
            result['calculated_inlets'] = inlets
            result['calculated_outlets'] = outlets
            
            # アウトレットタイプがあればそのまま使用
            if 'outlettype' in box_data:
                result['calculated_outlettype'] = box_data['outlettype']
            
            # UI専用の情報を追加
            if box_category == 'ui':
                result['ui_type'] = maxclass
        
        return result
    
    def _parse_object_text(self, text):
        """
        オブジェクトテキストからオブジェクトタイプと引数を解析
        
        Args:
            text: オブジェクトテキスト
            
        Returns:
            (object_type, args)のタプル
        """
        # テキストがない場合
        if not text or not isinstance(text, str):
            return 'unknown', []
        
        # 特殊パターンのチェック
        for pattern, obj_type in self.special_patterns.items():
            if re.match(pattern, text):
                # 一致したパターンを除去して引数を取得
                args_text = re.sub(pattern, '', text)
                args = self._parse_args(args_text)
                return obj_type, args
        
        # 通常のパース: 最初のスペースで分割
        parts = text.strip().split(None, 1)
        
        if not parts:
            return 'unknown', []
        
        # オブジェクトタイプ
        obj_type = parts[0]
        
        # 引数
        args = []
        if len(parts) > 1:
            args = self._parse_args(parts[1])
        
        # エイリアス解決
        if obj_type in self.object_aliases:
            obj_type = self.object_aliases[obj_type]
        
        return obj_type, args
    
    def _parse_args(self, args_text):
        """
        引数テキストを解析して引数リストを取得
        
        Args:
            args_text: 引数テキスト
            
        Returns:
            引数のリスト
        """
        if not args_text or not isinstance(args_text, str):
            return []
        
        args = []
        
        # 引用符内の引数を特別に処理
        in_quotes = False
        current_arg = ""
        
        for char in args_text:
            if char == '"':
                in_quotes = not in_quotes
                # 引用符自体は引数に含めない
                continue
            
            if char.isspace() and not in_quotes:
                # 空白で区切る（引用符内は除く）
                if current_arg:
                    args.append(current_arg)
                    current_arg = ""
            else:
                current_arg += char
        
        # 最後の引数を追加
        if current_arg:
            args.append(current_arg)
        
        return args
    
    def _determine_object_category(self, obj_type, text=None):
        """
        オブジェクトタイプからカテゴリを決定
        
        Args:
            obj_type: オブジェクトタイプ
            text: オプションの完全テキスト
            
        Returns:
            カテゴリー名
        """
        if not obj_type:
            return 'unknown'
        
        # XMLデータでカテゴリが定義されているか確認
        obj_lower = obj_type.lower()
        if obj_lower in self.xml_port_data and 'type' in self.xml_port_data[obj_lower]:
            return self.xml_port_data[obj_lower]['type']
        
        # チルダ付きならオーディオ
        if obj_type.endswith('~'):
            return 'audio'
        
        # 定義されたカテゴリから検索
        for category, keywords in self.object_categories.items():
            for keyword in keywords:
                if keyword in obj_type:
                    return category
        
        # テキスト全体からカテゴリを推測
        if text:
            for category, keywords in self.object_categories.items():
                for keyword in keywords:
                    if keyword in text:
                        return category
        
        # デフォルトカテゴリ
        return 'control'
    
    def _get_port_info(self, obj_type, args=None):
        """
        オブジェクトタイプと引数からポート情報を取得
        
        Args:
            obj_type: オブジェクトタイプ
            args: 引数リスト（オプション）
            
        Returns:
            (inlets, outlets)のタプル
        """
        # キャッシュ確認
        cache_key = f"{obj_type}:{','.join(args) if args else ''}"
        if cache_key in self.port_info_cache:
            return self.port_info_cache[cache_key]
        
        # XMLデータからポート情報を取得
        obj_lower = obj_type.lower()
        if obj_lower in self.xml_port_data:
            port_info = self.xml_port_data[obj_lower]
            
            # 動的ポート処理
            if port_info.get('is_dynamic') or port_info.get('has_dynamic_ports'):
                dynamic_rule = port_info.get('dynamic_rule')
                if dynamic_rule:
                    inlets, outlets = self._apply_dynamic_rule(dynamic_rule, args)
                    self.port_info_cache[cache_key] = (inlets, outlets)
                    return inlets, outlets
            
            # 静的ポート情報
            inlets = port_info.get('inlets', 1)
            outlets = port_info.get('outlets', 1)
            self.port_info_cache[cache_key] = (inlets, outlets)
            return inlets, outlets
        
        # 既知のオブジェクトタイプ
        if obj_type == 'metro':
            return 2, 1  # metro: 2 inlets, 1 outlet
        elif obj_type == 'counter':
            return 3, 3  # counter: 3 inlets, 3 outlets
        elif obj_type in ['select', 'sel']:
            # select: 1 inlet, args+1 outlets
            return 1, (len(args) or 1) + 1
        elif obj_type == 'route':
            # route: 1 inlet, args+1 outlets
            return 1, (len(args) or 1) + 1
        elif obj_type in ['pack', 'pak']:
            # pack: args inlets (デフォルト2), 1 outlet
            return len(args) or 2, 1
        elif obj_type == 'unpack':
            # unpack: 1 inlet, args outlets (デフォルト2)
            return 1, len(args) or 2
        elif obj_type in ['trigger', 't']:
            # trigger: 1 inlet, args outlets (デフォルト2)
            return 1, len(args) or 2
        elif obj_type.endswith('~'):
            # MSPオブジェクト: 多くは1 inlet, 1 outlet
            if obj_type == 'dac~':
                return 2, 0  # dac~: 2 inlets, 0 outlets
            elif obj_type == 'adc~':
                return 0, 2  # adc~: 0 inlets, 2 outlets
            else:
                return 1, 1  # デフォルト MSP
        elif obj_type in ['print', 'message']:
            return 1, 0  # print: 1 inlet, 0 outlets
        
        # デフォルト値
        self.port_info_cache[cache_key] = (1, 1)
        return 1, 1
    
    def _apply_dynamic_rule(self, rule, args):
        """
        動的ポートルールを適用
        
        Args:
            rule: ルール名
            args: 引数リスト
            
        Returns:
            (inlets, outlets)のタプル
        """
        if rule == 'select':
            return 1, (len(args) or 1) + 1
        elif rule == 'route':
            return 1, (len(args) or 1) + 1
        elif rule == 'pack':
            return len(args) or 2, 1
        elif rule == 'unpack':
            return 1, len(args) or 2
        elif rule == 'trigger':
            return 1, len(args) or 2
        elif rule == 'gate':
            # gate: 2 inlets, arg[0] outlets (デフォルト1)
            if args and args[0].isdigit():
                return 2, int(args[0])
            return 2, 1
        
        # ルールが不明な場合はデフォルト
        return 1, 1
    
    def _get_outlet_types(self, obj_type, args, num_outlets):
        """
        オブジェクトタイプと引数からアウトレットタイプを取得
        
        Args:
            obj_type: オブジェクトタイプ
            args: 引数リスト
            num_outlets: アウトレット数
            
        Returns:
            アウトレットタイプのリスト（不明な場合はNone）
        """
        # XMLデータからアウトレットタイプ情報を取得
        obj_lower = obj_type.lower()
        if obj_lower in self.xml_port_data:
            port_info = self.xml_port_data[obj_lower]
            if 'outlet_details' in port_info and isinstance(port_info['outlet_details'], list):
                outlet_types = []
                for outlet in port_info['outlet_details']:
                    if 'type' in outlet:
                        outlet_types.append(outlet['type'])
                
                # アウトレット数に合わせる
                while len(outlet_types) < num_outlets:
                    outlet_types.append("any")
                
                return outlet_types[:num_outlets]
        
        # 既知のオブジェクトタイプからアウトレットタイプを決定
        if obj_type in self.outlettype_mapping:
            base_types = self.outlettype_mapping[obj_type]
            
            # アウトレット数に合わせて拡張
            if len(base_types) < num_outlets:
                # 最後のタイプを繰り返す
                if base_types:
                    last_type = base_types[-1]
                    base_types = base_types + [last_type] * (num_outlets - len(base_types))
                else:
                    # デフォルトタイプ
                    base_types = ["any"] * num_outlets
            
            return base_types[:num_outlets]
        
        # MSPオブジェクト
        if obj_type.endswith('~'):
            return ["signal"] * num_outlets
        
        # それ以外は不明
        return None
    
    def _is_subpatcher(self, obj_type, text):
        """
        オブジェクトがサブパッチャーかどうかを判断
        
        Args:
            obj_type: オブジェクトタイプ
            text: オブジェクトテキスト
            
        Returns:
            サブパッチャーならTrue
        """
        # 定義済みのサブパッチャータイプ
        subpatcher_types = [
            'p', 'patcher', 'poly~', 'bpatcher', 'polybuffer~'
        ]
        
        # オブジェクトタイプがサブパッチャーリストにあるか
        if obj_type in subpatcher_types:
            return True
        
        # テキストがサブパッチャーパターンにマッチするか
        if text and isinstance(text, str):
            for pattern in ['^p\s+', '^patcher\s+', '^poly~\s+', '^bpatcher\s+']:
                if re.match(pattern, text):
                    return True
        
        return False
    
    def analyze_patch_file(self, file_path):
        """
        パッチファイル全体を解析
        
        Args:
            file_path: 解析するパッチファイルのパス
            
        Returns:
            解析結果の辞書
        """
        try:
            # ファイルを読み込み
            with open(file_path, 'r', encoding='utf-8') as f:
                patch_data = json.load(f)
            
            # パッチ構造を確認
            if 'patcher' not in patch_data:
                logger.error(f"無効なパッチファイル形式: {file_path}")
                return None
            
            # 解析結果
            result = {
                'file_path': file_path,
                'boxes': [],
                'connections': [],
                'stats': {
                    'box_types': defaultdict(int),
                    'object_types': defaultdict(int),
                    'box_categories': defaultdict(int),
                    'object_categories': defaultdict(int)
                }
            }
            
            # ボックスを解析
            if 'boxes' in patch_data['patcher']:
                for box_id, box_data in patch_data['patcher']['boxes'].items():
                    if 'box' in box_data:
                        # ボックスを解析
                        box_info = self.analyze_box(box_data['box'])
                        
                        # 結果に追加
                        result['boxes'].append(box_info)
                        
                        # 統計情報を更新
                        maxclass = box_info.get('maxclass')
                        if maxclass:
                            result['stats']['box_types'][maxclass] += 1
                        
                        box_category = box_info.get('box_category')
                        if box_category:
                            result['stats']['box_categories'][box_category] += 1
                        
                        if maxclass == 'newobj':
                            obj_type = box_info.get('object_type')
                            if obj_type:
                                result['stats']['object_types'][obj_type] += 1
                            
                            obj_category = box_info.get('object_category')
                            if obj_category:
                                result['stats']['object_categories'][obj_category] += 1
            
            # 接続を解析
            if 'lines' in patch_data['patcher']:
                for line_id, line_data in patch_data['patcher']['lines'].items():
                    if 'patchline' in line_data:
                        # 接続情報をそのまま追加
                        result['connections'].append(line_data['patchline'])
            
            # ボックスIDからマッピングを作成
            id_to_box = {box.get('id'): box for box in result['boxes']}
            
            # 接続にボックス情報を追加
            for conn in result['connections']:
                source_id = conn.get('source')
                dest_id = conn.get('destination')
                
                if source_id in id_to_box:
                    source_box = id_to_box[source_id]
                    conn['source_maxclass'] = source_box.get('maxclass')
                    
                    if source_box.get('maxclass') == 'newobj':
                        conn['source_object_type'] = source_box.get('object_type')
                    
                if dest_id in id_to_box:
                    dest_box = id_to_box[dest_id]
                    conn['dest_maxclass'] = dest_box.get('maxclass')
                    
                    if dest_box.get('maxclass') == 'newobj':
                        conn['dest_object_type'] = dest_box.get('object_type')
            
            # 階層構造を検出（サブパッチャー）
            result['subpatchers'] = self._find_subpatchers(patch_data['patcher'], result['boxes'])
            
            return result
            
        except Exception as e:
            logger.error(f"パッチファイル解析エラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _find_subpatchers(self, patcher_data, boxes):
        """
        パッチャー内のサブパッチャーを検索
        
        Args:
            patcher_data: パッチャーデータ
            boxes: 解析済みのボックスリスト
            
        Returns:
            サブパッチャー情報のリスト
        """
        subpatchers = []
        
        # サブパッチャーを持つボックスを探す
        for box in boxes:
            if box.get('is_subpatcher'):
                subpatcher_type = box.get('subpatcher_type')
                box_id = box.get('id')
                
                # サブパッチャー情報
                subpatcher_info = {
                    'id': box_id,
                    'type': subpatcher_type,
                    'parent_id': None
                }
                
                # ボックスにpatcherプロパティがあるか確認
                if 'patcher' in box:
                    # 再帰的に解析
                    child_patcher = self.analyze_patch_structure(box['patcher'])
                    subpatcher_info['contents'] = child_patcher
                
                subpatchers.append(subpatcher_info)
        
        return subpatchers
    
    def analyze_patch_structure(self, patcher_data):
        """
        パッチャー構造を解析（サブパッチャー用の再帰処理）
        
        Args:
            patcher_data: パッチャーデータ
            
        Returns:
            解析結果の辞書
        """
        result = {
            'boxes': [],
            'connections': []
        }
        
        # ボックスを解析
        if 'boxes' in patcher_data:
            for box_id, box_data in patcher_data['boxes'].items():
                if 'box' in box_data:
                    # ボックスを解析
                    box_info = self.analyze_box(box_data['box'])
                    
                    # 結果に追加
                    result['boxes'].append(box_info)
        
        # 接続を解析
        if 'lines' in patcher_data:
            for line_id, line_data in patcher_data['lines'].items():
                if 'patchline' in line_data:
                    # 接続情報をそのまま追加
                    result['connections'].append(line_data['patchline'])
        
        # サブパッチャーを再帰的に検索
        result['subpatchers'] = self._find_subpatchers(patcher_data, result['boxes'])
        
        return result
    
    def generate_connection_patterns(self, patch_data):
        """
        接続パターンを生成
        
        Args:
            patch_data: 解析済みのパッチデータ
            
        Returns:
            接続パターンの辞書
        """
        patterns = defaultdict(int)
        type_pairs = defaultdict(int)
        
        # ボックスIDから情報へのマッピングを作成
        id_to_box = {box.get('id'): box for box in patch_data['boxes']}
        
        # 接続からパターンを生成
        for conn in patch_data['connections']:
            source_id = conn.get('source')
            dest_id = conn.get('destination')
            
            if source_id not in id_to_box or dest_id not in id_to_box:
                continue
            
            source_box = id_to_box[source_id]
            dest_box = id_to_box[dest_id]
            
            source_port = conn.get('sourceoutlet', 0)
            dest_port = conn.get('destinationinlet', 0)
            
            # ボックスのmaxclassを取得
            source_maxclass = source_box.get('maxclass')
            dest_maxclass = dest_box.get('maxclass')
            
            # オブジェクトタイプを取得（newobjの場合）
            source_type = source_box.get('object_type') if source_maxclass == 'newobj' else source_maxclass
            dest_type = dest_box.get('object_type') if dest_maxclass == 'newobj' else dest_maxclass
            
            # パターン文字列を生成
            pattern_key = f"{source_type}:{source_port} -> {dest_type}:{dest_port}"
            patterns[pattern_key] += 1
            
            # タイプペア文字列を生成
            type_pair_key = f"{source_type} -> {dest_type}"
            type_pairs[type_pair_key] += 1
        
        return {
            'patterns': dict(patterns),
            'type_pairs': dict(type_pairs)
        }
    
    def analyze_box_distribution(self, patch_data):
        """
        ボックスタイプの分布を分析
        
        Args:
            patch_data: 解析済みのパッチデータ
            
        Returns:
            分布情報の辞書
        """
        if not patch_data or 'stats' not in patch_data:
            return {}
        
        # 基本統計から分布を計算
        stats = patch_data['stats']
        
        # 数値を集計
        total_boxes = sum(stats['box_types'].values())
        
        # パーセンテージを計算
        distribution = {
            'box_types': {
                k: {'count': v, 'percent': (v / total_boxes) * 100 if total_boxes else 0}
                for k, v in stats['box_types'].items()
            },
            'box_categories': {
                k: {'count': v, 'percent': (v / total_boxes) * 100 if total_boxes else 0}
                for k, v in stats['box_categories'].items()
            }
        }
        
        # newobjの場合はオブジェクト分布も計算
        newobj_count = stats['box_types'].get('newobj', 0)
        if newobj_count > 0:
            distribution['object_types'] = {
                k: {'count': v, 'percent': (v / newobj_count) * 100}
                for k, v in stats['object_types'].items()
            }
            distribution['object_categories'] = {
                k: {'count': v, 'percent': (v / newobj_count) * 100}
                for k, v in stats['object_categories'].items()
            }
        
        return distribution

def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='Max/MSPのボックスタイプとオブジェクトタイプを解析')
    parser.add_argument('--input', '-i', required=True, help='解析するパッチファイルのパス')
    parser.add_argument('--output', '-o', help='解析結果の出力先')
    parser.add_argument('--xml-port-data', '-x', help='XMLから抽出したポート情報ファイル')
    parser.add_argument('--patterns', '-p', action='store_true', help='接続パターンを生成')
    parser.add_argument('--distribution', '-d', action='store_true', help='ボックスタイプの分布を分析')
    parser.add_argument('--verbose', '-v', action='store_true', help='詳細情報を表示')
    
    return parser.parse_args()

def main():
    """メイン関数"""
    args = parse_args()
    
    # 入力ファイルを確認
    input_file = args.input
    if not os.path.exists(input_file):
        logger.error(f"入力ファイルが見つかりません: {input_file}")
        return 1
    
    # XML ポートデータを確認
    xml_port_data = args.xml_port_data
    if xml_port_data and not os.path.exists(xml_port_data):
        logger.warning(f"XMLポートデータファイルが見つかりません: {xml_port_data}")
        xml_port_data = None
    
    # アナライザーを初期化
    analyzer = BoxTypeAnalyzer(xml_port_data)
    
    logger.info(f"パッチファイルを解析中: {input_file}")
    
    # パッチファイルを解析
    patch_data = analyzer.analyze_patch_file(input_file)
    
    if not patch_data:
        logger.error("パッチファイルの解析に失敗しました")
        return 1
    
    # 追加分析
    if args.patterns:
        logger.info("接続パターンを生成中...")
        patterns = analyzer.generate_connection_patterns(patch_data)
        patch_data['connection_patterns'] = patterns
    
    if args.distribution:
        logger.info("ボックスタイプの分布を分析中...")
        distribution = analyzer.analyze_box_distribution(patch_data)
        patch_data['box_distribution'] = distribution
    
    # 解析結果を出力
    if args.output:
        logger.info(f"解析結果をファイルに保存: {args.output}")
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(patch_data, f, indent=2)
    
    # 詳細情報を表示
    if args.verbose:
        print("\n=== パッチ解析結果 ===")
        print(f"ファイル: {input_file}")
        print(f"ボックス数: {len(patch_data['boxes'])}")
        print(f"接続数: {len(patch_data['connections'])}")
        
        if 'stats' in patch_data:
            stats = patch_data['stats']
            print("\n== ボックスタイプ統計 ==")
            for box_type, count in sorted(stats['box_types'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {box_type}: {count}")
            
            if 'newobj' in stats['box_types']:
                print("\n== Top 10 オブジェクトタイプ ==")
                for obj_type, count in sorted(stats['object_types'].items(), key=lambda x: x[1], reverse=True)[:10]:
                    print(f"  {obj_type}: {count}")
        
        if 'subpatchers' in patch_data and patch_data['subpatchers']:
            print(f"\n== サブパッチャー数: {len(patch_data['subpatchers'])} ==")
            for i, subpatcher in enumerate(patch_data['subpatchers']):
                print(f"  {i+1}. タイプ: {subpatcher.get('type')}, ID: {subpatcher.get('id')}")
    
    logger.info("解析完了")
    return 0

if __name__ == "__main__":
    sys.exit(main())