#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
box_types.py - Max/MSPボックスタイプの包括的なクラス体系

このモジュールは、Max/MSPのパッチ内に存在する様々なボックスタイプを表現する
クラス階層を定義します。主要な目的は「オブジェクトボックス」と「オブジェクト」の
明確な区別を行い、各ボックスタイプの特性に基づいた処理を可能にすることです。
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Union, Tuple

class BoxType(Enum):
    """Max/MSPのボックスタイプを表す列挙型"""
    # 標準的なボックスタイプ
    OBJECT_BOX = "newobj"      # 標準的なオブジェクトボックス
    MESSAGE_BOX = "message"    # メッセージボックス
    NUMBER_BOX = "number"      # 整数ボックス
    FLOAT_BOX = "flonum"       # 浮動小数点ボックス
    COMMENT_BOX = "comment"    # コメントボックス
    
    # UI系ボックスタイプ
    TOGGLE = "toggle"          # トグルボタン
    SLIDER = "slider"          # スライダー
    BUTTON = "button"          # バンボタン
    DIAL = "dial"              # ダイアル
    MULTISLIDER = "multislider" # マルチスライダー
    PANEL = "panel"            # パネル
    FPIC = "fpic"              # 画像表示
    UMENU = "umenu"            # ポップアップメニュー
    KSLIDER = "kslider"        # キーボードスライダー
    RADIOGROUP = "radiogroup"  # ラジオボタングループ
    LISTBOX = "listbox"        # リストボックス
    LED = "led"                # LED表示
    METER = "meter~"           # メーター表示
    GAIN = "gain~"             # ゲインコントロール
    SCOPE = "scope~"           # オシロスコープ
    SPECTROSCOPE = "spectroscope~" # スペクトロスコープ
    TEXTBUTTON = "textbutton"  # テキストボタン
    
    # MSP特有のボックス
    EZDAC = "ezdac~"           # 簡易オーディオ出力
    EZADC = "ezadc~"           # 簡易オーディオ入力
    FILTERGRAPH = "filtergraph~" # フィルターグラフ
    FUNCTION = "function"      # 関数描画
    PATCHMATRIX = "patchmatrix" # パッチ行列
    
    # Live特有のボックス
    LIVE_DIAL = "live.dial"    # Live用ダイアル
    LIVE_SLIDER = "live.slider" # Live用スライダー
    LIVE_NUMBOX = "live.numbox" # Live用数値ボックス
    LIVE_MENU = "live.menu"    # Live用メニュー
    LIVE_TAB = "live.tab"      # Live用タブ
    LIVE_TOGGLE = "live.toggle" # Live用トグル
    LIVE_BUTTON = "live.button" # Live用ボタン
    LIVE_METER = "live.meter~" # Live用メーター
    LIVE_GAIN = "live.gain~"   # Live用ゲイン
    LIVE_TEXT = "live.text"    # Live用テキスト
    
    # その他特殊ボックス
    BPATCHER = "bpatcher"      # ビジュアルパッチャー
    PATCHER = "patcher"        # パッチャー
    FPIC = "fpic"              # 画像表示
    JSUI = "jsui"              # JavaScript UI
    JSPICTURE = "jspicture"    # JavaScript画像
    
    # Jitter特有のボックス
    JIT_WINDOW = "jit.window"  # Jitterウィンドウ
    JIT_PWINDOW = "jit.pwindow" # Jitterプレビューウィンドウ
    JIT_CELLBLOCK = "jit.cellblock" # Jitterセルブロック
    JIT_MATRIX = "jit.matrix"  # Jitterマトリックス
    JIT_GL_HANDLE = "jit.gl.handle" # JitterGL操作ハンドル
    JIT_GL_GRIDSHAPE = "jit.gl.gridshape" # JitterGLグリッド形状
    
    # Gen特有のボックス
    GEN_PATCHER = "gen~"       # Gen~パッチャー
    GEN_CODE = "codebox~"      # Gen~コードボックス
    
    @classmethod
    def from_maxclass(cls, maxclass: str) -> 'BoxType':
        """maxclass文字列からBoxTypeを返す"""
        try:
            return cls(maxclass)
        except ValueError:
            # Live.オブジェクトの処理
            if maxclass and maxclass.startswith("live."):
                for box_type in cls:
                    if box_type.value == maxclass:
                        return box_type
            
            # Jitter.オブジェクトの処理
            elif maxclass and maxclass.startswith("jit."):
                for box_type in cls:
                    if box_type.value == maxclass:
                        return box_type
                return cls.OBJECT_BOX
            
            # Gen特有のオブジェクト
            elif maxclass and (maxclass == "gen~" or maxclass == "codebox~"):
                for box_type in cls:
                    if box_type.value == maxclass:
                        return box_type
            
            # デフォルトはOBJECT_BOX
            return cls.OBJECT_BOX
    
    @property
    def is_ui_element(self) -> bool:
        """UIエレメントかどうかを判定"""
        ui_types = [
            BoxType.TOGGLE, BoxType.SLIDER, BoxType.BUTTON, BoxType.DIAL,
            BoxType.MULTISLIDER, BoxType.PANEL, BoxType.FPIC, BoxType.UMENU,
            BoxType.KSLIDER, BoxType.RADIOGROUP, BoxType.LISTBOX, BoxType.LED,
            BoxType.METER, BoxType.GAIN, BoxType.SCOPE, BoxType.SPECTROSCOPE,
            BoxType.TEXTBUTTON, BoxType.LIVE_DIAL, BoxType.LIVE_SLIDER,
            BoxType.LIVE_NUMBOX, BoxType.LIVE_MENU, BoxType.LIVE_TAB,
            BoxType.LIVE_TOGGLE, BoxType.LIVE_BUTTON, BoxType.LIVE_METER,
            BoxType.LIVE_GAIN, BoxType.LIVE_TEXT, BoxType.JSUI, BoxType.JSPICTURE
        ]
        return self in ui_types
    
    @property
    def is_msp_element(self) -> bool:
        """MSP関連要素かどうかを判定"""
        msp_types = [
            BoxType.EZDAC, BoxType.EZADC, BoxType.FILTERGRAPH,
            BoxType.METER, BoxType.GAIN, BoxType.SCOPE, BoxType.SPECTROSCOPE,
            BoxType.LIVE_METER, BoxType.LIVE_GAIN, BoxType.GEN_PATCHER,
            BoxType.GEN_CODE
        ]
        return self in msp_types
    
    @property
    def is_jitter_element(self) -> bool:
        """Jitter関連要素かどうかを判定"""
        jitter_types = [
            BoxType.JIT_WINDOW, BoxType.JIT_PWINDOW, BoxType.JIT_CELLBLOCK,
            BoxType.JIT_MATRIX, BoxType.JIT_GL_HANDLE, BoxType.JIT_GL_GRIDSHAPE
        ]
        return self in jitter_types
    
    @property
    def is_live_element(self) -> bool:
        """Live関連要素かどうかを判定"""
        live_types = [
            BoxType.LIVE_DIAL, BoxType.LIVE_SLIDER, BoxType.LIVE_NUMBOX,
            BoxType.LIVE_MENU, BoxType.LIVE_TAB, BoxType.LIVE_TOGGLE,
            BoxType.LIVE_BUTTON, BoxType.LIVE_METER, BoxType.LIVE_GAIN,
            BoxType.LIVE_TEXT
        ]
        return self in live_types

class MaxBox:
    """全てのMax/MSPボックスの基底クラス"""
    
    def __init__(
        self, 
        box_id: str, 
        box_type: BoxType, 
        position: List[float], 
        size: List[float]
    ):
        """ボックスの初期化
        
        Args:
            box_id: ボックスの一意ID
            box_type: ボックスのタイプ (BoxType列挙型)
            position: [x, y] 位置座標
            size: [width, height] サイズ
        """
        self.id = box_id
        self.box_type = box_type
        self.position = position
        self.size = size
        self.inlets = []  # インレット情報
        self.outlets = []  # アウトレット情報
        self.numinlets = 0
        self.numoutlets = 0
        self.properties = {}  # その他の属性
        self.parent_id = None  # 親パッチャーID (サブパッチの場合)
        self.hierarchy_level = 0  # 階層レベル
    
    def add_inlet(self, index: int, data_type: str, description: str = "", 
                 is_optional: bool = False) -> None:
        """インレット情報を追加"""
        self.inlets.append({
            "index": index,
            "type": data_type,
            "description": description,
            "is_optional": is_optional
        })
        self.numinlets = max(self.numinlets, index + 1)
    
    def add_outlet(self, index: int, data_type: str, description: str = "", 
                  is_optional: bool = False) -> None:
        """アウトレット情報を追加"""
        self.outlets.append({
            "index": index,
            "type": data_type,
            "description": description,
            "is_optional": is_optional
        })
        self.numoutlets = max(self.numoutlets, index + 1)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書表現を返す"""
        return {
            "id": self.id,
            "type": self.box_type.value,
            "position": self.position,
            "size": self.size,
            "inlets": self.inlets,
            "outlets": self.outlets,
            "numinlets": self.numinlets,
            "numoutlets": self.numoutlets,
            "properties": self.properties,
            "parent_id": self.parent_id,
            "hierarchy_level": self.hierarchy_level
        }
    
    @staticmethod
    def from_json_box(box_id: str, box_data: Dict[str, Any]) -> 'MaxBox':
        """JSONボックスデータから適切なボックスオブジェクトを生成"""
        maxclass = box_data.get("maxclass", "")
        box_type = BoxType.from_maxclass(maxclass)
        
        # パッチングレクト（位置とサイズ）
        rect = box_data.get("patching_rect", [0, 0, 0, 0])
        position = rect[:2] if len(rect) >= 2 else [0, 0]
        size = rect[2:4] if len(rect) >= 4 else [0, 0]
        
        # ボックスタイプに基づいて適切なクラスをインスタンス化
        if box_type == BoxType.OBJECT_BOX:
            return ObjectBox(box_id, position, size, box_data.get("text", ""))
        
        elif box_type == BoxType.MESSAGE_BOX:
            return MessageBox(box_id, position, size, box_data.get("text", ""))
        
        elif box_type == BoxType.NUMBER_BOX:
            return NumberBox(box_id, position, size)
        
        elif box_type == BoxType.FLOAT_BOX:
            return FloatBox(box_id, position, size)
        
        elif box_type == BoxType.COMMENT_BOX:
            return CommentBox(box_id, position, size, box_data.get("text", ""))
        
        # UI系コントロール
        elif box_type.is_ui_element:
            if box_type.is_live_element:
                return LiveUIBox(box_id, box_type, position, size, box_data)
            else:
                return UIControlBox(box_id, box_type, position, size, box_data)
        
        # MSP特化ボックス
        elif box_type.is_msp_element:
            if box_type in [BoxType.GEN_PATCHER, BoxType.GEN_CODE]:
                return GenBox(box_id, box_type, position, size, box_data)
            else:
                return MSPBox(box_id, box_type, position, size, box_data)
        
        # Jitter特化ボックス
        elif box_type.is_jitter_element:
            return JitterBox(box_id, box_type, position, size, box_data)
        
        # JavaScript関連ボックス
        elif box_type in [BoxType.JSUI, BoxType.JSPICTURE]:
            return JavaScriptBox(box_id, box_type, position, size, box_data)
        
        # ビジュアルパッチャー
        elif box_type == BoxType.BPATCHER:
            return BPatcherBox(box_id, position, size, box_data.get("name", ""))
        
        # パッチャー
        elif box_type == BoxType.PATCHER:
            return PatcherBox(box_id, position, size, box_data)
        
        # その他の型はGenericBoxとして処理
        else:
            return GenericBox(box_id, box_type, position, size, box_data)


class ObjectBox(MaxBox):
    """オブジェクトボックス（maxclass="newobj"）
    
    Max/MSPパッチ内で最も一般的なボックスタイプで、実際の処理を行う
    オブジェクトを含む。textフィールドの最初の単語が実際のオブジェクト名。
    """
    
    def __init__(self, box_id: str, position: List[float], size: List[float], text: str):
        """オブジェクトボックスの初期化
        
        Args:
            box_id: ボックスの一意ID
            position: [x, y] 位置座標
            size: [width, height] サイズ
            text: オブジェクトのテキスト（オブジェクト名と引数を含む）
        """
        super().__init__(box_id, BoxType.OBJECT_BOX, position, size)
        
        # オブジェクト名と引数を分離
        parts = text.split(None, 1)
        self.object_name = parts[0] if parts else ""
        self.arguments_text = parts[1] if len(parts) > 1 else ""
        self.arguments = self.arguments_text.split() if self.arguments_text else []
        
        # サブパッチャー情報
        self.has_subpatch = False
        self.subpatch = None  # サブパッチが存在する場合、子ボックスのリスト
    
    @property
    def is_msp_object(self) -> bool:
        """MSPオーディオオブジェクト（チルダ付き）かどうか"""
        return self.object_name.endswith('~')
    
    @property
    def is_jitter_object(self) -> bool:
        """Jitterオブジェクト（jit.プレフィックス）かどうか"""
        return self.object_name.startswith('jit.')
    
    @property
    def is_subpatcher(self) -> bool:
        """サブパッチャーを表すオブジェクトかどうか"""
        return self.object_name in ["p", "patcher", "poly~"] or self.has_subpatch
    
    @property
    def is_abstraction(self) -> bool:
        """外部抽象化を参照するかどうか
        
        サブパッチャーではなく、標準オブジェクトでもない場合は
        抽象化（外部パッチファイル）を参照している可能性が高い
        """
        return (not self.is_subpatcher and 
                not self.object_name in STANDARD_MAX_OBJECTS)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書表現を返す（オーバーライド）"""
        result = super().to_dict()
        result.update({
            "object_name": self.object_name,
            "arguments_text": self.arguments_text,
            "arguments": self.arguments,
            "is_msp_object": self.is_msp_object,
            "is_jitter_object": self.is_jitter_object,
            "is_subpatcher": self.is_subpatcher,
            "is_abstraction": self.is_abstraction,
            "has_subpatch": self.has_subpatch
        })
        return result


class MessageBox(MaxBox):
    """メッセージボックス（maxclass="message"）
    
    クリックされたときやトリガーされたときにメッセージを送信する。
    $1,$2などの変数置換やセミコロンによる送信先指定などの
    特殊な構文を持つ。
    """
    
    def __init__(self, box_id: str, position: List[float], size: List[float], text: str):
        """メッセージボックスの初期化"""
        super().__init__(box_id, BoxType.MESSAGE_BOX, position, size)
        self.message_text = text
        
        # 標準的なインレット/アウトレット設定
        self.add_inlet(0, "any", "メッセージを送信", False)
        self.add_inlet(1, "bang", "内容を設定", True)
        self.add_outlet(0, "any", "メッセージ出力", False)
        
        # メッセージの特殊構文を解析
        self.message_parts = self._parse_message(text)
        self.variables = self._extract_variables(text)
        self.variable_types = self._analyze_variable_types(text)
        self.has_forward = ";" in text
        self.forwarding_targets = self._extract_forwarding_targets(text)
        self.special_messages = self._identify_special_messages(text)
        self.comma_lists = self._extract_comma_lists(text)
    
    def _parse_message(self, text: str) -> List[Dict[str, Any]]:
        """メッセージを構造化された部分に分解して解析"""
        import re
        
        # 結果格納用
        parsed_parts = []
        
        # セミコロンでメッセージと転送先を分割
        segments = text.split(";")
        
        for i, segment in enumerate(segments):
            # 先頭のセグメントはメインメッセージ、それ以降は転送先
            is_main = (i == 0)
            segment_text = segment.strip()
            
            if not segment_text:
                continue
                
            # カンマでリスト要素を分割
            comma_parts = []
            if "," in segment_text:
                # 空白を含むカンマ区切りに対応するため、単純な split(',') は使わない
                # 例: "1, 2, test value, 3" -> ["1", "2", "test value", "3"]
                in_quote = False
                current_part = ""
                for char in segment_text:
                    if char == '"':
                        in_quote = not in_quote
                        current_part += char
                    elif char == ',' and not in_quote:
                        comma_parts.append(current_part.strip())
                        current_part = ""
                    else:
                        current_part += char
                
                # 最後の部分を追加
                if current_part:
                    comma_parts.append(current_part.strip())
            
            # 変数置換を検出
            variables = []
            variable_formats = []
            
            # 基本的な$N形式
            basic_vars = re.findall(r'\$([1-9]\d*)', segment_text)
            for var in basic_vars:
                variables.append(int(var))
                variable_formats.append("basic")
            
            # $N-形式 (残りの引数を結合)
            combine_vars = re.findall(r'\$([1-9]\d*)-', segment_text)
            for var in combine_vars:
                variables.append(int(var))
                variable_formats.append("combine")
            
            # $iN形式 (整数として扱う)
            int_vars = re.findall(r'\$i([1-9]\d*)', segment_text)
            for var in int_vars:
                variables.append(int(var))
                variable_formats.append("int")
            
            # $fN形式 (浮動小数点として扱う)
            float_vars = re.findall(r'\$f([1-9]\d*)', segment_text)
            for var in float_vars:
                variables.append(int(var))
                variable_formats.append("float")
            
            # $sN形式 (シンボルとして扱う)
            symbol_vars = re.findall(r'\$s([1-9]\d*)', segment_text)
            for var in symbol_vars:
                variables.append(int(var))
                variable_formats.append("symbol")
            
            # 特殊メッセージタイプを検出
            message_type = None
            special_message_types = [
                "set", "bang", "int", "float", "symbol", 
                "list", "clear", "append", "prepend"
            ]
            
            first_word = segment_text.split()[0] if segment_text.split() else ""
            if first_word in special_message_types:
                message_type = first_word
            
            # 解析結果を格納
            parsed_part = {
                "text": segment_text,
                "is_main": is_main,
                "variables": list(zip(variables, variable_formats)),
                "comma_list": comma_parts if comma_parts else None,
                "special_message_type": message_type,
                "target": first_word if not is_main else None
            }
            
            parsed_parts.append(parsed_part)
        
        return parsed_parts
    
    def _extract_variables(self, text: str) -> List[int]:
        """$1,$2などの変数参照を抽出（拡張版）"""
        import re
        
        # 様々な変数形式に対応するパターン
        patterns = [
            r'\$([1-9]\d*)',    # 基本的な$N
            r'\$([1-9]\d*)-',   # $N- (残りの引数を結合)
            r'\$i([1-9]\d*)',   # $iN (整数)
            r'\$f([1-9]\d*)',   # $fN (浮動小数点)
            r'\$s([1-9]\d*)'    # $sN (シンボル)
        ]
        
        all_matches = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            all_matches.extend([int(m) for m in matches])
        
        # 重複を排除して昇順ソート
        return sorted(set(all_matches))
    
    def _analyze_variable_types(self, text: str) -> Dict[int, str]:
        """変数の型を解析"""
        import re
        
        # 変数番号と型のマッピング
        var_types = {}
        
        # 型付き変数を検索
        typed_patterns = [
            (r'\$i([1-9]\d*)', "int"),      # 整数型
            (r'\$f([1-9]\d*)', "float"),    # 浮動小数点型
            (r'\$s([1-9]\d*)', "symbol"),   # シンボル型
        ]
        
        for pattern, var_type in typed_patterns:
            for match in re.finditer(pattern, text):
                var_num = int(match.group(1))
                var_types[var_num] = var_type
        
        # $N-形式（残りの引数を結合）
        for match in re.finditer(r'\$([1-9]\d*)-', text):
            var_num = int(match.group(1))
            var_types[var_num] = "combine"
        
        # 通常の$N型（既に他の型が割り当てられていなければ）
        for match in re.finditer(r'\$([1-9]\d*)', text):
            var_num = int(match.group(1))
            if var_num not in var_types:
                var_types[var_num] = "any"
        
        return var_types
    
    def _extract_forwarding_targets(self, text: str) -> List[Dict[str, Any]]:
        """セミコロン後の転送先と転送メッセージを抽出"""
        if ";" not in text:
            return []
        
        parts = text.split(";")
        # 最初の部分はメッセージ自体なので無視
        targets = []
        
        for i, part in enumerate(parts[1:], 1):
            part = part.strip()
            if not part:
                continue
                
            # 転送先名と残りのメッセージを分離
            dest_parts = part.split(None, 1)
            
            target = {
                "index": i,
                "target": dest_parts[0] if dest_parts else "",
                "message": dest_parts[1] if len(dest_parts) > 1 else "",
                "variables": self._extract_variables(part),
                "variable_types": self._analyze_variable_types(part)
            }
            
            targets.append(target)
        
        return targets
    
    def _identify_special_messages(self, text: str) -> List[str]:
        """特別なメッセージタイプを識別"""
        special_types = []
        
        # セミコロンで分割
        segments = text.split(";")
        
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
                
            # 最初の単語を取得
            words = segment.split()
            if not words:
                continue
                
            first_word = words[0]
            
            # 特別なメッセージタイプを確認
            if first_word in ["set", "bang", "int", "float", "symbol", 
                             "list", "clear", "append", "prepend"]:
                special_types.append(first_word)
        
        return special_types
    
    def _extract_comma_lists(self, text: str) -> List[List[str]]:
        """カンマ区切りのリストを抽出"""
        comma_lists = []
        
        # セミコロンで分割
        segments = text.split(";")
        
        for segment in segments:
            segment = segment.strip()
            
            # カンマが含まれているか確認
            if "," in segment:
                # 引用符内のカンマを考慮した分割（単純なsplit(',')は使わない）
                in_quote = False
                current_part = ""
                parts = []
                
                for char in segment:
                    if char == '"':
                        in_quote = not in_quote
                        current_part += char
                    elif char == ',' and not in_quote:
                        parts.append(current_part.strip())
                        current_part = ""
                    else:
                        current_part += char
                
                # 最後の部分を追加
                if current_part:
                    parts.append(current_part.strip())
                
                if parts:
                    comma_lists.append(parts)
        
        return comma_lists
    
    def get_dependency_objects(self) -> List[str]:
        """このメッセージボックスが依存するオブジェクト名を返す"""
        dependencies = []
        
        # 転送先のオブジェクト名を収集
        if self.has_forward:
            for target in self.forwarding_targets:
                target_name = target.get("target", "")
                if target_name and target_name not in dependencies:
                    dependencies.append(target_name)
        
        return dependencies
    
    def resolve_variable(self, var_num: int, arguments: List[Any]) -> Any:
        """変数参照を解決する"""
        if var_num <= 0 or var_num > len(arguments):
            return None
            
        # インデックスは0ベースに変換
        idx = var_num - 1
        
        # 変数の型に基づいて変換
        var_type = self.variable_types.get(var_num, "any")
        
        if var_type == "int":
            try:
                return int(arguments[idx])
            except (ValueError, TypeError):
                return 0
        elif var_type == "float":
            try:
                return float(arguments[idx])
            except (ValueError, TypeError):
                return 0.0
        elif var_type == "symbol":
            return str(arguments[idx])
        elif var_type == "combine":
            # $N-形式: N番目以降の引数をすべて結合
            return " ".join(str(arg) for arg in arguments[idx:])
        else:
            # 標準的な変数
            return arguments[idx]
    
    def resolve_message(self, arguments: List[Any]) -> str:
        """メッセージテキストを解決（変数置換を実行）"""
        if not arguments or not self.variables:
            return self.message_text
            
        resolved = self.message_text
        
        # 変数を置換
        for var_num in sorted(self.variable_types.keys(), reverse=True):
            var_type = self.variable_types[var_num]
            
            # 置換パターンを構築
            if var_type == "int":
                pattern = f"$i{var_num}"
            elif var_type == "float":
                pattern = f"$f{var_num}"
            elif var_type == "symbol":
                pattern = f"$s{var_num}"
            elif var_type == "combine":
                pattern = f"${var_num}-"
            else:
                pattern = f"${var_num}"
            
            # 変数の解決
            value = self.resolve_variable(var_num, arguments)
            
            if value is not None:
                resolved = resolved.replace(pattern, str(value))
        
        return resolved
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書表現を返す（オーバーライド）"""
        result = super().to_dict()
        result.update({
            "message_text": self.message_text,
            "message_parts": self.message_parts,
            "variables": self.variables,
            "variable_types": self.variable_types,
            "has_forward": self.has_forward,
            "forwarding_targets": self.forwarding_targets,
            "special_messages": self.special_messages,
            "comma_lists": self.comma_lists
        })
        return result


class NumberBox(MaxBox):
    """整数ボックス（maxclass="number"）
    
    整数値を表示・編集するためのUIオブジェクト。
    """
    
    def __init__(self, box_id: str, position: List[float], size: List[float]):
        """整数ボックスの初期化"""
        super().__init__(box_id, BoxType.NUMBER_BOX, position, size)
        self.value = 0
        
        # 標準的なインレット/アウトレット設定
        self.add_inlet(0, "int", "値を設定", False)
        self.add_outlet(0, "int", "整数値出力", False)
        self.add_outlet(1, "bang", "値が変更されたときにbang", True)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書表現を返す（オーバーライド）"""
        result = super().to_dict()
        result.update({
            "value": self.value
        })
        return result


class FloatBox(MaxBox):
    """浮動小数点ボックス（maxclass="flonum"）
    
    小数点を持つ値を表示・編集するためのUIオブジェクト。
    """
    
    def __init__(self, box_id: str, position: List[float], size: List[float]):
        """浮動小数点ボックスの初期化"""
        super().__init__(box_id, BoxType.FLOAT_BOX, position, size)
        self.value = 0.0
        
        # 標準的なインレット/アウトレット設定
        self.add_inlet(0, "float", "値を設定", False)
        self.add_outlet(0, "float", "浮動小数点値出力", False)
        self.add_outlet(1, "bang", "値が変更されたときにbang", True)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書表現を返す（オーバーライド）"""
        result = super().to_dict()
        result.update({
            "value": self.value
        })
        return result


class CommentBox(MaxBox):
    """コメントボックス（maxclass="comment"）
    
    パッチ内に説明テキストを表示するためのボックス。
    機能的な処理は行わない。
    """
    
    def __init__(self, box_id: str, position: List[float], size: List[float], text: str):
        """コメントボックスの初期化"""
        super().__init__(box_id, BoxType.COMMENT_BOX, position, size)
        self.comment_text = text
        
        # コメントにはポートがない
        self.numinlets = 0
        self.numoutlets = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書表現を返す（オーバーライド）"""
        result = super().to_dict()
        result.update({
            "comment_text": self.comment_text
        })
        return result


class UIControlBox(MaxBox):
    """UIコントロールボックス（スライダー、トグル、ボタンなど）
    
    様々なユーザーインターフェース要素を表すボックス。
    それぞれ独自のmaxclass値を持つ。
    """
    
    def __init__(
        self, 
        box_id: str, 
        box_type: BoxType, 
        position: List[float], 
        size: List[float],
        raw_data: Dict[str, Any] = None
    ):
        """UIコントロールボックスの初期化"""
        super().__init__(box_id, box_type, position, size)
        self.value = 0  # 初期値
        self.min_value = 0  # 最小値
        self.max_value = 127  # 最大値
        self.range_mode = 0  # 範囲モード
        self.parameter_name = ""  # パラメータ名
        
        # 生データから必要な情報を抽出
        if raw_data:
            self.min_value = raw_data.get("min", 0)
            self.max_value = raw_data.get("max", 127)
            self.parameter_name = raw_data.get("varname", "")
            self.value = raw_data.get("value", 0)
            # 追加のプロパティをコピー
            for key, value in raw_data.items():
                if key not in ["id", "maxclass", "patching_rect"]:
                    self.properties[key] = value
        
        # 標準的なインレット/アウトレット設定
        self.add_inlet(0, "any", "値を設定", False)
        
        # ボックスタイプに応じたアウトレット設定
        if box_type == BoxType.TOGGLE:
            self.add_outlet(0, "int", "トグル状態(0/1)", False)
        elif box_type == BoxType.BUTTON:
            self.add_outlet(0, "bang", "ボタンがクリックされたときにbang", False)
        elif box_type == BoxType.SLIDER:
            self.add_outlet(0, "float", "スライダー値", False)
        elif box_type == BoxType.DIAL:
            self.add_outlet(0, "float", "ダイアル値", False)
        elif box_type == BoxType.MULTISLIDER:
            self.add_outlet(0, "list", "複数値リスト", False)
            self.add_outlet(1, "int", "選択されたスロット番号", True)
        elif box_type == BoxType.KSLIDER:
            self.add_outlet(0, "int", "MIDI音程", False)
            self.add_outlet(1, "int", "MIDI速度", False)
        elif box_type == BoxType.UMENU:
            self.add_outlet(0, "int", "選択されたインデックス", False)
            self.add_outlet(1, "symbol", "選択されたアイテム", True)
        elif box_type == BoxType.RADIOGROUP:
            self.add_outlet(0, "int", "選択されたボタン番号", False)
        elif box_type == BoxType.LISTBOX:
            self.add_outlet(0, "list", "選択されたアイテム番号のリスト", False)
            self.add_outlet(1, "int", "クリックされたアイテム番号", True)
        elif box_type == BoxType.METER or box_type == BoxType.SCOPE or box_type == BoxType.SPECTROSCOPE:
            # 表示のみで出力なし
            pass
        elif box_type == BoxType.LED:
            # LEDは入力のみ
            pass
        elif box_type == BoxType.GAIN:
            self.add_outlet(0, "signal", "スケーリングされた信号", False)
            self.add_outlet(1, "float", "現在の値", True)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書表現を返す（オーバーライド）"""
        result = super().to_dict()
        result.update({
            "value": self.value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "range_mode": self.range_mode,
            "parameter_name": self.parameter_name
        })
        return result


class LiveUIBox(UIControlBox):
    """Live特有のUI要素
    
    Ableton Liveとの統合のために最適化されたUI要素。
    Live.* プレフィックスを持つオブジェクト。
    """
    
    def __init__(
        self, 
        box_id: str, 
        box_type: BoxType, 
        position: List[float], 
        size: List[float],
        raw_data: Dict[str, Any]
    ):
        """Live UI要素の初期化"""
        super().__init__(box_id, box_type, position, size, raw_data)
        
        # Live特有のプロパティ
        self.parameter_enable = raw_data.get("parameter_enable", 0)
        self.live_shortname = raw_data.get("shortname", "")
        self.live_display_mode = raw_data.get("fontsize", 0)
        self.automation_name = raw_data.get("automation_name", "")
        
        # アウトレット設定を調整
        self.outlets = []  # リセット
        self.numoutlets = 0
        
        # Live要素に応じたアウトレット設定
        if box_type in [BoxType.LIVE_DIAL, BoxType.LIVE_SLIDER, BoxType.LIVE_NUMBOX]:
            self.add_outlet(0, "float", "現在の値", False)
            self.add_outlet(1, "bang", "値が変更されたとき", True)
        elif box_type == BoxType.LIVE_TOGGLE:
            self.add_outlet(0, "int", "トグル状態(0/1)", False)
        elif box_type == BoxType.LIVE_BUTTON:
            self.add_outlet(0, "bang", "ボタンがクリックされたとき", False)
        elif box_type == BoxType.LIVE_MENU:
            self.add_outlet(0, "int", "選択されたインデックス", False)
            self.add_outlet(1, "symbol", "選択されたアイテム", True)
        elif box_type == BoxType.LIVE_TAB:
            self.add_outlet(0, "int", "選択されたタブインデックス", False)
        elif box_type == BoxType.LIVE_TEXT:
            self.add_outlet(0, "bang", "テキストがクリックされたとき", False)
        elif box_type == BoxType.LIVE_METER:
            # 表示のみで出力なし
            pass
        elif box_type == BoxType.LIVE_GAIN:
            self.add_outlet(0, "signal", "スケーリングされた信号", False)
            self.add_inlet(0, "signal", "入力信号", False)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書表現を返す（オーバーライド）"""
        result = super().to_dict()
        result.update({
            "parameter_enable": self.parameter_enable,
            "live_shortname": self.live_shortname,
            "live_display_mode": self.live_display_mode,
            "automation_name": self.automation_name
        })
        return result


class MSPBox(MaxBox):
    """MSP特有のボックス
    
    オーディオ処理に特化したボックス。
    MSP要素には通常シグナルポートがある。
    """
    
    def __init__(
        self, 
        box_id: str, 
        box_type: BoxType, 
        position: List[float], 
        size: List[float],
        raw_data: Dict[str, Any]
    ):
        """MSPボックスの初期化"""
        super().__init__(box_id, box_type, position, size)
        
        # MSP固有のプロパティ
        self.msp_properties = {}
        for key, value in raw_data.items():
            if key not in ["id", "maxclass", "patching_rect", "text"]:
                self.msp_properties[key] = value
                self.properties[key] = value
        
        # ボックスタイプに応じたインレット/アウトレット設定
        if box_type == BoxType.EZDAC:
            self.add_inlet(0, "signal", "左チャンネル信号", False)
            self.add_inlet(1, "signal", "右チャンネル信号", False)
        elif box_type == BoxType.EZADC:
            self.add_outlet(0, "signal", "左チャンネル信号", False)
            self.add_outlet(1, "signal", "右チャンネル信号", False)
        elif box_type == BoxType.FILTERGRAPH:
            self.add_inlet(0, "list", "フィルター設定", False)
            self.add_outlet(0, "list", "フィルター設定", False)
        elif box_type == BoxType.FUNCTION:
            self.add_inlet(0, "bang", "出力を送信", False)
            self.add_inlet(1, "list", "関数ポイントのリスト", True)
            self.add_outlet(0, "list", "関数値", False)
            self.add_outlet(1, "bang", "完了時", True)
        elif box_type == BoxType.GAIN:
            self.add_inlet(0, "signal", "入力信号", False)
            self.add_outlet(0, "signal", "スケーリングされた信号", False)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書表現を返す（オーバーライド）"""
        result = super().to_dict()
        result.update({
            "msp_properties": self.msp_properties
        })
        return result


class JitterBox(MaxBox):
    """Jitter特有のボックス
    
    マトリックス処理とビジュアル出力に特化したボックス。
    通常jit.プレフィックスを持つ。
    """
    
    def __init__(
        self, 
        box_id: str, 
        box_type: BoxType, 
        position: List[float], 
        size: List[float],
        raw_data: Dict[str, Any]
    ):
        """Jitterボックスの初期化"""
        super().__init__(box_id, box_type, position, size)
        
        # Jitter固有のプロパティ
        self.jitter_properties = {}
        for key, value in raw_data.items():
            if key not in ["id", "maxclass", "patching_rect", "text"]:
                self.jitter_properties[key] = value
                self.properties[key] = value
        
        # ボックスタイプに応じたインレット/アウトレット設定
        if box_type == BoxType.JIT_WINDOW:
            self.add_inlet(0, "jit_matrix", "表示するマトリックス", False)
            self.add_inlet(1, "bang", "表示を更新", True)
        elif box_type == BoxType.JIT_PWINDOW:
            self.add_inlet(0, "jit_matrix", "表示するマトリックス", False)
        elif box_type == BoxType.JIT_MATRIX:
            self.add_inlet(0, "jit_matrix", "入力マトリックス", False)
            self.add_inlet(1, "bang", "処理を開始", True)
            self.add_outlet(0, "jit_matrix", "出力マトリックス", False)
        elif box_type == BoxType.JIT_CELLBLOCK:
            self.add_inlet(0, "jit_matrix", "表示するマトリックス", False)
            self.add_inlet(1, "list", "セル座標と値", True)
            self.add_outlet(0, "list", "選択されたセル座標", False)
            self.add_outlet(1, "bang", "編集時", True)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書表現を返す（オーバーライド）"""
        result = super().to_dict()
        result.update({
            "jitter_properties": self.jitter_properties
        })
        return result


class JavaScriptBox(MaxBox):
    """JavaScriptボックス
    
    JavaScriptコードを実行するためのボックス。
    jsui, js, jsextなどのオブジェクト。
    """
    
    def __init__(
        self, 
        box_id: str, 
        box_type: BoxType, 
        position: List[float], 
        size: List[float],
        raw_data: Dict[str, Any]
    ):
        """JavaScriptボックスの初期化"""
        super().__init__(box_id, box_type, position, size)
        
        # JSファイル名
        self.js_filename = raw_data.get("filename", "")
        if not self.js_filename and "text" in raw_data:
            # text属性にファイル名が格納されていることもある
            text = raw_data.get("text", "")
            if text:
                parts = text.split()
                if len(parts) > 0:
                    self.js_filename = parts[0]
        
        # JS固有のプロパティ
        self.js_properties = {}
        for key, value in raw_data.items():
            if key not in ["id", "maxclass", "patching_rect", "text"]:
                self.js_properties[key] = value
                self.properties[key] = value
        
        # ボックスタイプに応じたインレット/アウトレット設定
        # jsui/jspictureは描画専用なので入力のみ
        if box_type == BoxType.JSUI:
            self.add_inlet(0, "bang", "再描画", False)
            self.add_inlet(1, "list", "引数", True)
        elif box_type == BoxType.JSPICTURE:
            self.add_inlet(0, "bang", "再描画", False)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書表現を返す（オーバーライド）"""
        result = super().to_dict()
        result.update({
            "js_filename": self.js_filename,
            "js_properties": self.js_properties
        })
        return result


class PatcherBox(MaxBox):
    """パッチャーボックス
    
    通常のオブジェクトボックスと似ているが、サブパッチャーを含む特殊なボックス。
    patcher, p などのオブジェクト。
    """
    
    def __init__(
        self, 
        box_id: str, 
        position: List[float], 
        size: List[float],
        raw_data: Dict[str, Any]
    ):
        """パッチャーボックスの初期化"""
        super().__init__(box_id, BoxType.PATCHER, position, size)
        
        # パッチャーの名前
        self.patcher_name = ""
        if "text" in raw_data:
            self.patcher_name = raw_data.get("text", "")
        
        # サブパッチャー情報
        self.has_subpatch = True
        self.subpatch = None  # 後でパーサーによって設定される
        
        # サブパッチャーの参照情報
        self.patcher_properties = {}
        for key, value in raw_data.items():
            if key not in ["id", "maxclass", "patching_rect", "text"]:
                self.patcher_properties[key] = value
                self.properties[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書表現を返す（オーバーライド）"""
        result = super().to_dict()
        result.update({
            "patcher_name": self.patcher_name,
            "has_subpatch": self.has_subpatch,
            "patcher_properties": self.patcher_properties
        })
        return result


class GenBox(MaxBox):
    """Gen特有のボックス
    
    Genパッチャーとコードボックスを表すボックス。
    """
    
    def __init__(
        self, 
        box_id: str, 
        box_type: BoxType, 
        position: List[float], 
        size: List[float],
        raw_data: Dict[str, Any]
    ):
        """Genボックスの初期化"""
        super().__init__(box_id, box_type, position, size)
        
        # Gen固有のプロパティ
        self.gen_properties = {}
        for key, value in raw_data.items():
            if key not in ["id", "maxclass", "patching_rect", "text"]:
                self.gen_properties[key] = value
                self.properties[key] = value
        
        # ファイル名またはコード
        self.filename = raw_data.get("filename", "")
        self.code = raw_data.get("code", "")
        
        # サブパッチャー情報（gen~の場合）
        self.has_subpatch = box_type == BoxType.GEN_PATCHER
        self.subpatch = None  # 後でパーサーによって設定される
        
        # ボックスタイプに応じたインレット/アウトレット設定
        if box_type == BoxType.GEN_PATCHER:
            # デフォルトは1つの入出力、実際のポート数は後で設定される
            self.add_inlet(0, "signal", "入力信号", False)
            self.add_outlet(0, "signal", "出力信号", False)
        elif box_type == BoxType.GEN_CODE:
            # コードボックスはサンプル単位の処理なので信号ポートのみ
            self.add_inlet(0, "signal", "入力信号", False)
            self.add_outlet(0, "signal", "出力信号", False)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書表現を返す（オーバーライド）"""
        result = super().to_dict()
        result.update({
            "gen_properties": self.gen_properties,
            "filename": self.filename,
            "code": self.code,
            "has_subpatch": self.has_subpatch
        })
        return result


class BPatcherBox(MaxBox):
    """bpatcherボックス（maxclass="bpatcher"）
    
    外部パッチの視覚的な要素を表示するための特殊なボックス。
    """
    
    def __init__(
        self,
        box_id: str,
        position: List[float],
        size: List[float],
        name: str
    ):
        """bpatcherボックスの初期化"""
        super().__init__(box_id, BoxType.BPATCHER, position, size)
        self.patch_name = name  # 参照先のパッチ名
        self.bgmode = 0  # 背景モード
        self.border = 0  # 境界線の状態
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書表現を返す（オーバーライド）"""
        result = super().to_dict()
        result.update({
            "patch_name": self.patch_name,
            "bgmode": self.bgmode,
            "border": self.border
        })
        return result


class GenericBox(MaxBox):
    """その他の一般的なボックス
    
    特定のクラスに分類できないボックスを処理するための汎用クラス。
    """
    
    def __init__(
        self,
        box_id: str,
        box_type: BoxType,
        position: List[float],
        size: List[float],
        raw_data: Dict[str, Any]
    ):
        """一般的なボックスの初期化"""
        super().__init__(box_id, box_type, position, size)
        self.raw_data = raw_data
        
        # インレット/アウトレット数の設定
        self.numinlets = raw_data.get("numinlets", 0)
        self.numoutlets = raw_data.get("numoutlets", 0)
        
        # インレット/アウトレットタイプの取得と設定
        inlet_types = ["any"] * self.numinlets
        outlet_types = []
        
        # アウトレットタイプの処理
        outlettype = raw_data.get("outlettype", [])
        if isinstance(outlettype, list):
            outlet_types = outlettype
        
        # インレット/アウトレット情報の追加
        for i in range(self.numinlets):
            self.add_inlet(i, inlet_types[i] if i < len(inlet_types) else "any", 
                          f"インレット {i}", i > 0)
        
        for i in range(self.numoutlets):
            outlet_type = outlet_types[i] if i < len(outlet_types) else "any"
            self.add_outlet(i, outlet_type, f"アウトレット {i}", False)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書表現を返す（オーバーライド）"""
        result = super().to_dict()
        # raw_dataはJSON化できないかもしれないので除外する場合もある
        # ここでは含めておく
        result.update({
            "raw_data": self.raw_data
        })
        return result


# 標準的なMax/MSPオブジェクトの一部（参照用）
STANDARD_MAX_OBJECTS = {
    # Core
    "metro", "counter", "select", "route", "pak", "pack", "gate", 
    
    # UI
    "slider", "toggle", "button", "number", "flonum",
    
    # MSP
    "cycle~", "dac~", "adc~", "gain~", "sig~", "*~", "+~",
    
    # Jitter
    "jit.matrix", "jit.window", "jit.pwindow", "jit.gl.render"
}