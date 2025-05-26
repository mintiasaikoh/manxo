#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_integrated_features.py - 3つの主要機能（パッチ可視化システム、複合接続タイプ処理、UI要素のグラフ表現）
の統合テスト

このテストスクリプトは以下の機能の統合をテストします：
1. パッチ可視化システム
2. 複合接続タイプの処理
3. UI要素の特性を考慮したグラフ表現
"""

import unittest
import os
import sys
import tempfile
import json
import shutil
from typing import Dict, List, Any, Optional
import matplotlib
matplotlib.use('Agg')  # GUIを使わないバックエンド設定

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 機能1: パッチ可視化システム
from scripts.patch_graph_converter import PatchGraph
from scripts.structure_preserving_converter import StructurePreservingConverter
from src.enhanced_converter.enhanced_graph_converter import EnhancedGraphConverter
from src.enhanced_converter.box_converter import BoxConverter

# 機能2: 複合接続タイプの処理
from src.box_types import (
    BoxType, ObjectBox, MessageBox, NumberBox, FloatBox, CommentBox, 
    UIControlBox, BPatcherBox
)
from src.patch_parser import EnhancedPatchParser

# 機能3: UI要素の特性を考慮したグラフ表現
from tools.ui_node import (
    UIObjectNode, SliderNode, DialNode, ToggleNode, NumberBoxNode,
    MenuNode, UIElementAnalyzer, UIElementFactory, UIElementType
)
from tools.edge import Edge, EdgeFactory, ConnectionType

# サンプルデータ（テスト用の簡易パッチ）
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
            },
            "obj-9": {
                "box": {
                    "maxclass": "slider",
                    "patching_rect": [250.0, 250.0, 20.0, 140.0],
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": ["float"],
                    "parameter_enable": 0,
                    "id": "obj-9"
                }
            },
            "obj-10": {
                "box": {
                    "maxclass": "dial",
                    "patching_rect": [300.0, 250.0, 40.0, 40.0],
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": ["float"],
                    "parameter_enable": 0,
                    "id": "obj-10"
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
            },
            "line-4": {
                "patchline": {
                    "source": ["obj-9", 0],
                    "destination": ["obj-6", 0],
                    "id": "line-4"
                }
            },
            "line-5": {
                "patchline": {
                    "source": ["obj-10", 0],
                    "destination": ["obj-7", 1],
                    "id": "line-5",
                    "midpoints": [320.0, 300.0, 400.0, 300.0, 400.0, 140.0],
                    "order": 0
                }
            }
        }
    }
}

# 複合接続タイプを含むサンプルデータ
COMPLEX_CONNECTION_PATCH = {
    "patcher": {
        "fileversion": 1,
        "appversion": {"major": 8, "minor": 3, "revision": 1, "architecture": "x64"},
        "classnamespace": "box",
        "rect": [100, 100, 640, 480],
        "bglocked": 0,
        "openinpresentation": 0,
        "boxes": {
            "obj-1": {
                "box": {
                    "maxclass": "newobj",
                    "text": "cycle~ 440",
                    "patching_rect": [50.0, 100.0, 68.0, 22.0],
                    "numinlets": 2,
                    "numoutlets": 1,
                    "outlettype": ["signal"],
                    "id": "obj-1"
                }
            },
            "obj-2": {
                "box": {
                    "maxclass": "newobj",
                    "text": "gain~ 0.5",
                    "patching_rect": [50.0, 150.0, 68.0, 22.0],
                    "numinlets": 1,
                    "numoutlets": 1,
                    "outlettype": ["signal"],
                    "id": "obj-2"
                }
            },
            "obj-3": {
                "box": {
                    "maxclass": "newobj",
                    "text": "dac~",
                    "patching_rect": [50.0, 200.0, 68.0, 22.0],
                    "numinlets": 2,
                    "numoutlets": 0,
                    "id": "obj-3"
                }
            },
            "obj-4": {
                "box": {
                    "maxclass": "newobj",
                    "text": "jit.matrix 4 char 320 240",
                    "patching_rect": [200.0, 100.0, 150.0, 22.0],
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": ["jit_matrix", ""],
                    "id": "obj-4"
                }
            },
            "obj-5": {
                "box": {
                    "maxclass": "newobj",
                    "text": "jit.window",
                    "patching_rect": [200.0, 150.0, 68.0, 22.0],
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": ["", ""],
                    "id": "obj-5"
                }
            },
            "obj-6": {
                "box": {
                    "maxclass": "flonum",
                    "patching_rect": [300.0, 100.0, 50.0, 22.0],
                    "numinlets": 1,
                    "numoutlets": 2,
                    "outlettype": ["float", "bang"],
                    "parameter_enable": 0,
                    "id": "obj-6"
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
                    "source": ["obj-2", 0],
                    "destination": ["obj-3", 0],
                    "id": "line-2"
                }
            },
            "line-3": {
                "patchline": {
                    "source": ["obj-2", 0],
                    "destination": ["obj-3", 1],
                    "id": "line-3"
                }
            },
            "line-4": {
                "patchline": {
                    "source": ["obj-4", 0],
                    "destination": ["obj-5", 0],
                    "id": "line-4"
                }
            },
            "line-5": {
                "patchline": {
                    "source": ["obj-6", 0],
                    "destination": ["obj-1", 0],
                    "id": "line-5"
                }
            }
        }
    }
}

class TestIntegratedFeatures(unittest.TestCase):
    """3つの主要機能の統合テスト"""
    
    def setUp(self):
        """テスト環境の準備"""
        # テスト用ディレクトリの作成
        self.test_dir = tempfile.mkdtemp()
        
        # サンプルパッチの保存
        self.sample_patch_path = os.path.join(self.test_dir, "sample_patch.maxpat")
        with open(self.sample_patch_path, 'w', encoding='utf-8') as f:
            json.dump(SAMPLE_PATCH, f)
            
        # 複合接続サンプルの保存
        self.complex_patch_path = os.path.join(self.test_dir, "complex_patch.maxpat")
        with open(self.complex_patch_path, 'w', encoding='utf-8') as f:
            json.dump(COMPLEX_CONNECTION_PATCH, f)
        
        # パーサー、コンバーターなどのコンポーネントを初期化
        self.parser = EnhancedPatchParser()
        self.converter = EnhancedGraphConverter()
        self.box_converter = BoxConverter()
        
        # 結果ディレクトリの作成
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def tearDown(self):
        """テスト後のクリーンアップ"""
        # テスト用ディレクトリの削除
        shutil.rmtree(self.test_dir)
    
    def test_patch_visualization(self):
        """パッチ可視化システムのテスト"""
        # グラフ変換
        graph = self.converter.convert_file(self.sample_patch_path)
        self.assertIsNotNone(graph)
        
        # パッチの基本情報の確認
        self.assertEqual(len(graph.nodes), 10)  # 10個のノード
        self.assertEqual(len(graph.edges), 5)   # 5つの接続
        
        # 視覚化用ファイル名
        visualization_path = os.path.join(self.output_dir, "sample_patch_visualization.png")
        
        # グラフの視覚化（簡易版）- NetworkXとmatplotlibを使用
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            
            # NetworkXグラフの作成
            G = nx.DiGraph()
            
            # ノードの追加
            for node_id, node in graph.nodes.items():
                node_type = node.type
                node_label = node.properties.get("display_text", node_type)
                G.add_node(node_id, type=node_type, label=node_label)
            
            # エッジの追加
            for edge_id, edge in graph.edges.items():
                source_id = edge.source["node_id"]
                target_id = edge.target["node_id"]
                G.add_edge(source_id, target_id, type=edge.type)
            
            # グラフの描画
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, seed=42)  # レイアウトの決定
            
            # ノードの描画（タイプごとに色分け）
            node_colors = []
            for node in G.nodes:
                node_type = G.nodes[node]["type"]
                if node_type == "message":
                    node_colors.append('lightblue')
                elif node_type == "toggle":
                    node_colors.append('lightgreen')
                elif node_type == "slider":
                    node_colors.append('pink')
                elif node_type == "dial":
                    node_colors.append('orange')
                elif node_type in ["metro", "cycle~", "delay"]:
                    node_colors.append('yellow')
                elif node_type == "p":
                    node_colors.append('purple')
                else:
                    node_colors.append('gray')
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, alpha=0.8)
            
            # エッジの描画（タイプごとに色分け）
            signal_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'signal']
            control_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'control']
            
            nx.draw_networkx_edges(G, pos, edgelist=signal_edges, width=2, edge_color='red', arrows=True)
            nx.draw_networkx_edges(G, pos, edgelist=control_edges, width=1.5, edge_color='blue', arrows=True)
            
            # ラベルの描画
            nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]['label'] for n in G.nodes})
            
            plt.title('Max/MSP パッチの視覚化')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(visualization_path)
            plt.close()
            
            # ファイルの存在を確認
            self.assertTrue(os.path.exists(visualization_path))
            
        except ImportError:
            print("NetworkXまたはmatplotlibが見つかりません。視覚化テストをスキップします。")
    
    def test_complex_connection_types(self):
        """複合接続タイプの処理テスト"""
        # 複合接続を含むパッチの変換
        graph = self.converter.convert_file(self.complex_patch_path)
        self.assertIsNotNone(graph)
        
        # 基本情報の確認
        self.assertEqual(len(graph.nodes), 6)  # 6個のノード
        self.assertEqual(len(graph.edges), 5)  # 5つの接続
        
        # 接続タイプの検証
        signal_connections = 0
        matrix_connections = 0
        control_connections = 0
        
        for edge_id, edge in graph.edges.items():
            edge_type = edge.type
            source_id = edge.source["node_id"]
            target_id = edge.target["node_id"]
            
            source_node = graph.nodes[source_id]
            target_node = graph.nodes[target_id]
            
            # オーディオ接続の検出
            if "signal" in edge_type or (
                source_node.type in ["cycle~", "gain~"] and
                target_node.type in ["gain~", "dac~"]
            ):
                signal_connections += 1
            
            # Jitter接続の検出
            elif "jit_matrix" in edge_type or (
                source_node.type.startswith("jit.") and
                target_node.type.startswith("jit.")
            ):
                matrix_connections += 1
            
            # コントロール接続の検出
            else:
                control_connections += 1
        
        # それぞれの接続タイプの数を検証
        self.assertGreaterEqual(signal_connections, 3)  # 3つ以上のシグナル接続
        self.assertGreaterEqual(matrix_connections, 1)  # 1つ以上のマトリックス接続
        self.assertGreaterEqual(control_connections, 1)  # 1つ以上のコントロール接続
        
        # 接続タイプの統計情報の確認
        connection_types = graph.metadata.get("connection_types", {})
        self.assertIn("control", connection_types)
        
        # 接続タイプレポートの生成
        report_path = os.path.join(self.output_dir, "connection_types_report.json")
        connection_report = {
            "total_connections": len(graph.edges),
            "signal_connections": signal_connections,
            "matrix_connections": matrix_connections,
            "control_connections": control_connections,
            "connection_types": connection_types
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(connection_report, f, indent=2)
        
        # ファイルの存在を確認
        self.assertTrue(os.path.exists(report_path))
    
    def test_ui_elements_graph_representation(self):
        """UI要素の特性を考慮したグラフ表現のテスト"""
        # サンプルパッチからUI要素を抽出
        graph = self.converter.convert_file(self.sample_patch_path)
        self.assertIsNotNone(graph)
        
        # UI要素の抽出
        ui_elements = []
        for node_id, node in graph.nodes.items():
            node_type = node.type
            
            # UI要素の検出
            if node_type in ["slider", "dial", "toggle", "number", "flonum"]:
                ui_elements.append(node)
        
        # UI要素の数を検証
        self.assertGreaterEqual(len(ui_elements), 4)  # 4つ以上のUI要素があるはず
        
        # UI要素の特性を考慮したノードの生成
        enhanced_ui_elements = []
        for node in ui_elements:
            node_type = node.type
            node_id = node.id
            position = node.properties.get("position", {"x": 0, "y": 0})
            
            # UIElementFactoryを使ってUI要素を生成
            if node_type == "slider":
                ui_node = UIElementFactory.create_slider(
                    node_id, 0, 127, "vertical", 
                    current_value=node.properties.get("value", 0)
                )
            elif node_type == "dial":
                ui_node = UIElementFactory.create_dial(
                    node_id, 0, 1, 
                    current_value=node.properties.get("value", 0)
                )
            elif node_type == "toggle":
                ui_node = UIElementFactory.create_toggle(
                    node_id, 0, 
                    current_value=node.properties.get("value", 0)
                )
            elif node_type in ["number", "flonum"]:
                ui_node = UIElementFactory.create_number_box(
                    node_id, -1000, 1000, 
                    current_value=node.properties.get("value", 0)
                )
            else:
                # その他のUI要素タイプ
                continue
            
            # 位置情報の設定
            ui_node.set_position(position.get("x", 0), position.get("y", 0))
            enhanced_ui_elements.append(ui_node)
        
        # UI要素の数を検証
        self.assertGreaterEqual(len(enhanced_ui_elements), 3)  # 3つ以上の拡張UI要素があるはず
        
        # UI要素の解析
        ui_analysis = UIElementAnalyzer.analyze_ui_elements(enhanced_ui_elements)
        self.assertIsNotNone(ui_analysis)
        
        # UI要素のレイアウト分析
        layout_analysis = UIElementAnalyzer.analyze_layout(enhanced_ui_elements)
        self.assertIsNotNone(layout_analysis)
        
        # UI要素の接続関係を抽出
        ui_connections = []
        for edge_id, edge in graph.edges.items():
            source_id = edge.source["node_id"]
            target_id = edge.target["node_id"]
            
            source_node = graph.nodes[source_id]
            target_node = graph.nodes[target_id]
            
            # UIノードが含まれる接続を検出
            if source_node.type in ["slider", "dial", "toggle", "number", "flonum"]:
                ui_connections.append({
                    "source": source_id,
                    "source_type": source_node.type,
                    "target": target_id,
                    "target_type": target_node.type,
                    "edge_type": edge.type
                })
        
        # UI接続の検証
        self.assertTrue(len(ui_connections) > 0)
        
        # UIレポートの生成
        report_path = os.path.join(self.output_dir, "ui_elements_report.json")
        ui_report = {
            "ui_elements_count": len(enhanced_ui_elements),
            "ui_element_types": {e.element_type: 0 for e in enhanced_ui_elements},
            "ui_connections": ui_connections,
            "layout": {
                "bounding_box": layout_analysis["bounding_box"],
                "element_distribution": layout_analysis["element_distribution"]
            }
        }
        
        # UI要素タイプのカウント
        for element in enhanced_ui_elements:
            ui_report["ui_element_types"][element.element_type] += 1
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(ui_report, f, indent=2)
        
        # ファイルの存在を確認
        self.assertTrue(os.path.exists(report_path))
    
    def test_integrated_functionality(self):
        """3つの機能を統合した総合テスト"""
        # 1. グラフ変換
        graph = self.converter.convert_file(self.sample_patch_path)
        self.assertIsNotNone(graph)
        
        # 2. UI要素の特定と拡張
        ui_elements = []
        for node_id, node in graph.nodes.items():
            if node.type in ["slider", "dial", "toggle", "number", "flonum"]:
                # UIElementFactoryを使ったUI要素の生成
                position = node.properties.get("position", {"x": 0, "y": 0})
                ui_element = None
                
                if node.type == "slider":
                    ui_element = UIElementFactory.create_slider(node_id, 0, 127, "vertical")
                elif node.type == "dial":
                    ui_element = UIElementFactory.create_dial(node_id, 0, 1)
                elif node.type == "toggle":
                    ui_element = UIElementFactory.create_toggle(node_id, 0, False)
                elif node.type in ["number", "flonum"]:
                    ui_element = UIElementFactory.create_number_box(node_id, -1000, 1000, 2)
                
                if ui_element:
                    ui_element.set_position(position.get("x", 0), position.get("y", 0))
                    ui_elements.append(ui_element)
        
        # 3. 接続タイプの分析
        connection_types = {}
        for edge_id, edge in graph.edges.items():
            edge_type = edge.type
            if edge_type not in connection_types:
                connection_types[edge_type] = 0
            connection_types[edge_type] += 1
        
        # 4. 統合レポートの生成
        report_path = os.path.join(self.output_dir, "integrated_analysis_report.json")
        integrated_report = {
            "patch_info": {
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
                "ui_element_count": len(ui_elements),
                "connection_types": connection_types
            },
            "ui_elements": [
                {
                    "id": ui_element.id,
                    "type": ui_element.element_type,
                    "position": {"x": ui_element.position["x"], "y": ui_element.position["y"]}
                }
                for ui_element in ui_elements
            ],
            "connections": [
                {
                    "id": edge_id,
                    "source": edge.source["node_id"],
                    "source_port": edge.source["port"],
                    "target": edge.target["node_id"],
                    "target_port": edge.target["port"],
                    "type": edge.type
                }
                for edge_id, edge in graph.edges.items()
            ]
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(integrated_report, f, indent=2)
        
        # ファイルの存在を確認
        self.assertTrue(os.path.exists(report_path))
        
        # 5. 可視化の生成
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            
            # NetworkXグラフの作成
            G = nx.DiGraph()
            
            # ノードの追加
            for node_id, node in graph.nodes.items():
                node_type = node.type
                node_label = node.properties.get("display_text", node_type)
                # UIノードかどうかをフラグとして追加
                is_ui = node_type in ["slider", "dial", "toggle", "number", "flonum"]
                G.add_node(node_id, type=node_type, label=node_label, is_ui=is_ui)
            
            # エッジの追加（接続タイプを保持）
            for edge_id, edge in graph.edges.items():
                source_id = edge.source["node_id"]
                target_id = edge.target["node_id"]
                edge_type = edge.type
                G.add_edge(source_id, target_id, type=edge_type)
            
            # グラフの描画
            plt.figure(figsize=(14, 10))
            pos = nx.spring_layout(G, seed=42)  # レイアウトの決定
            
            # ノードの描画（タイプごとに色分け）
            ui_nodes = [n for n, d in G.nodes(data=True) if d['is_ui']]
            object_nodes = [n for n, d in G.nodes(data=True) if not d['is_ui'] and d['type'] not in ["message", "comment"]]
            message_nodes = [n for n, d in G.nodes(data=True) if d['type'] == "message"]
            comment_nodes = [n for n, d in G.nodes(data=True) if d['type'] == "comment"]
            
            # UI要素
            nx.draw_networkx_nodes(G, pos, nodelist=ui_nodes, node_color='lightgreen', 
                                 node_size=800, alpha=0.8, node_shape='o')
            
            # オブジェクト
            nx.draw_networkx_nodes(G, pos, nodelist=object_nodes, node_color='lightblue', 
                                 node_size=700, alpha=0.8, node_shape='s')
            
            # メッセージ
            nx.draw_networkx_nodes(G, pos, nodelist=message_nodes, node_color='orange', 
                                 node_size=600, alpha=0.8, node_shape='d')
            
            # コメント
            nx.draw_networkx_nodes(G, pos, nodelist=comment_nodes, node_color='lightgray', 
                                 node_size=500, alpha=0.6, node_shape='h')
            
            # エッジの描画（タイプごとに色分け）
            signal_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'signal']
            matrix_edges = [(u, v) for u, v, d in G.edges(data=True) if 'matrix' in d['type']]
            control_edges = [(u, v) for u, v, d in G.edges(data=True) 
                            if d['type'] != 'signal' and 'matrix' not in d['type']]
            
            nx.draw_networkx_edges(G, pos, edgelist=signal_edges, width=2, edge_color='red', 
                                 arrows=True, arrowsize=15)
            nx.draw_networkx_edges(G, pos, edgelist=matrix_edges, width=2, edge_color='purple', 
                                 arrows=True, arrowsize=15, style='dashed')
            nx.draw_networkx_edges(G, pos, edgelist=control_edges, width=1.5, edge_color='blue', 
                                 arrows=True, arrowsize=12)
            
            # ラベルの描画
            nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]['label'] for n in G.nodes})
            
            # 凡例の作成
            legend_elements = [
                mpatches.Patch(color='lightgreen', label='UI要素'),
                mpatches.Patch(color='lightblue', label='オブジェクト'),
                mpatches.Patch(color='orange', label='メッセージ'),
                mpatches.Patch(color='lightgray', label='コメント'),
                mpatches.Patch(color='red', label='シグナル接続'),
                mpatches.Patch(color='purple', label='マトリックス接続'),
                mpatches.Patch(color='blue', label='コントロール接続')
            ]
            plt.legend(handles=legend_elements, loc='upper left')
            
            plt.title('統合機能による Max/MSP パッチの詳細視覚化')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "integrated_visualization.png"))
            plt.close()
            
        except ImportError:
            print("NetworkXまたはmatplotlibが見つかりません。視覚化をスキップします。")
        
        # HTMLレポートの生成
        html_report_path = os.path.join(self.output_dir, "integrated_report.html")
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Max/MSPパッチ統合分析レポート</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .flex-container {{ display: flex; flex-wrap: wrap; }}
                .card {{ width: 300px; margin: 10px; padding: 15px; border: 1px solid #ccc; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .visualization {{ text-align: center; margin: 20px 0; }}
                .visualization img {{ max-width: 100%; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Max/MSPパッチ統合分析レポート</h1>
                
                <div class="section">
                    <h2>パッチ概要</h2>
                    <table>
                        <tr><th>項目</th><th>値</th></tr>
                        <tr><td>ノード数</td><td>{len(graph.nodes)}</td></tr>
                        <tr><td>接続数</td><td>{len(graph.edges)}</td></tr>
                        <tr><td>UI要素数</td><td>{len(ui_elements)}</td></tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>接続タイプ分析</h2>
                    <table>
                        <tr><th>接続タイプ</th><th>数</th></tr>
                        {' '.join(f'<tr><td>{ctype}</td><td>{count}</td></tr>' for ctype, count in connection_types.items())}
                    </table>
                </div>
                
                <div class="section">
                    <h2>UI要素</h2>
                    <div class="flex-container">
                        {' '.join(f'<div class="card"><h3>{ui.element_type}</h3><p>ID: {ui.id}</p><p>位置: ({ui.position["x"]}, {ui.position["y"]})</p></div>' for ui in ui_elements)}
                    </div>
                </div>
                
                <div class="section">
                    <h2>パッチ視覚化</h2>
                    <div class="visualization">
                        <img src="integrated_visualization.png" alt="パッチ視覚化">
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(html_report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # ファイルの存在を確認
        self.assertTrue(os.path.exists(html_report_path))

if __name__ == "__main__":
    unittest.main()