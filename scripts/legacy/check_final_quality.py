#!/usr/bin/env python3
import pickle

# 最終的なサンプルデータを確認
with open("data/graphs_sample/sample_with_ports.pkl", "rb") as f:
    graphs = pickle.load(f)
    
print("=== 最終的なポートタイプ情報付きグラフデータセット ===")
print(f"グラフ数: {len(graphs)}")

# ポートタイプの統計
total_edges = 0
valid_edges = 0
port_type_stats = {}

for graph in graphs:
    if hasattr(graph, "edge_port_types"):
        for src, tgt in graph.edge_port_types:
            total_edges += 1
            
            # unknownや空文字でない場合は有効
            if src and src != "unknown" and src != "" and tgt and tgt != "unknown" and tgt != "":
                valid_edges += 1
                
            # 統計を収集
            key = f"{src} → {tgt}"
            port_type_stats[key] = port_type_stats.get(key, 0) + 1
                
print(f"\n全グラフの統計:")
print(f"  総エッジ数: {total_edges:,}")
print(f"  有効なポートタイプペア: {valid_edges:,} ({valid_edges/total_edges*100:.1f}%)")
print(f"  unknownまたは空: {total_edges-valid_edges:,} ({(total_edges-valid_edges)/total_edges*100:.1f}%)")

# 有効な接続パターンの例
print(f"\n有効な接続パターン（上位10）:")
valid_patterns = {}
for k, v in port_type_stats.items():
    if "unknown" not in k and " →  " not in k and " → unknown" not in k:
        valid_patterns[k] = v
        
sorted_patterns = sorted(valid_patterns.items(), key=lambda x: x[1], reverse=True)
for pattern, count in sorted_patterns[:10]:
    print(f"  {pattern}: {count}回")
    
# ポートタイプの種類
src_types = set()
tgt_types = set()
for graph in graphs:
    if hasattr(graph, "edge_port_types"):
        for src, tgt in graph.edge_port_types:
            if src and src != "unknown":
                src_types.add(src)
            if tgt and tgt != "unknown":
                tgt_types.add(tgt)
                
print(f"\nユニークなポートタイプ:")
print(f"  ソース側: {len(src_types)}種類")
print(f"  ターゲット側: {len(tgt_types)}種類")
print(f"\nソース側の例: {list(sorted(src_types))[:10]}")
print(f"ターゲット側の例: {list(sorted(tgt_types))[:10]}")