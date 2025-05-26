import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from collections import defaultdict, Counter
from datetime import datetime

from node import Node
from edge import Edge, ConnectionType, EdgeDirection, EdgeAnalyzer
from patch_graph import PatchGraph

class ConnectionPatternAnalyzer:
    """Advanced analysis of connection patterns in Max/MSP patches"""
    
    def __init__(self, patch_graph: PatchGraph):
        self.graph = patch_graph
        self.nodes = patch_graph.nodes
        self.edges = patch_graph.edges
        self.connection_matrices = {}  # Cached connection matrices by type
    
    def generate_connection_matrix(self, connection_type: Optional[str] = None) -> pd.DataFrame:
        """Generate a connection matrix for a specific connection type (or all types if None)"""
        # Check if we already have this matrix cached
        cache_key = connection_type if connection_type else "all"
        if cache_key in self.connection_matrices:
            return self.connection_matrices[cache_key]
        
        # Get all node IDs
        node_ids = list(self.nodes.keys())
        
        # Create empty matrix
        matrix = pd.DataFrame(0, index=node_ids, columns=node_ids)
        
        # Fill the matrix
        for edge in self.edges.values():
            # Skip edges that don't match the requested type
            if connection_type and not edge.has_type(connection_type):
                continue
                
            source = edge.source["node_id"]
            target = edge.target["node_id"]
            
            if source in node_ids and target in node_ids:
                # Increment connection count
                matrix.at[source, target] += 1
                
                # Handle bidirectional edges
                if edge.direction == EdgeDirection.BIDIRECTIONAL.value:
                    matrix.at[target, source] += 1
        
        # Cache the result
        self.connection_matrices[cache_key] = matrix
        return matrix
    
    def detect_hubs(self, threshold: int = 3) -> Dict[str, Dict[str, int]]:
        """Detect hub nodes in the patch (nodes with many connections)"""
        matrix = self.generate_connection_matrix()
        
        # Calculate in-degree and out-degree for each node
        in_degree = matrix.sum(axis=0)
        out_degree = matrix.sum(axis=1)
        total_degree = in_degree + out_degree
        
        # Find hubs (nodes with high connection counts)
        hubs = {
            "in_hubs": {node: count for node, count in in_degree.items() if count >= threshold},
            "out_hubs": {node: count for node, count in out_degree.items() if count >= threshold},
            "total_hubs": {node: count for node, count in total_degree.items() if count >= threshold}
        }
        
        return hubs
    
    def detect_connection_clusters(self) -> Dict[str, List[str]]:
        """Detect clusters of nodes that are heavily interconnected"""
        # Convert to networkx graph for community detection
        G = nx.DiGraph()
        
        # Add all nodes
        for node_id in self.nodes:
            G.add_node(node_id)
        
        # Add all edges
        for edge in self.edges.values():
            source = edge.source["node_id"]
            target = edge.target["node_id"]
            
            # Skip self-loops or missing nodes
            if source == target or source not in self.nodes or target not in self.nodes:
                continue
                
            # Add the edge with the connection type as an attribute
            G.add_edge(source, target, types=list(edge.connection_types))
        
        # Use Louvain community detection algorithm
        communities = nx.community.louvain_communities(G.to_undirected())
        
        # Convert to dictionary
        clusters = {}
        for i, community in enumerate(communities):
            clusters[f"cluster_{i+1}"] = list(community)
            
        return clusters
    
    def analyze_connection_type_flows(self) -> Dict[str, Any]:
        """Analyze how different connection types flow through the patch"""
        # Get connection type counts by node
        node_connection_types = defaultdict(Counter)
        
        for edge in self.edges.values():
            source = edge.source["node_id"]
            target = edge.target["node_id"]
            
            for conn_type in edge.connection_types:
                # Count outgoing connections by type
                node_connection_types[source]["out_" + conn_type] += 1
                # Count incoming connections by type
                node_connection_types[target]["in_" + conn_type] += 1
        
        # Identify nodes that convert between connection types
        type_converters = {}
        for node_id, type_counts in node_connection_types.items():
            # Check for conversion between types
            in_types = {k[3:]: v for k, v in type_counts.items() if k.startswith("in_")}
            out_types = {k[4:]: v for k, v in type_counts.items() if k.startswith("out_")}
            
            # A node is a converter if it has different incoming and outgoing connection types
            if set(in_types.keys()) != set(out_types.keys()):
                type_converters[node_id] = {
                    "in_types": in_types,
                    "out_types": out_types,
                    "node_type": self.nodes[node_id].type if node_id in self.nodes else "unknown"
                }
        
        # Create flow paths (sequences of nodes connected by the same type)
        flow_paths = {}
        for conn_type in ConnectionType:
            type_value = conn_type.value
            if type_value == ConnectionType.COMPOUND.value:
                continue  # Skip compound type
                
            # Find all edges of this type
            type_edges = [e for e in self.edges.values() if e.has_type(type_value)]
            
            # Build a subgraph for this connection type
            G = nx.DiGraph()
            for edge in type_edges:
                G.add_edge(edge.source["node_id"], edge.target["node_id"])
            
            # Find paths in this subgraph
            paths = []
            for source in G.nodes():
                for target in G.nodes():
                    if source != target:
                        try:
                            # Find all simple paths between source and target
                            simple_paths = list(nx.all_simple_paths(G, source, target, cutoff=5))
                            paths.extend(simple_paths)
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            continue
            
            # Store up to 20 longest paths
            paths.sort(key=len, reverse=True)
            flow_paths[type_value] = paths[:20]
        
        return {
            "node_connection_types": {node: dict(types) for node, types in node_connection_types.items()},
            "type_converters": type_converters,
            "flow_paths": flow_paths
        }
    
    def find_signal_processing_chains(self) -> List[List[str]]:
        """Find chains of nodes involved in audio signal processing"""
        return self._find_typed_chains(ConnectionType.SIGNAL.value)
    
    def find_matrix_processing_chains(self) -> List[List[str]]:
        """Find chains of nodes involved in Jitter matrix processing"""
        return self._find_typed_chains(ConnectionType.MATRIX.value)
    
    def _find_typed_chains(self, connection_type: str) -> List[List[str]]:
        """Find chains of nodes connected by a specific connection type"""
        # Find all edges of this type
        type_edges = [e for e in self.edges.values() if e.has_type(connection_type)]
        
        # Build a subgraph for this connection type
        G = nx.DiGraph()
        for edge in type_edges:
            G.add_edge(edge.source["node_id"], edge.target["node_id"])
        
        # Find all simple paths in this subgraph (limit to length 10)
        chains = []
        for source in G.nodes():
            # Find nodes that have no incoming edges (sources) or have a different connection type coming in
            if G.in_degree(source) == 0:
                for target in G.nodes():
                    # Find nodes that have no outgoing edges (sinks) or have a different connection type going out
                    if source != target and G.out_degree(target) == 0:
                        try:
                            # Find all simple paths between source and target
                            simple_paths = list(nx.all_simple_paths(G, source, target, cutoff=10))
                            chains.extend(simple_paths)
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            continue
        
        # Sort by length (longest first) and return top 20
        chains.sort(key=len, reverse=True)
        return chains[:20]
    
    def analyze_bidirectional_patterns(self) -> Dict[str, Any]:
        """Analyze bidirectional connection patterns"""
        # Find all bidirectional edges
        bidirectional_edges = [e for e in self.edges.values() 
                              if e.direction == EdgeDirection.BIDIRECTIONAL.value]
        
        # Build a subgraph for bidirectional connections
        G = nx.Graph()  # Undirected graph for bidirectional edges
        
        for edge in bidirectional_edges:
            source = edge.source["node_id"]
            target = edge.target["node_id"]
            
            # Get node types if available
            source_type = self.nodes[source].type if source in self.nodes else "unknown"
            target_type = self.nodes[target].type if target in self.nodes else "unknown"
            
            # Add edge with node type information
            G.add_edge(source, target, 
                       source_type=source_type, 
                       target_type=target_type,
                       conn_types=list(edge.connection_types))
        
        # Analyze bidirectional connection patterns
        bidir_stats = {
            "count": len(bidirectional_edges),
            "nodes_involved": len(G.nodes()),
            "connection_types": Counter([conn_type for edge in bidirectional_edges 
                                        for conn_type in edge.connection_types]),
            "common_node_type_pairs": Counter(),
            "cliques": []
        }
        
        # Find common node type pairs in bidirectional connections
        for u, v, data in G.edges(data=True):
            source_type = data["source_type"]
            target_type = data["target_type"]
            pair = tuple(sorted([source_type, target_type]))
            bidir_stats["common_node_type_pairs"][pair] += 1
        
        # Find cliques (groups of fully interconnected nodes)
        cliques = list(nx.find_cliques(G))
        cliques.sort(key=len, reverse=True)
        bidir_stats["cliques"] = cliques[:10]  # Keep top 10 largest cliques
        
        return bidir_stats
    
    def create_subgraph_by_type(self, connection_type: str) -> PatchGraph:
        """Create a subgraph containing only connections of a specific type"""
        # Create a new graph with the same ID but specialized for a connection type
        subgraph = PatchGraph(
            f"{self.graph.id}_{connection_type}",
            f"{self.graph.name} - {connection_type} connections"
        )
        
        # Find all edges with this connection type
        type_edges = {edge_id: edge for edge_id, edge in self.edges.items() 
                     if edge.has_type(connection_type)}
        
        # Get all nodes involved in these edges
        involved_nodes = set()
        for edge in type_edges.values():
            involved_nodes.add(edge.source["node_id"])
            involved_nodes.add(edge.target["node_id"])
        
        # Add all involved nodes to the subgraph
        for node_id in involved_nodes:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                subgraph.add_node(node)
        
        # Add all edges of this type to the subgraph
        for edge_id, edge in type_edges.items():
            subgraph.add_edge(edge)
        
        return subgraph
    
    def compare_connection_types(self) -> Dict[str, Any]:
        """Compare different connection types in the patch"""
        # Get statistics for each connection type
        type_stats = {}
        
        for conn_type in ConnectionType:
            type_value = conn_type.value
            if type_value == ConnectionType.COMPOUND.value:
                continue  # Skip compound type
                
            # Find all edges with this type
            type_edges = [e for e in self.edges.values() if e.has_type(type_value)]
            
            if not type_edges:
                continue  # Skip if no edges of this type
                
            # Calculate basic statistics
            type_stats[type_value] = {
                "count": len(type_edges),
                "bidirectional": sum(1 for e in type_edges if e.direction == EdgeDirection.BIDIRECTIONAL.value),
                "avg_midpoints": np.mean([len(e.midpoints) for e in type_edges]),
                "node_count": len(set([e.source["node_id"] for e in type_edges] + 
                                     [e.target["node_id"] for e in type_edges]))
            }
        
        # Calculate correlation between different connection type matrices
        correlations = {}
        for type1 in type_stats:
            for type2 in type_stats:
                if type1 >= type2:  # Only compute once per pair
                    continue
                    
                matrix1 = self.generate_connection_matrix(type1)
                matrix2 = self.generate_connection_matrix(type2)
                
                # Mask non-shared nodes
                shared_nodes = list(set(matrix1.index) & set(matrix2.index))
                
                if shared_nodes:
                    # Calculate correlation on shared nodes
                    m1 = matrix1.loc[shared_nodes, shared_nodes].values.flatten()
                    m2 = matrix2.loc[shared_nodes, shared_nodes].values.flatten()
                    
                    if len(m1) > 0 and np.var(m1) > 0 and np.var(m2) > 0:
                        corr = np.corrcoef(m1, m2)[0, 1]
                        correlations[f"{type1}_{type2}"] = corr
        
        return {
            "type_stats": type_stats,
            "correlations": correlations,
            "type_counts": {
                type_value: sum(1 for e in self.edges.values() if e.has_type(type_value))
                for type_value in [t.value for t in ConnectionType]
            }
        }
    
    def generate_connection_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report on connection patterns"""
        report = {
            "patch_id": self.graph.id,
            "patch_name": self.graph.name,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "timestamp": datetime.now().isoformat(),
            
            # Run all analyses
            "hub_analysis": self.detect_hubs(),
            "clusters": self.detect_connection_clusters(),
            "connection_flow": self.analyze_connection_type_flows(),
            "bidirectional_patterns": self.analyze_bidirectional_patterns(),
            "type_comparison": self.compare_connection_types(),
            
            # Connection type chains
            "signal_chains": self.find_signal_processing_chains(),
            "matrix_chains": self.find_matrix_processing_chains(),
        }
        
        return report
    
    def visualize_connection_types(self, output_path: str = None) -> None:
        """Visualize the distribution of connection types"""
        # Get counts for each connection type
        type_counts = Counter()
        for edge in self.edges.values():
            for conn_type in edge.connection_types:
                type_counts[conn_type] += 1
        
        # Create a bar chart
        plt.figure(figsize=(10, 6))
        
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        
        plt.bar(types, counts, color='skyblue')
        plt.xlabel('Connection Type')
        plt.ylabel('Count')
        plt.title('Distribution of Connection Types')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()