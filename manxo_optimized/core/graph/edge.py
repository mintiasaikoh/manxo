from typing import Dict, List, Any, Optional, Union, Set
from enum import Enum, auto
import numpy as np
import pandas as pd
from datetime import datetime

class ConnectionType(Enum):
    """Enumeration of connection types in Max/MSP patches"""
    CONTROL = "control"    # Basic event flow
    SIGNAL = "signal"      # Audio processing connections (MSP)
    MATRIX = "matrix"      # Matrix/visual processing connections (Jitter)
    MIDIMAN = "midiman"    # Special connections created by Max scripts
    COMPOUND = "compound"  # Compound connection type (multiple types)

class EdgeDirection(Enum):
    """Enumeration of edge directions"""
    UNIDIRECTIONAL = "unidirectional"
    BIDIRECTIONAL = "bidirectional"

class Edge:
    """Graph edge representation with support for compound connection types"""
    
    def __init__(self, edge_id: str, source_node: str, source_port: int, 
                 target_node: str, target_port: int):
        self.id = edge_id
        self.source = {
            "node_id": source_node,
            "port": source_port
        }
        self.target = {
            "node_id": target_node,
            "port": target_port
        }
        self.created_at = datetime.now().isoformat()
        self._connection_types = {ConnectionType.CONTROL.value}  # Default to control type
        self.is_hidden = False
        self.is_disabled = False
        self.midpoints = []
        self.direction = EdgeDirection.UNIDIRECTIONAL.value
        self.properties = {}
        
    @property
    def type(self) -> str:
        """Return the primary connection type, or 'compound' if multiple types exist"""
        if len(self._connection_types) == 1:
            return next(iter(self._connection_types))
        return ConnectionType.COMPOUND.value
    
    @property
    def connection_types(self) -> Set[str]:
        """Return the set of all connection types for this edge"""
        return self._connection_types
        
    def set_type(self, edge_type: str):
        """Set the type of connection (control, signal, matrix, etc.)"""
        # Clear existing types and set the new one
        self._connection_types = {edge_type}
        return self
    
    def add_type(self, edge_type: str):
        """Add another connection type to this edge (making it a compound edge)"""
        self._connection_types.add(edge_type)
        return self
    
    def remove_type(self, edge_type: str):
        """Remove a connection type from this edge"""
        if edge_type in self._connection_types and len(self._connection_types) > 1:
            self._connection_types.remove(edge_type)
        return self
    
    def has_type(self, edge_type: str) -> bool:
        """Check if this edge has a specific connection type"""
        return edge_type in self._connection_types
    
    def is_compound(self) -> bool:
        """Check if this edge is a compound connection (multiple types)"""
        return len(self._connection_types) > 1
    
    def set_direction(self, direction: str):
        """Set the direction of the edge (unidirectional/bidirectional)"""
        if direction in [EdgeDirection.UNIDIRECTIONAL.value, EdgeDirection.BIDIRECTIONAL.value]:
            self.direction = direction
        return self
    
    def hide(self, hidden: bool = True):
        """Set edge visibility"""
        self.is_hidden = hidden
        return self
    
    def disable(self, disabled: bool = True):
        """Enable or disable the edge"""
        self.is_disabled = disabled
        return self
    
    def add_midpoint(self, x: float, y: float):
        """Add a midpoint for edge routing"""
        self.midpoints.append({"x": x, "y": y})
        return self
    
    def add_property(self, key: str, value: Any):
        """Add a custom property to the edge"""
        self.properties[key] = value
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "connection_types": list(self._connection_types),
            "is_hidden": self.is_hidden,
            "is_disabled": self.is_disabled,
            "midpoints": self.midpoints,
            "direction": self.direction,
            "properties": self.properties,
            "created_at": self.created_at
        }

class EdgeAnalyzer:
    """Utility class for analyzing collections of edges"""
    
    @staticmethod
    def analyze_edge_types(edges: List[Edge]) -> pd.DataFrame:
        """Analyze the distribution of edge types in a collection"""
        # Prepare data for analysis
        edge_data = []
        for edge in edges:
            for conn_type in edge.connection_types:
                edge_data.append({
                    "edge_id": edge.id,
                    "source_node": edge.source["node_id"],
                    "target_node": edge.target["node_id"],
                    "connection_type": conn_type,
                    "is_compound": edge.is_compound(),
                    "direction": edge.direction
                })
        
        # Create DataFrame for analysis
        df = pd.DataFrame(edge_data)
        
        # Return the DataFrame for further analysis
        return df
    
    @staticmethod
    def compute_type_statistics(edges: List[Edge]) -> Dict[str, Any]:
        """Compute statistics about edge types"""
        df = EdgeAnalyzer.analyze_edge_types(edges)
        
        # Basic statistics
        stats = {
            "total_edges": len(edges),
            "type_counts": df['connection_type'].value_counts().to_dict(),
            "compound_edges": df['is_compound'].sum(),
            "bidirectional_edges": (df['direction'] == EdgeDirection.BIDIRECTIONAL.value).sum(),
            "avg_types_per_edge": len(df) / len(edges) if edges else 0
        }
        
        return stats
    
    @staticmethod
    def find_complex_patterns(edges: List[Edge]) -> Dict[str, Any]:
        """Identify complex connection patterns in the edge collection"""
        df = EdgeAnalyzer.analyze_edge_types(edges)
        
        # Find nodes with multiple incoming connection types
        source_target_pairs = df.groupby(['source_node', 'target_node'])['connection_type'].nunique()
        complex_routes = source_target_pairs[source_target_pairs > 1].to_dict()
        
        # Find nodes that are part of both control and signal flows
        node_types = {}
        for node_id in set(df['source_node'].unique()) | set(df['target_node'].unique()):
            node_types[node_id] = set(
                df[(df['source_node'] == node_id) | (df['target_node'] == node_id)]['connection_type']
            )
        
        hybrid_nodes = {
            node_id: types for node_id, types in node_types.items() 
            if ConnectionType.CONTROL.value in types and ConnectionType.SIGNAL.value in types
        }
        
        return {
            "complex_routes": complex_routes,
            "hybrid_nodes": hybrid_nodes
        }

class EdgeFactory:
    """Factory class for creating various types of edges"""
    
    @staticmethod
    def create_control_edge(edge_id: str, source_node: str, source_port: int,
                           target_node: str, target_port: int) -> Edge:
        """Create a standard control connection"""
        edge = Edge(edge_id, source_node, source_port, target_node, target_port)
        edge.set_type(ConnectionType.CONTROL.value)
        return edge
    
    @staticmethod
    def create_signal_edge(edge_id: str, source_node: str, source_port: int,
                          target_node: str, target_port: int) -> Edge:
        """Create an audio signal connection (MSP)"""
        edge = Edge(edge_id, source_node, source_port, target_node, target_port)
        edge.set_type(ConnectionType.SIGNAL.value)
        return edge
    
    @staticmethod
    def create_matrix_edge(edge_id: str, source_node: str, source_port: int,
                          target_node: str, target_port: int) -> Edge:
        """Create a matrix connection (Jitter)"""
        edge = Edge(edge_id, source_node, source_port, target_node, target_port)
        edge.set_type(ConnectionType.MATRIX.value)
        return edge
    
    @staticmethod
    def create_midiman_edge(edge_id: str, source_node: str, source_port: int,
                           target_node: str, target_port: int) -> Edge:
        """Create a midiman connection (Max script)"""
        edge = Edge(edge_id, source_node, source_port, target_node, target_port)
        edge.set_type(ConnectionType.MIDIMAN.value)
        return edge
    
    @staticmethod
    def create_compound_edge(edge_id: str, source_node: str, source_port: int,
                            target_node: str, target_port: int, 
                            connection_types: List[str]) -> Edge:
        """Create a compound connection with multiple types"""
        edge = Edge(edge_id, source_node, source_port, target_node, target_port)
        
        # Set the first type and add the rest
        if connection_types:
            edge.set_type(connection_types[0])
            for conn_type in connection_types[1:]:
                edge.add_type(conn_type)
        
        return edge
    
    @staticmethod
    def create_bidirectional_edge(edge_id: str, node_a: str, port_a: int,
                                 node_b: str, port_b: int, 
                                 connection_type: str = ConnectionType.CONTROL.value) -> Edge:
        """Create a bidirectional connection between nodes"""
        edge = Edge(edge_id, node_a, port_a, node_b, port_b)
        edge.set_type(connection_type)
        edge.set_direction(EdgeDirection.BIDIRECTIONAL.value)
        return edge