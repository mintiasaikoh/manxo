import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from datetime import datetime
from node import Node
from edge import Edge, EdgeAnalyzer, EdgeFactory, ConnectionType, EdgeDirection

class PatchGraph:
    """Graph representation of a patch with support for compound connections"""
    
    def __init__(self, patch_id: str, name: str = ""):
        self.id = patch_id
        self.name = name if name else f"graph_{patch_id}"
        self.nodes = {}  # Dictionary of Node objects
        self.edges = {}  # Dictionary of Edge objects
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "source_file": "",
            "node_count": 0,
            "edge_count": 0,
            "max_hierarchy_level": 0,
            "connection_type_counts": {},
            "compound_connections": 0
        }
    
    def add_node(self, node: Node) -> Node:
        """Add a node to the graph"""
        self.nodes[node.id] = node
        self.metadata["node_count"] = len(self.nodes)
        
        # Update hierarchy metadata if needed
        if node.hierarchy_level > self.metadata["max_hierarchy_level"]:
            self.metadata["max_hierarchy_level"] = node.hierarchy_level
            
        return node
    
    def add_node_from_dict(self, node_dict: Dict[str, Any]) -> Node:
        """Create and add a node from a dictionary"""
        node_id = node_dict.get("id")
        node_type = node_dict.get("type")
        category = node_dict.get("category")
        
        node = Node(node_id, node_type, category)
        
        # Set position if available
        if "position" in node_dict:
            pos = node_dict["position"]
            node.set_position(pos.get("x", 0), pos.get("y", 0))
        
        # Add arguments if available
        if "arguments" in node_dict:
            for arg in node_dict["arguments"]:
                node.add_argument(arg)
        
        # Add inlets if available
        if "inlets" in node_dict:
            for inlet in node_dict["inlets"]:
                node.add_inlet(
                    inlet.get("type", "control"),
                    inlet.get("description", "")
                )
        
        # Add outlets if available
        if "outlets" in node_dict:
            for outlet in node_dict["outlets"]:
                node.add_outlet(
                    outlet.get("type", "control"),
                    outlet.get("description", "")
                )
        
        # Set hierarchy if available
        if "parent_id" in node_dict and node_dict["parent_id"]:
            node.set_parent(node_dict["parent_id"], node_dict.get("hierarchy_level", 1))
        
        # Add properties if available
        if "properties" in node_dict:
            for key, value in node_dict["properties"].items():
                node.add_property(key, value)
        
        # Add to graph
        return self.add_node(node)
    
    def add_edge(self, edge: Edge) -> Edge:
        """Add an edge to the graph and update metadata"""
        self.edges[edge.id] = edge
        self.metadata["edge_count"] = len(self.edges)
        
        # Update connection type statistics
        connection_types = edge.connection_types
        for conn_type in connection_types:
            self.metadata["connection_type_counts"][conn_type] = \
                self.metadata["connection_type_counts"].get(conn_type, 0) + 1
        
        # Track compound connections
        if edge.is_compound():
            self.metadata["compound_connections"] += 1
            
        return edge
    
    def add_edge_from_dict(self, edge_dict: Dict[str, Any]) -> Edge:
        """Create and add an edge from a dictionary"""
        edge_id = edge_dict.get("id")
        source = edge_dict.get("source", {})
        target = edge_dict.get("target", {})
        
        edge = Edge(
            edge_id,
            source.get("node_id", ""),
            source.get("port", 0),
            target.get("node_id", ""),
            target.get("port", 0)
        )
        
        # Set connection types
        if "connection_types" in edge_dict:
            # Handle compound connections
            types_list = edge_dict["connection_types"]
            if types_list:
                edge.set_type(types_list[0])  # Set the first type
                for conn_type in types_list[1:]:
                    edge.add_type(conn_type)  # Add additional types
        elif "type" in edge_dict:
            # Handle simple connection type
            edge.set_type(edge_dict["type"])
        
        # Set visibility if available
        if "is_hidden" in edge_dict:
            edge.hide(edge_dict["is_hidden"])
        
        # Set enabled/disabled if available
        if "is_disabled" in edge_dict:
            edge.disable(edge_dict["is_disabled"])
            
        # Set direction if available
        if "direction" in edge_dict:
            edge.set_direction(edge_dict["direction"])
        
        # Add midpoints if available
        if "midpoints" in edge_dict:
            for point in edge_dict["midpoints"]:
                edge.add_midpoint(point.get("x", 0), point.get("y", 0))
                
        # Add properties if available
        if "properties" in edge_dict:
            for key, value in edge_dict["properties"].items():
                edge.add_property(key, value)
        
        # Add to graph
        return self.add_edge(edge)
    
    def connect_nodes(self, edge_id: str, source_node: str, source_port: int,
                     target_node: str, target_port: int, 
                     connection_types: Union[str, List[str]] = "control") -> Edge:
        """Create a connection between nodes with support for multiple types"""
        if isinstance(connection_types, str):
            connection_types = [connection_types]  # Convert single type to list
            
        # Create edge with the first connection type
        edge = Edge(edge_id, source_node, source_port, target_node, target_port)
        
        # Set connection types
        if connection_types:
            edge.set_type(connection_types[0])  # Set first type
            for conn_type in connection_types[1:]:  # Add any additional types
                edge.add_type(conn_type)
                
        return self.add_edge(edge)
    
    def connect_nodes_bidirectional(self, edge_id: str, node_a: str, port_a: int,
                                  node_b: str, port_b: int,
                                  connection_type: str = "control") -> Edge:
        """Create a bidirectional connection between nodes"""
        edge = EdgeFactory.create_bidirectional_edge(
            edge_id, node_a, port_a, node_b, port_b, connection_type
        )
        return self.add_edge(edge)
    
    def add_compound_connection(self, edge_id: str, source_node: str, source_port: int,
                               target_node: str, target_port: int,
                               connection_types: List[str]) -> Edge:
        """Create a compound connection with multiple types"""
        edge = EdgeFactory.create_compound_edge(
            edge_id, source_node, source_port, target_node, target_port, connection_types
        )
        return self.add_edge(edge)
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID"""
        return self.nodes.get(node_id)
    
    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Get an edge by ID"""
        return self.edges.get(edge_id)
    
    def find_connected_nodes(self, node_id: str, direction: str = "both",
                           connection_type: Optional[str] = None) -> List[str]:
        """Find nodes connected to the given node with optional filtering by connection type"""
        connected = []
        
        if direction in ["out", "both"]:
            # Find outgoing connections
            for edge in self.edges.values():
                if edge.source["node_id"] == node_id:
                    # Filter by connection type if specified
                    if connection_type is None or edge.has_type(connection_type):
                        if edge.target["node_id"] not in connected:
                            connected.append(edge.target["node_id"])
        
        if direction in ["in", "both"]:
            # Find incoming connections
            for edge in self.edges.values():
                if edge.target["node_id"] == node_id:
                    # Filter by connection type if specified
                    if connection_type is None or edge.has_type(connection_type):
                        if edge.source["node_id"] not in connected:
                            connected.append(edge.source["node_id"])
        
        return connected
    
    def find_connected_edges(self, node_id: str, direction: str = "both",
                           connection_type: Optional[str] = None) -> List[str]:
        """Find edges connected to the given node with optional filtering by connection type"""
        connected = []
        
        if direction in ["out", "both"]:
            # Find outgoing connections
            for edge_id, edge in self.edges.items():
                if edge.source["node_id"] == node_id:
                    # Filter by connection type if specified
                    if connection_type is None or edge.has_type(connection_type):
                        if edge_id not in connected:
                            connected.append(edge_id)
        
        if direction in ["in", "both"]:
            # Find incoming connections
            for edge_id, edge in self.edges.items():
                if edge.target["node_id"] == node_id:
                    # Filter by connection type if specified
                    if connection_type is None or edge.has_type(connection_type):
                        if edge_id not in connected:
                            connected.append(edge_id)
        
        return connected
    
    def find_compound_edges(self) -> List[str]:
        """Find all compound edges in the graph"""
        return [edge_id for edge_id, edge in self.edges.items() if edge.is_compound()]
    
    def find_bidirectional_edges(self) -> List[str]:
        """Find all bidirectional edges in the graph"""
        return [
            edge_id for edge_id, edge in self.edges.items() 
            if edge.direction == EdgeDirection.BIDIRECTIONAL.value
        ]
    
    def analyze_connection_types(self) -> Dict[str, Any]:
        """Analyze the connection types in the graph"""
        # Use EdgeAnalyzer to compute statistics
        return EdgeAnalyzer.compute_type_statistics(list(self.edges.values()))
    
    def find_connection_patterns(self) -> Dict[str, Any]:
        """Find complex connection patterns in the graph"""
        # Use EdgeAnalyzer to find patterns
        return EdgeAnalyzer.find_complex_patterns(list(self.edges.values()))
    
    def create_connection_matrix(self) -> pd.DataFrame:
        """Create a connection matrix showing how nodes are connected"""
        # Get all node IDs
        node_ids = list(self.nodes.keys())
        
        # Create empty matrix
        matrix = pd.DataFrame(0, index=node_ids, columns=node_ids)
        
        # Fill the matrix
        for edge in self.edges.values():
            source = edge.source["node_id"]
            target = edge.target["node_id"]
            
            if source in node_ids and target in node_ids:
                # Increment connection count
                matrix.at[source, target] += 1
                
                # Handle bidirectional edges
                if edge.direction == EdgeDirection.BIDIRECTIONAL.value:
                    matrix.at[target, source] += 1
        
        return matrix
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "edges": {edge_id: edge.to_dict() for edge_id, edge in self.edges.items()},
            "metadata": self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save_to_file(self, file_path: str) -> None:
        """Save the graph as JSON to a file"""
        with open(file_path, 'w') as f:
            f.write(self.to_json())
        
        # Update metadata
        self.metadata["source_file"] = file_path