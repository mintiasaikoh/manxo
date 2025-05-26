from typing import Dict, List, Any, Optional
from datetime import datetime

class Node:
    """Graph node representation"""
    
    def __init__(self, node_id: str, node_type: str, category: str = None):
        self.id = node_id
        self.type = node_type
        self.category = category
        self.arguments = []
        self.inlets = []
        self.outlets = []
        self.hierarchy_level = 0
        self.parent_id = None
        self.position = {"x": 0, "y": 0}
        self.properties = {}
        self.hierarchy_path = node_id
    
    def add_argument(self, value: Any):
        """Add an argument to the node"""
        self.arguments.append(value)
        return self
    
    def add_inlet(self, inlet_type: str, description: str = ""):
        """Add an inlet to the node"""
        self.inlets.append({
            "type": inlet_type,
            "description": description
        })
        return self
    
    def add_outlet(self, outlet_type: str, description: str = ""):
        """Add an outlet to the node"""
        self.outlets.append({
            "type": outlet_type,
            "description": description
        })
        return self
    
    def set_position(self, x: float, y: float):
        """Set the 2D position of the node"""
        self.position = {"x": x, "y": y}
        return self
    
    def set_parent(self, parent_id: str, level: int = 1):
        """Set the parent node and hierarchy level"""
        self.parent_id = parent_id
        self.hierarchy_level = level
        self.hierarchy_path = f"{parent_id}/{self.id}"
        return self
    
    def add_property(self, key: str, value: Any):
        """Add a custom property to the node"""
        self.properties[key] = value
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "type": self.type,
            "category": self.category,
            "arguments": self.arguments,
            "inlets": self.inlets,
            "outlets": self.outlets,
            "hierarchy_level": self.hierarchy_level,
            "parent_id": self.parent_id,
            "position": self.position,
            "properties": self.properties,
            "hierarchy_path": self.hierarchy_path
        }