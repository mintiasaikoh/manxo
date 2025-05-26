from typing import Dict, List, Any, Optional, Union
from node import Node
from edge import Edge

class GraphNode(Node):
    """Base class for specialized graph nodes representing different box types"""
    
    def __init__(self, node_id: str, node_type: str, category: str = None):
        super().__init__(node_id, node_type, category)
        self.box_type = "generic"  # Base box type
        self.display_name = node_type  # Default display name
        self.is_disabled = False
        self.is_collapsed = False
        self.comment = ""
        self.style = {}  # Visual styling properties
        
    def disable(self, disabled: bool = True):
        """Enable or disable the node"""
        self.is_disabled = disabled
        return self
        
    def collapse(self, collapsed: bool = True):
        """Collapse or expand the node for visual representation"""
        self.is_collapsed = collapsed
        return self
        
    def set_comment(self, comment: str):
        """Set a comment for this node"""
        self.comment = comment
        return self
        
    def set_style(self, style_dict: Dict[str, Any]):
        """Set visual styling properties"""
        self.style.update(style_dict)
        return self
        
    def set_display_name(self, name: str):
        """Set a custom display name for this node"""
        self.display_name = name
        return self
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation, extending the base Node class"""
        node_dict = super().to_dict()
        # Add GraphNode specific fields
        node_dict.update({
            "box_type": self.box_type,
            "display_name": self.display_name,
            "is_disabled": self.is_disabled,
            "is_collapsed": self.is_collapsed,
            "comment": self.comment,
            "style": self.style
        })
        return node_dict


class MessageNode(GraphNode):
    """Represents a message box"""
    
    def __init__(self, node_id: str, message_text: str = ""):
        super().__init__(node_id, "message", "basic")
        self.box_type = "message"
        self.message_text = message_text
        # Message boxes typically have one inlet and one outlet
        self.add_inlet("control", "Control inlet")
        self.add_outlet("control", "Control outlet")
        
    def set_message(self, message_text: str):
        """Set the message text"""
        self.message_text = message_text
        return self
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        node_dict = super().to_dict()
        node_dict["message_text"] = self.message_text
        return node_dict


class ObjectNode(GraphNode):
    """Represents an object box (function/utility)"""
    
    def __init__(self, node_id: str, object_name: str, args: List[str] = None):
        super().__init__(node_id, object_name, "object")
        self.box_type = "object"
        self.object_name = object_name
        self.arguments = args or []
        # Ports will be configured based on the specific object type
        
    def set_object(self, object_name: str, args: List[str] = None):
        """Set or change the object name and arguments"""
        self.type = object_name
        self.object_name = object_name
        if args is not None:
            self.arguments = args
        return self
        
    def configure_ports(self, num_inlets: int = 1, num_outlets: int = 1, 
                        inlet_types: List[str] = None, outlet_types: List[str] = None):
        """Configure the ports based on the object type"""
        # Clear existing ports
        self.inlets = []
        self.outlets = []
        
        # Set default types
        inlet_types = inlet_types or ["control"] * num_inlets
        outlet_types = outlet_types or ["control"] * num_outlets
        
        # Add inlets
        for i in range(num_inlets):
            port_type = inlet_types[i] if i < len(inlet_types) else "control"
            self.add_inlet(port_type, f"Inlet {i}")
            
        # Add outlets
        for i in range(num_outlets):
            port_type = outlet_types[i] if i < len(outlet_types) else "control"
            self.add_outlet(port_type, f"Outlet {i}")
            
        return self
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        node_dict = super().to_dict()
        node_dict["object_name"] = self.object_name
        return node_dict


class CommentNode(GraphNode):
    """Represents a comment box"""
    
    def __init__(self, node_id: str, comment_text: str = ""):
        super().__init__(node_id, "comment", "utility")
        self.box_type = "comment"
        self.comment_text = comment_text
        # Comments don't have inlets or outlets
        
    def set_comment_text(self, comment_text: str):
        """Set the comment text"""
        self.comment_text = comment_text
        return self
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        node_dict = super().to_dict()
        node_dict["comment_text"] = self.comment_text
        return node_dict


class PatcherNode(GraphNode):
    """Represents a subpatcher or abstraction"""
    
    def __init__(self, node_id: str, patcher_name: str = "subpatch"):
        super().__init__(node_id, "patcher", "container")
        self.box_type = "patcher"
        self.patcher_name = patcher_name
        self.subgraph_id = f"subpatch_{node_id}"  # Default ID for the subgraph
        
    def set_subgraph(self, subgraph_id: str):
        """Set the ID of the associated subgraph"""
        self.subgraph_id = subgraph_id
        return self
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        node_dict = super().to_dict()
        node_dict.update({
            "patcher_name": self.patcher_name,
            "subgraph_id": self.subgraph_id
        })
        return node_dict


class AudioObjectNode(ObjectNode):
    """Specialized node for audio processing objects"""
    
    def __init__(self, node_id: str, object_name: str, args: List[str] = None):
        super().__init__(node_id, object_name, args)
        self.category = "audio"
        # Default to signal connections for audio objects
        self.configure_ports(1, 1, ["signal"], ["signal"])
        
    def set_channel_count(self, input_channels: int = 1, output_channels: int = 1):
        """Configure the node for multiple audio channels"""
        inlet_types = ["signal"] * input_channels
        outlet_types = ["signal"] * output_channels
        self.configure_ports(input_channels, output_channels, inlet_types, outlet_types)
        return self


class VisualObjectNode(ObjectNode):
    """Specialized node for visual processing (Jitter) objects"""
    
    def __init__(self, node_id: str, object_name: str, args: List[str] = None):
        super().__init__(node_id, object_name, args)
        self.category = "visual"
        # Default matrix connection for main I/O, with control for parameters
        self.add_inlet("control", "Control inlet")
        self.add_inlet("matrix", "Matrix inlet")
        self.add_outlet("matrix", "Matrix outlet")
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        node_dict = super().to_dict()
        node_dict["category"] = "visual"
        return node_dict


class UIObjectNode(ObjectNode):
    """Specialized node for UI elements (sliders, buttons, etc.)"""
    
    def __init__(self, node_id: str, object_name: str, args: List[str] = None):
        super().__init__(node_id, object_name, args)
        self.category = "ui"
        self.value = None  # Current value of the UI element
        self.range = {"min": 0, "max": 1}  # Default range
        self.configure_ports(1, 1)  # Default single input/output
        
    def set_value(self, value: Any):
        """Set current value of the UI element"""
        self.value = value
        return self
        
    def set_range(self, min_val: float, max_val: float):
        """Set the range for the UI element"""
        self.range = {"min": min_val, "max": max_val}
        return self
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        node_dict = super().to_dict()
        node_dict.update({
            "value": self.value,
            "range": self.range
        })
        return node_dict