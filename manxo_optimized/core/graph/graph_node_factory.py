from typing import Dict, List, Any, Optional, Union
import re
from graph_node import (
    GraphNode, MessageNode, ObjectNode, CommentNode, PatcherNode,
    AudioObjectNode, VisualObjectNode, UIObjectNode
)

class GraphNodeFactory:
    """Factory class for creating appropriate GraphNode subclasses based on object type"""
    
    # Dictionary of known audio objects
    AUDIO_OBJECTS = {
        "cycle~", "osc~", "noise~", "sig~", "line~", "phasor~", "saw~", "tri~", 
        "rect~", "groove~", "play~", "dac~", "adc~", "gain~", "biquad~", "zerox~",
        "hip~", "lop~", "bp~", "vcf~", "svf~", "teeth~", "sfplay~", "tapout~", 
        "tapin~", "delay~", "comb~", "allpass~", "freqshift~", "pfft~", "fft~", 
        "ifft~", "filterdesign", "average~", "avg~", "slide~", "snapshot~"
    }
    
    # Dictionary of known visual objects (Jitter)
    VISUAL_OBJECTS = {
        "jit.matrix", "jit.window", "jit.pwindow", "jit.cellblock", "jit.fetch", 
        "jit.gen", "jit.gl.handle", "jit.gl.model", "jit.gl.render", "jit.gl.sketch",
        "jit.gl.texture", "jit.gl.videoplane", "jit.movie", "jit.qt.movie", 
        "jit.noise", "jit.op", "jit.pack", "jit.unpack", "jit.phys", "jit.qball",
        "jit.rota", "jit.slide", "jit.xfade", "jit.clip", "jit.catch~"
    }
    
    # Dictionary of known UI objects
    UI_OBJECTS = {
        "slider", "dial", "kslider", "number", "flonum", "toggle", "button", 
        "matrixctrl", "pictslider", "live.slider", "live.dial", "live.numbox", 
        "live.toggle", "live.text", "live.menu", "live.tab", "live.grid",
        "bpatcher", "panel", "radiogroup", "tab", "umenu", "attrui", "spectroscope~"
    }
    
    @classmethod
    def create_node(cls, node_id: str, node_type: str, args: List[str] = None, category: str = None) -> GraphNode:
        """
        Create a node of the appropriate type based on node_type.
        
        Args:
            node_id: Unique identifier for the node
            node_type: The type of the node (object name)
            args: Optional list of arguments for the node
            category: Optional category hint
            
        Returns:
            A GraphNode instance of the appropriate subclass
        """
        # Check for special box types first
        if node_type == "message":
            message_text = " ".join(args) if args else ""
            return MessageNode(node_id, message_text)
            
        elif node_type == "comment":
            comment_text = " ".join(args) if args else ""
            return CommentNode(node_id, comment_text)
            
        elif node_type in ["p", "patcher", "subpatch"]:
            patcher_name = args[0] if args else "subpatch"
            return PatcherNode(node_id, patcher_name)
        
        # Check if it's an audio object (ends with ~)
        elif node_type.endswith("~") or node_type in cls.AUDIO_OBJECTS:
            return AudioObjectNode(node_id, node_type, args)
            
        # Check if it's a visual (Jitter) object
        elif node_type.startswith("jit.") or node_type in cls.VISUAL_OBJECTS:
            return VisualObjectNode(node_id, node_type, args)
            
        # Check if it's a UI object
        elif node_type.startswith("live.") or node_type in cls.UI_OBJECTS:
            return UIObjectNode(node_id, node_type, args)
            
        # Default to regular object
        else:
            return ObjectNode(node_id, node_type, args)
    
    @classmethod
    def configure_object_ports(cls, node: ObjectNode) -> GraphNode:
        """
        Configure the ports (inlets/outlets) for an object based on its type
        
        Args:
            node: The ObjectNode to configure
            
        Returns:
            The configured node
        """
        object_name = node.object_name
        
        # Configure common objects with known inlet/outlet configurations
        if object_name == "dac~":
            # Default stereo dac~
            node.configure_ports(2, 0, ["signal", "signal"], [])
        
        elif object_name == "adc~":
            # Default stereo adc~
            node.configure_ports(0, 2, [], ["signal", "signal"])
        
        elif object_name == "gain~":
            # Default mono gain~ with control inlet
            args = node.arguments
            channels = 1
            if args and args[0].isdigit():
                channels = int(args[0])
            inlet_types = ["signal"] * channels + ["control"]
            outlet_types = ["signal"] * channels
            node.configure_ports(len(inlet_types), len(outlet_types), inlet_types, outlet_types)
        
        elif object_name == "cycle~" or object_name == "osc~":
            # Oscillator with frequency inlet and phase inlet
            node.configure_ports(2, 1, ["signal", "signal"], ["signal"])
        
        elif object_name == "selector~":
            # Selector with control inlet + signal inlets, and one signal outlet
            args = node.arguments
            inputs = 2
            if args and args[0].isdigit():
                inputs = int(args[0])
            inlet_types = ["control"] + ["signal"] * inputs
            node.configure_ports(len(inlet_types), 1, inlet_types, ["signal"])
        
        elif object_name == "jit.matrix":
            # Jitter matrix with control and matrix inlets and outlets
            node.configure_ports(2, 2, ["control", "matrix"], ["control", "matrix"])
        
        elif object_name == "delay~":
            # Delay with signal input and feedback 
            node.configure_ports(2, 1, ["signal", "signal"], ["signal"])
        
        # Add more custom port configurations as needed for other objects
            
        return node
    
    @classmethod
    def create_from_maxpat_dict(cls, node_dict: Dict[str, Any]) -> GraphNode:
        """
        Create a GraphNode from a Max/MSP JSON dictionary object
        
        Args:
            node_dict: Dictionary representing a Max/MSP object
            
        Returns:
            A GraphNode instance
        """
        # Extract basic information
        node_id = node_dict.get("id", str(id(node_dict)))
        box_type = node_dict.get("box_type", "")
        
        if box_type == "message":
            text = node_dict.get("text", "")
            node = MessageNode(node_id, text)
            
        elif box_type == "comment":
            text = node_dict.get("text", "")
            node = CommentNode(node_id, text)
            
        elif box_type == "patcher":
            name = node_dict.get("patcher_name", "subpatch")
            node = PatcherNode(node_id, name)
            if "subpatcher_id" in node_dict:
                node.set_subgraph(node_dict["subpatcher_id"])
                
        else:
            # Object box
            object_name = node_dict.get("object_name", "")
            args = node_dict.get("args", [])
            node = cls.create_node(node_id, object_name, args)
            
            # Configure ports based on inlet and outlet specifications
            if isinstance(node, ObjectNode) and "inlets" in node_dict and "outlets" in node_dict:
                inlets = node_dict["inlets"]
                outlets = node_dict["outlets"]
                
                # Clear existing ports
                node.inlets = []
                node.outlets = []
                
                # Add inlets
                for inlet in inlets:
                    node.add_inlet(
                        inlet.get("type", "control"),
                        inlet.get("description", "")
                    )
                
                # Add outlets
                for outlet in outlets:
                    node.add_outlet(
                        outlet.get("type", "control"),
                        outlet.get("description", "")
                    )
            else:
                # Try to configure ports based on object type
                if isinstance(node, ObjectNode):
                    cls.configure_object_ports(node)
        
        # Set common properties
        if "position" in node_dict:
            node.position = node_dict["position"]
        
        if "properties" in node_dict:
            node.properties = node_dict["properties"]
            
        if "comment" in node_dict:
            node.set_comment(node_dict["comment"])
            
        if "is_disabled" in node_dict:
            node.disable(node_dict["is_disabled"])
            
        return node