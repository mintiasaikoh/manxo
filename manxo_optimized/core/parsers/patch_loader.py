import json
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from patch_graph import PatchGraph
from graph_node import GraphNode
from graph_node_factory import GraphNodeFactory
from edge import Edge

class PatchLoader:
    """Utility class for loading and saving patches in different formats"""
    
    @staticmethod
    def load_maxpat(file_path: str) -> Tuple[PatchGraph, Dict[str, PatchGraph]]:
        """
        Load a Max/MSP .maxpat file and convert it to a PatchGraph
        
        Args:
            file_path: Path to the .maxpat file
            
        Returns:
            A tuple containing:
            - The main PatchGraph
            - A dictionary of subpatches indexed by ID
        """
        with open(file_path, 'r') as f:
            max_json = json.load(f)
            
        # Extract metadata
        patch_id = os.path.basename(file_path).split('.')[0]
        patch_name = max_json.get("patcher", {}).get("name", patch_id)
        
        # Create main graph
        main_graph = PatchGraph(patch_id, patch_name)
        
        # Dictionary to store subpatches
        subpatches = {}
        
        # Process the patcher content
        patcher = max_json.get("patcher", {})
        PatchLoader._process_maxpat_patcher(patcher, main_graph, subpatches)
        
        return main_graph, subpatches
    
    @staticmethod
    def _process_maxpat_patcher(patcher: Dict[str, Any], graph: PatchGraph, 
                               subpatches: Dict[str, PatchGraph], parent_id: str = None) -> None:
        """
        Process a Max/MSP patcher dictionary and populate the graph
        
        Args:
            patcher: Dictionary containing the patcher data
            graph: PatchGraph to populate
            subpatches: Dictionary to store any subpatches
            parent_id: ID of the parent patcher, if any
        """
        # Process boxes (nodes)
        boxes = patcher.get("boxes", [])
        for box in boxes:
            # Create a node from the box
            node = PatchLoader._create_node_from_maxpat_box(box, graph.id)
            
            # Set hierarchy info if needed
            if parent_id:
                node.parent_id = parent_id
                node.hierarchy_level = 1  # Adjust as needed for deeper nesting
            
            # Add node to graph
            graph.add_node(node)
            
            # Handle subpatchers recursively
            if "patcher" in box:
                subpatch_data = box["patcher"]
                subpatch_id = f"subpatch_{node.id}"
                
                # Create subpatch
                subpatch_name = subpatch_data.get("name", f"subpatch_{node.id}")
                subpatch = PatchGraph(subpatch_id, subpatch_name)
                
                # Process the subpatch
                PatchLoader._process_maxpat_patcher(subpatch_data, subpatch, subpatches, node.id)
                
                # Store the subpatch
                subpatches[subpatch_id] = subpatch
                
                # Link the patcher node to its subgraph
                if isinstance(node, GraphNode) and hasattr(node, 'subgraph_id'):
                    node.subgraph_id = subpatch_id
        
        # Process lines (edges)
        lines = patcher.get("lines", [])
        for i, line in enumerate(lines):
            edge = PatchLoader._create_edge_from_maxpat_line(line, i, graph.id)
            if edge:
                graph.add_edge(edge)
    
    @staticmethod
    def _create_node_from_maxpat_box(box: Dict[str, Any], graph_id: str) -> GraphNode:
        """
        Create a GraphNode from a Max/MSP box
        
        Args:
            box: Dictionary containing the box data
            graph_id: ID of the graph this node belongs to
            
        Returns:
            A GraphNode instance
        """
        # Extract basic box info
        box_id = box.get("box_id", str(id(box)))
        box_type = box.get("box_type", "")
        
        # Create dictionary for GraphNodeFactory
        node_dict = {
            "id": box_id,
            "box_type": box_type
        }
        
        # Add position if available
        if "patching_rect" in box:
            rect = box["patching_rect"]
            node_dict["position"] = {"x": rect[0], "y": rect[1]}
        
        # Handle different box types
        if box_type == "message":
            node_dict["text"] = box.get("text", "")
            
        elif box_type == "comment":
            node_dict["text"] = box.get("text", "")
            
        elif box_type == "patcher":
            node_dict["patcher_name"] = box.get("patcher", {}).get("name", "subpatch")
            node_dict["subpatcher_id"] = f"subpatch_{box_id}"
            
        else:
            # Default to object box
            node_dict["object_name"] = box.get("text", "").split()[0] if box.get("text") else ""
            if box.get("text") and len(box.get("text").split()) > 1:
                node_dict["args"] = box.get("text").split()[1:]
            
        # Process inlets and outlets
        if "numoutlets" in box:
            num_outlets = box.get("numoutlets", 0)
            node_dict["outlets"] = []
            for i in range(num_outlets):
                outlet_type = "control"
                # Check for signal outlets
                if "outlettype" in box and i < len(box["outlettype"]):
                    if box["outlettype"][i] == "signal":
                        outlet_type = "signal"
                    elif box["outlettype"][i] == "jit_matrix":
                        outlet_type = "matrix"
                
                node_dict["outlets"].append({
                    "type": outlet_type,
                    "description": f"Outlet {i}"
                })
                
        if "numinlets" in box:
            num_inlets = box.get("numinlets", 0)
            node_dict["inlets"] = []
            for i in range(num_inlets):
                inlet_type = "control"
                # Check for signal inlets
                if "inlettype" in box and i < len(box["inlettype"]):
                    if box["inlettype"][i] == "signal":
                        inlet_type = "signal"
                    elif box["inlettype"][i] == "jit_matrix":
                        inlet_type = "matrix"
                
                node_dict["inlets"].append({
                    "type": inlet_type,
                    "description": f"Inlet {i}"
                })
        
        # Create the node using the factory
        return GraphNodeFactory.create_from_maxpat_dict(node_dict)
    
    @staticmethod
    def _create_edge_from_maxpat_line(line: Dict[str, Any], index: int, graph_id: str) -> Optional[Edge]:
        """
        Create an Edge from a Max/MSP line
        
        Args:
            line: Dictionary containing the line data
            index: Index of the line for generating an ID
            graph_id: ID of the graph this edge belongs to
            
        Returns:
            An Edge instance or None if invalid
        """
        # Check if we have all necessary information
        if not all(k in line for k in ["patchline", "source", "destination"]):
            return None
        
        # Extract connection info
        source = line["source"]
        dest = line["destination"]
        
        if not all(k in source for k in ["id", "index"]) or not all(k in dest for k in ["id", "index"]):
            return None
        
        # Create edge ID
        edge_id = f"e{index}_{graph_id}"
        
        # Determine connection type
        edge_type = "control"
        if line.get("patchline", {}).get("type", "") == "signal":
            edge_type = "signal"
        elif line.get("patchline", {}).get("type", "") == "jit_matrix":
            edge_type = "matrix"
        
        # Create the edge
        edge = Edge(edge_id, source["id"], source["index"], dest["id"], dest["index"])
        edge.set_type(edge_type)
        
        # Set additional properties if available
        if "disabled" in line.get("patchline", {}):
            edge.disable(line["patchline"]["disabled"])
            
        if "hidden" in line.get("patchline", {}):
            edge.hide(line["patchline"]["hidden"])
        
        return edge
    
    @staticmethod
    def save_to_maxpat(graph: PatchGraph, subpatches: Dict[str, PatchGraph], file_path: str) -> None:
        """
        Convert a PatchGraph to Max/MSP format and save as .maxpat
        
        Args:
            graph: The main PatchGraph
            subpatches: Dictionary of subpatches indexed by ID
            file_path: Path to save the .maxpat file
        """
        # Create base Max/MSP structure
        max_json = {
            "patcher": {
                "fileversion": 1,
                "appversion": {
                    "major": 8,
                    "minor": 5,
                    "revision": 5,
                    "architecture": "x64",
                    "modernui": 1
                },
                "classnamespace": "box",
                "rect": [0, 0, 800, 600],
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
                "boxes": [],
                "lines": [],
                "dependency_cache": [],
                "autosave": 0,
                "name": graph.name
            }
        }
        
        # Convert nodes to boxes
        for node_id, node in graph.nodes.items():
            box = PatchLoader._convert_node_to_maxpat_box(node, subpatches)
            max_json["patcher"]["boxes"].append(box)
        
        # Convert edges to lines
        for edge_id, edge in graph.edges.items():
            line = PatchLoader._convert_edge_to_maxpat_line(edge)
            max_json["patcher"]["lines"].append(line)
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(max_json, f, indent=2)
    
    @staticmethod
    def _convert_node_to_maxpat_box(node: GraphNode, subpatches: Dict[str, PatchGraph]) -> Dict[str, Any]:
        """
        Convert a GraphNode to a Max/MSP box
        
        Args:
            node: The GraphNode to convert
            subpatches: Dictionary of subpatches
            
        Returns:
            Dictionary representing a Max/MSP box
        """
        # Base box structure
        box = {
            "box_id": node.id,
            "patching_rect": [node.position["x"], node.position["y"], 100, 22],  # Default size
            "numinlets": len(node.inlets),
            "numoutlets": len(node.outlets),
            "outlettype": [outlet.get("type", "control") for outlet in node.outlets],
            "inlettype": [inlet.get("type", "control") for inlet in node.inlets]
        }
        
        # Convert based on box type
        if hasattr(node, 'box_type'):
            if node.box_type == "message":
                box["box_type"] = "message"
                if hasattr(node, 'message_text'):
                    box["text"] = node.message_text
                
            elif node.box_type == "comment":
                box["box_type"] = "comment"
                if hasattr(node, 'comment_text'):
                    box["text"] = node.comment_text
                # Adjust size for comments
                box["patching_rect"][2] = 200  # Width
                box["patching_rect"][3] = 50   # Height
                
            elif node.box_type == "patcher":
                box["box_type"] = "patcher"
                if hasattr(node, 'patcher_name'):
                    box["text"] = node.patcher_name
                
                # Include subpatcher if available
                if hasattr(node, 'subgraph_id') and node.subgraph_id in subpatches:
                    subpatch = subpatches[node.subgraph_id]
                    subpatch_dict = PatchLoader._convert_graph_to_maxpat_patcher(subpatch, subpatches)
                    box["patcher"] = subpatch_dict
                
            else:
                # Standard object box
                box["box_type"] = "object"
                
                # Construct object text
                text = node.type
                if hasattr(node, 'arguments') and node.arguments:
                    args_text = " ".join(str(arg) for arg in node.arguments)
                    text = f"{text} {args_text}"
                box["text"] = text
        
        # Handle disabled state
        if hasattr(node, 'is_disabled'):
            box["disabled"] = 1 if node.is_disabled else 0
        
        # Add additional properties
        if hasattr(node, 'properties') and node.properties:
            for key, value in node.properties.items():
                box[key] = value
        
        return box
    
    @staticmethod
    def _convert_edge_to_maxpat_line(edge: Edge) -> Dict[str, Any]:
        """
        Convert an Edge to a Max/MSP line
        
        Args:
            edge: The Edge to convert
            
        Returns:
            Dictionary representing a Max/MSP line
        """
        line = {
            "patchline": {
                "type": edge.type,
                "disabled": 1 if edge.is_disabled else 0,
                "hidden": 1 if edge.is_hidden else 0
            },
            "source": {
                "id": edge.source["node_id"],
                "index": edge.source["port"]
            },
            "destination": {
                "id": edge.target["node_id"],
                "index": edge.target["port"]
            }
        }
        
        # Add midpoints if available
        if edge.midpoints:
            line["patchline"]["midpoints"] = edge.midpoints
        
        return line
    
    @staticmethod
    def _convert_graph_to_maxpat_patcher(graph: PatchGraph, 
                                        subpatches: Dict[str, PatchGraph]) -> Dict[str, Any]:
        """
        Convert a PatchGraph to a Max/MSP patcher dictionary
        
        Args:
            graph: The PatchGraph to convert
            subpatches: Dictionary of subpatches
            
        Returns:
            Dictionary representing a Max/MSP patcher
        """
        # Create patcher structure
        patcher = {
            "fileversion": 1,
            "rect": [0, 0, 640, 480],
            "bglocked": 0,
            "openinpresentation": 0,
            "boxes": [],
            "lines": [],
            "name": graph.name
        }
        
        # Convert nodes to boxes
        for node_id, node in graph.nodes.items():
            box = PatchLoader._convert_node_to_maxpat_box(node, subpatches)
            patcher["boxes"].append(box)
        
        # Convert edges to lines
        for edge_id, edge in graph.edges.items():
            line = PatchLoader._convert_edge_to_maxpat_line(edge)
            patcher["lines"].append(line)
        
        return patcher