#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
subpatcher_analyzer.py - Analyzes the hierarchical structure of Max/MSP patches

This module provides functionality to analyze the hierarchical structure of Max/MSP patches,
including parent-child relationships, data flow between hierarchy levels, and visualization
of the hierarchical structure.
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"subpatcher_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SubpatcherAnalyzer:
    """Analyzes hierarchical structures in Max/MSP patches"""
    
    # Dictionary of known subpatcher types and their characteristics
    SUBPATCHER_TYPES = {
        "p": {"type": "embedded", "description": "Standard embedded subpatcher"},
        "patcher": {"type": "embedded", "description": "Standard embedded subpatcher"},
        "bpatcher": {"type": "referenced", "description": "Referenced patcher file (bpatcher)"},
        "poly~": {"type": "audio", "description": "Polyphonic audio subpatcher"},
        "pattrstorage": {"type": "state", "description": "State storage for parameters"},
        "autopattr": {"type": "binding", "description": "Automatic parameter binding"},
        "plugout~": {"type": "audio_out", "description": "Audio output routing"},
        "plugin~": {"type": "audio_in", "description": "Audio input routing"},
        "pfft~": {"type": "fft", "description": "FFT-based processing subpatcher"},
        "gen~": {"type": "codegen", "description": "Gen signal processing patcher"},
        "jit.gen": {"type": "visual_codegen", "description": "Gen-based Jitter processing"},
        "js": {"type": "script", "description": "JavaScript embedded code"},
        "jsui": {"type": "ui_script", "description": "JavaScript UI element"},
        "bp.": {"type": "module", "description": "BEAP module (usually bpatcher)"},
        "M4L.": {"type": "m4l", "description": "Max for Live device component"}
    }
    
    def __init__(self, debug=False):
        """Initialize the SubpatcherAnalyzer"""
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)
        
        # Main storage for hierarchical information
        self.hierarchy_graph = nx.DiGraph()  # Parent-child relationships
        self.connection_graph = nx.MultiDiGraph()  # Data flow connections
        
        # Storage for patch data
        self.patch_data = {}  # Original JSON data
        self.processed_patches = {}  # Processed patch information
        self.subpatcher_mapping = {}  # Maps patcher IDs to their subpatcher nodes
        
        # Analysis results
        self.hierarchy_stats = {}
        self.connection_flows = {}
        self.inlet_outlet_mappings = {}
    
    def load_patch(self, file_path: str) -> bool:
        """
        Load a Max/MSP patch file (.maxpat) for analysis
        
        Args:
            file_path: Path to the .maxpat file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                patch_data = json.load(f)
            
            patch_id = os.path.basename(file_path).split('.')[0]
            self.patch_data[patch_id] = patch_data
            logger.info(f"Successfully loaded patch: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading patch file {file_path}: {str(e)}")
            return False
    
    def analyze_hierarchy(self, patch_id: str) -> nx.DiGraph:
        """
        Analyze the hierarchical structure of a patch
        
        Args:
            patch_id: ID of the patch to analyze
            
        Returns:
            A networkx DiGraph representing the hierarchy
        """
        if patch_id not in self.patch_data:
            logger.error(f"Patch ID {patch_id} not found in loaded patches")
            return nx.DiGraph()
        
        # Reset hierarchy graph for this analysis
        self.hierarchy_graph = nx.DiGraph()
        
        # Extract patcher data
        patch_data = self.patch_data[patch_id]
        patcher = patch_data.get("patcher", {})
        
        # Add the root patcher to the hierarchy graph
        root_id = f"{patch_id}_root"
        self.hierarchy_graph.add_node(
            root_id, 
            type="root", 
            level=0, 
            name=patcher.get("name", patch_id),
            box_count=len(patcher.get("boxes", {}))
        )
        
        # Process the patch hierarchy recursively
        self._process_patcher_hierarchy(patcher, root_id, 0, patch_id)
        
        # Calculate hierarchy statistics
        self._calculate_hierarchy_stats()
        
        logger.info(f"Hierarchy analysis complete for {patch_id}")
        logger.info(f"Found {self.hierarchy_graph.number_of_nodes()} patcher objects in hierarchy")
        
        return self.hierarchy_graph
    
    def _process_patcher_hierarchy(self, patcher: Dict, parent_id: str, level: int, patch_id: str) -> None:
        """
        Recursively process a patcher and its subpatchers to build the hierarchy
        
        Args:
            patcher: Dictionary containing the patcher data
            parent_id: ID of the parent patcher node
            level: Current hierarchy level (depth)
            patch_id: ID of the original patch
        """
        # Process boxes in this patcher
        boxes = patcher.get("boxes", {})
        
        # Track processed box IDs to avoid duplicates
        processed_box_ids = set()
        
        # First, identify potential subpatchers and their types
        for box_id, box_data in boxes.items():
            # Skip if already processed
            if box_id in processed_box_ids:
                continue
                
            # Get the actual box content (handle different versions)
            if "box" in box_data:
                box = box_data["box"]
            else:
                box = box_data
            
            # Mark as processed
            processed_box_ids.add(box_id)
            
            # Extract basic information
            maxclass = box.get("maxclass", "")
            
            # Check if this is a subpatcher type
            if self._is_subpatcher_type(maxclass):
                # Create subpatcher node ID
                subpatcher_id = f"{patch_id}_{box_id}"
                
                # Get subpatcher attributes
                subpatcher_name = self._extract_subpatcher_name(box)
                subpatcher_type = self._get_subpatcher_type(maxclass)
                
                # Add to hierarchy graph
                self.hierarchy_graph.add_node(
                    subpatcher_id,
                    type=subpatcher_type["type"],
                    level=level + 1,
                    name=subpatcher_name,
                    maxclass=maxclass,
                    box_id=box_id,
                    description=subpatcher_type["description"]
                )
                
                # Add edge from parent to this subpatcher
                self.hierarchy_graph.add_edge(parent_id, subpatcher_id)
                
                # Map this box ID to its subpatcher node
                self.subpatcher_mapping[box_id] = subpatcher_id
                
                # If this box contains an embedded subpatcher, process it recursively
                if "patcher" in box:
                    subpatcher_data = box["patcher"]
                    subbox_count = len(subpatcher_data.get("boxes", {}))
                    
                    # Update node with box count
                    self.hierarchy_graph.nodes[subpatcher_id]["box_count"] = subbox_count
                    
                    # Process recursively
                    self._process_patcher_hierarchy(subpatcher_data, subpatcher_id, level + 1, patch_id)
                else:
                    # External patcher reference
                    self.hierarchy_graph.nodes[subpatcher_id]["box_count"] = 0
                    self.hierarchy_graph.nodes[subpatcher_id]["external_reference"] = True
    
    def _is_subpatcher_type(self, maxclass: str) -> bool:
        """
        Check if a maxclass represents a subpatcher type
        
        Args:
            maxclass: The maxclass string to check
            
        Returns:
            True if this is a subpatcher type, False otherwise
        """
        # Check direct matches in our subpatcher types dictionary
        if maxclass in self.SUBPATCHER_TYPES:
            return True
        
        # Check for pattern matches (e.g., bp.* for BEAP modules)
        for prefix in ["bp.", "M4L."]:
            if maxclass.startswith(prefix):
                return True
        
        return False
    
    def _extract_subpatcher_name(self, box: Dict) -> str:
        """
        Extract the name of a subpatcher from its box data
        
        Args:
            box: Dictionary containing the box data
            
        Returns:
            The name of the subpatcher
        """
        # Try to get name from various possible locations
        if "patcher" in box and "name" in box["patcher"]:
            return box["patcher"]["name"]
        
        if "name" in box:
            return box["name"]
        
        # Try to extract from text
        if "text" in box:
            text = box["text"]
            # If text contains multiple parts (e.g., "p mysubpatcher")
            if isinstance(text, str) and " " in text:
                parts = text.split(None, 1)
                if len(parts) > 1:
                    return parts[1]
        
        # Default to box ID or generic name
        return box.get("id", "unnamed_subpatcher")
    
    def _get_subpatcher_type(self, maxclass: str) -> Dict:
        """
        Get information about a subpatcher type
        
        Args:
            maxclass: The maxclass string
            
        Returns:
            Dictionary with type information
        """
        # Check direct matches
        if maxclass in self.SUBPATCHER_TYPES:
            return self.SUBPATCHER_TYPES[maxclass]
        
        # Check for pattern matches
        if maxclass.startswith("bp."):
            return {
                "type": "module",
                "description": "BEAP module (usually bpatcher)"
            }
        
        if maxclass.startswith("M4L."):
            return {
                "type": "m4l",
                "description": "Max for Live device component"
            }
        
        # Default type
        return {
            "type": "unknown",
            "description": f"Unknown subpatcher type: {maxclass}"
        }
    
    def analyze_connections(self, patch_id: str) -> nx.MultiDiGraph:
        """
        Analyze connections between hierarchy levels
        
        Args:
            patch_id: ID of the patch to analyze
            
        Returns:
            A networkx MultiDiGraph representing the connections
        """
        if patch_id not in self.patch_data:
            logger.error(f"Patch ID {patch_id} not found in loaded patches")
            return nx.MultiDiGraph()
        
        # Reset connection graph for this analysis
        self.connection_graph = nx.MultiDiGraph()
        
        # Extract patcher data
        patch_data = self.patch_data[patch_id]
        patcher = patch_data.get("patcher", {})
        
        # Process the connections recursively
        self._process_patcher_connections(patcher, None, patch_id)
        
        # Analyze connection flows between hierarchy levels
        self._analyze_connection_flows()
        
        logger.info(f"Connection analysis complete for {patch_id}")
        logger.info(f"Found {self.connection_graph.number_of_edges()} connections between hierarchy levels")
        
        return self.connection_graph
    
    def _process_patcher_connections(self, patcher: Dict, parent_box_id: Optional[str], patch_id: str) -> None:
        """
        Process connections within a patcher and across hierarchy levels
        
        Args:
            patcher: Dictionary containing the patcher data
            parent_box_id: ID of the parent box, if any
            patch_id: ID of the original patch
        """
        # Get boxes and lines
        boxes = patcher.get("boxes", {})
        lines = patcher.get("lines", [])
        
        # Map box IDs to their maxclass for quick lookup
        box_types = {}
        for box_id, box_data in boxes.items():
            # Get the actual box content
            if "box" in box_data:
                box = box_data["box"]
            else:
                box = box_data
                
            box_types[box_id] = box.get("maxclass", "")
            
            # Process subpatchers recursively
            if self._is_subpatcher_type(box.get("maxclass", "")):
                if "patcher" in box:
                    self._process_patcher_connections(box["patcher"], box_id, patch_id)
        
        # Process inlet and outlet objects to map connection points between hierarchy levels
        inlet_outlet_mapping = self._map_inlets_outlets(boxes, parent_box_id, patch_id)
        
        # Process connections
        for line in lines:
            # Skip if missing essential information
            if not all(key in line for key in ["source", "destination"]):
                continue
                
            source = line["source"]
            dest = line["destination"]
            
            # Skip if missing source or destination details
            if not all(key in source for key in ["id", "index"]) or \
               not all(key in dest for key in ["id", "index"]):
                continue
            
            # Extract connection information
            source_id = source["id"]
            source_port = source["index"]
            dest_id = dest["id"]
            dest_port = dest["index"]
            
            # Determine connection type
            if "patchline" in line:
                patchline = line["patchline"]
                connection_type = patchline.get("type", "control")
            else:
                connection_type = "control"
            
            # Check if this is a hierarchy-crossing connection
            self._handle_hierarchy_connection(
                source_id, source_port, dest_id, dest_port,
                box_types, inlet_outlet_mapping, connection_type, patch_id
            )
    
    def _map_inlets_outlets(self, boxes: Dict, parent_box_id: Optional[str], patch_id: str) -> Dict:
        """
        Map inlet and outlet objects to their respective ports on the parent patcher
        
        Args:
            boxes: Dictionary of boxes in the patcher
            parent_box_id: ID of the parent box, if any
            patch_id: ID of the original patch
            
        Returns:
            Dictionary mapping inlet/outlet objects to parent ports
        """
        mapping = {
            "inlets": {},  # Maps inlet objects to parent inlet ports
            "outlets": {}  # Maps outlet objects to parent outlet ports
        }
        
        if parent_box_id is None:
            # Root level has no parent
            return mapping
        
        # Track the order of inlet and outlet objects
        inlet_objects = []
        outlet_objects = []
        
        # First pass: identify inlet and outlet objects
        for box_id, box_data in boxes.items():
            # Get the actual box content
            if "box" in box_data:
                box = box_data["box"]
            else:
                box = box_data
                
            maxclass = box.get("maxclass", "")
            
            # Check if this is an inlet or outlet object
            if maxclass == "inlet":
                inlet_objects.append((box_id, box))
            elif maxclass == "outlet":
                outlet_objects.append((box_id, box))
        
        # Sort inlet and outlet objects by their x-coordinate to determine order
        inlet_objects.sort(key=lambda x: self._get_box_x_coordinate(x[1]))
        outlet_objects.sort(key=lambda x: self._get_box_x_coordinate(x[1]))
        
        # Map inlets
        for idx, (box_id, _) in enumerate(inlet_objects):
            mapping["inlets"][box_id] = idx
        
        # Map outlets
        for idx, (box_id, _) in enumerate(outlet_objects):
            mapping["outlets"][box_id] = idx
        
        # Store this mapping
        if parent_box_id not in self.inlet_outlet_mappings:
            self.inlet_outlet_mappings[parent_box_id] = mapping
        
        return mapping
    
    def _get_box_x_coordinate(self, box: Dict) -> float:
        """
        Get the x-coordinate of a box
        
        Args:
            box: Dictionary containing the box data
            
        Returns:
            The x-coordinate as a float
        """
        if "patching_rect" in box and len(box["patching_rect"]) >= 1:
            return float(box["patching_rect"][0])
        return 0.0
    
    def _handle_hierarchy_connection(
        self, source_id: str, source_port: int, 
        dest_id: str, dest_port: int,
        box_types: Dict, inlet_outlet_mapping: Dict, 
        connection_type: str, patch_id: str
    ) -> None:
        """
        Handle a potential hierarchy-crossing connection
        
        Args:
            source_id: ID of the source box
            source_port: Port index of the source
            dest_id: ID of the destination box
            dest_port: Port index of the destination
            box_types: Dictionary mapping box IDs to their maxclass
            inlet_outlet_mapping: Mapping of inlet/outlet objects to parent ports
            connection_type: Type of the connection (control, signal, etc.)
            patch_id: ID of the original patch
        """
        # Check if source is an outlet object (connecting to parent patcher)
        if source_id in box_types and box_types[source_id] == "outlet":
            # This is a connection from a subpatcher to its parent
            if source_id in inlet_outlet_mapping["outlets"]:
                parent_port = inlet_outlet_mapping["outlets"][source_id]
                
                # Add this to our connection graph
                if source_id in self.subpatcher_mapping and dest_id in self.subpatcher_mapping:
                    source_patcher = self.subpatcher_mapping[source_id]
                    dest_patcher = self.subpatcher_mapping[dest_id]
                    
                    self.connection_graph.add_edge(
                        source_patcher, dest_patcher,
                        type=connection_type,
                        source_port=parent_port,
                        dest_port=dest_port
                    )
        
        # Check if destination is an inlet object (connecting from parent patcher)
        if dest_id in box_types and box_types[dest_id] == "inlet":
            # This is a connection from a parent to a subpatcher
            if dest_id in inlet_outlet_mapping["inlets"]:
                parent_port = inlet_outlet_mapping["inlets"][dest_id]
                
                # Add this to our connection graph
                if source_id in self.subpatcher_mapping and dest_id in self.subpatcher_mapping:
                    source_patcher = self.subpatcher_mapping[source_id]
                    dest_patcher = self.subpatcher_mapping[dest_id]
                    
                    self.connection_graph.add_edge(
                        source_patcher, dest_patcher,
                        type=connection_type,
                        source_port=source_port,
                        dest_port=parent_port
                    )
    
    def _calculate_hierarchy_stats(self) -> None:
        """Calculate statistics about the hierarchy"""
        # Reset hierarchy stats
        self.hierarchy_stats = {
            "total_levels": 0,
            "total_patchers": 0,
            "patchers_by_level": {},
            "patchers_by_type": {},
            "max_depth": 0,
            "avg_boxes_per_patcher": 0.0,
            "largest_patcher": {"id": None, "box_count": 0},
            "smallest_patcher": {"id": None, "box_count": float('inf')}
        }
        
        total_box_count = 0
        patcher_count = 0
        
        # Analyze each node in the hierarchy
        for node_id, node_data in self.hierarchy_graph.nodes(data=True):
            level = node_data.get("level", 0)
            node_type = node_data.get("type", "unknown")
            box_count = node_data.get("box_count", 0)
            
            # Skip the root node for certain calculations
            if node_type != "root":
                # Update total counts
                patcher_count += 1
                total_box_count += box_count
                
                # Track by level
                if level not in self.hierarchy_stats["patchers_by_level"]:
                    self.hierarchy_stats["patchers_by_level"][level] = 0
                self.hierarchy_stats["patchers_by_level"][level] += 1
                
                # Track by type
                if node_type not in self.hierarchy_stats["patchers_by_type"]:
                    self.hierarchy_stats["patchers_by_type"][node_type] = 0
                self.hierarchy_stats["patchers_by_type"][node_type] += 1
                
                # Track largest and smallest
                if box_count > self.hierarchy_stats["largest_patcher"]["box_count"]:
                    self.hierarchy_stats["largest_patcher"] = {"id": node_id, "box_count": box_count}
                
                if box_count < self.hierarchy_stats["smallest_patcher"]["box_count"] and box_count > 0:
                    self.hierarchy_stats["smallest_patcher"] = {"id": node_id, "box_count": box_count}
            
            # Update max depth
            if level > self.hierarchy_stats["max_depth"]:
                self.hierarchy_stats["max_depth"] = level
        
        # Calculate averages
        self.hierarchy_stats["total_patchers"] = patcher_count
        self.hierarchy_stats["total_levels"] = self.hierarchy_stats["max_depth"] + 1
        
        if patcher_count > 0:
            self.hierarchy_stats["avg_boxes_per_patcher"] = total_box_count / patcher_count
        
        # If no patchers were found with boxes, reset smallest patcher
        if self.hierarchy_stats["smallest_patcher"]["box_count"] == float('inf'):
            self.hierarchy_stats["smallest_patcher"] = {"id": None, "box_count": 0}
    
    def _analyze_connection_flows(self) -> None:
        """Analyze connection flows between hierarchy levels"""
        # Reset connection flows
        self.connection_flows = {
            "total_connections": 0,
            "connection_types": {},
            "level_flows": {},
            "type_flows": {}
        }
        
        # Count total connections
        self.connection_flows["total_connections"] = self.connection_graph.number_of_edges()
        
        # Analyze each connection
        for source, target, edge_data in self.connection_graph.edges(data=True):
            # Get node information
            source_data = self.hierarchy_graph.nodes[source]
            target_data = self.hierarchy_graph.nodes[target]
            
            source_level = source_data.get("level", 0)
            target_level = target_data.get("level", 0)
            source_type = source_data.get("type", "unknown")
            target_type = target_data.get("type", "unknown")
            connection_type = edge_data.get("type", "control")
            
            # Track connection types
            if connection_type not in self.connection_flows["connection_types"]:
                self.connection_flows["connection_types"][connection_type] = 0
            self.connection_flows["connection_types"][connection_type] += 1
            
            # Track level flows
            level_flow = f"L{source_level}->L{target_level}"
            if level_flow not in self.connection_flows["level_flows"]:
                self.connection_flows["level_flows"][level_flow] = 0
            self.connection_flows["level_flows"][level_flow] += 1
            
            # Track type flows
            type_flow = f"{source_type}->{target_type}"
            if type_flow not in self.connection_flows["type_flows"]:
                self.connection_flows["type_flows"][type_flow] = 0
            self.connection_flows["type_flows"][type_flow] += 1
    
    def visualize_hierarchy(self, patch_id: str, output_file: Optional[str] = None) -> None:
        """
        Visualize the patch hierarchy as a graph
        
        Args:
            patch_id: ID of the patch to visualize
            output_file: Optional path to save the visualization
        """
        if patch_id not in self.patch_data:
            logger.error(f"Patch ID {patch_id} not found in loaded patches")
            return
        
        # Create a new graph layout
        plt.figure(figsize=(12, 8))
        pos = nx.nx_agraph.graphviz_layout(self.hierarchy_graph, prog="dot")
        
        # Prepare node colors based on type
        colors = []
        node_types = nx.get_node_attributes(self.hierarchy_graph, "type")
        for node in self.hierarchy_graph.nodes():
            node_type = node_types.get(node, "unknown")
            if node_type == "root":
                colors.append("lightblue")
            elif node_type == "embedded":
                colors.append("lightgreen")
            elif node_type == "audio":
                colors.append("lightcoral")
            elif node_type == "referenced":
                colors.append("yellow")
            else:
                colors.append("lightgray")
        
        # Draw nodes and edges
        nx.draw(
            self.hierarchy_graph, 
            pos, 
            with_labels=True, 
            node_color=colors, 
            node_size=1000, 
            font_size=8, 
            font_weight="bold",
            arrowsize=15, 
            alpha=0.8
        )
        
        # Add title
        plt.title(f"Hierarchy Structure for {patch_id}")
        
        # Save or display
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            logger.info(f"Saved hierarchy visualization to {output_file}")
        else:
            plt.show()
    
    def visualize_connection_flows(self, patch_id: str, output_file: Optional[str] = None) -> None:
        """
        Visualize connection flows between hierarchy levels
        
        Args:
            patch_id: ID of the patch to visualize
            output_file: Optional path to save the visualization
        """
        if patch_id not in self.patch_data:
            logger.error(f"Patch ID {patch_id} not found in loaded patches")
            return
        
        # Prepare data
        if not self.connection_flows or "level_flows" not in self.connection_flows:
            logger.error("No connection flow data available. Run analyze_connections first.")
            return
        
        # Create a DataFrame for visualization
        flow_data = []
        for flow, count in self.connection_flows["level_flows"].items():
            source_level, target_level = flow.split("->")
            flow_data.append({
                "source": source_level,
                "target": target_level,
                "value": count
            })
        
        df = pd.DataFrame(flow_data)
        
        # Create a new graph layout
        plt.figure(figsize=(10, 6))
        
        # Create a matrix representation
        levels = sorted(set(df["source"].unique()) | set(df["target"].unique()))
        flow_matrix = np.zeros((len(levels), len(levels)))
        
        level_indices = {level: i for i, level in enumerate(levels)}
        
        for _, row in df.iterrows():
            source_idx = level_indices[row["source"]]
            target_idx = level_indices[row["target"]]
            flow_matrix[source_idx, target_idx] = row["value"]
        
        # Plot the heatmap
        plt.imshow(flow_matrix, cmap='viridis')
        plt.colorbar(label='Connection count')
        
        # Add labels
        plt.xticks(range(len(levels)), levels)
        plt.yticks(range(len(levels)), levels)
        plt.xlabel('Destination Level')
        plt.ylabel('Source Level')
        
        # Add title
        plt.title(f"Connection Flows Between Hierarchy Levels for {patch_id}")
        
        # Add text annotations
        for i in range(len(levels)):
            for j in range(len(levels)):
                if flow_matrix[i, j] > 0:
                    plt.text(j, i, int(flow_matrix[i, j]), 
                             ha="center", va="center", 
                             color="white" if flow_matrix[i, j] > flow_matrix.max() / 2 else "black")
        
        # Save or display
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            logger.info(f"Saved connection flow visualization to {output_file}")
        else:
            plt.show()
    
    def generate_hierarchy_report(self, patch_id: str, output_file: Optional[str] = None) -> Dict:
        """
        Generate a comprehensive report on the patch hierarchy
        
        Args:
            patch_id: ID of the patch to report on
            output_file: Optional path to save the report
            
        Returns:
            Dictionary containing the report data
        """
        if patch_id not in self.patch_data:
            logger.error(f"Patch ID {patch_id} not found in loaded patches")
            return {}
        
        # Create report data
        report = {
            "patch_id": patch_id,
            "analysis_time": datetime.now().isoformat(),
            "hierarchy": self.hierarchy_stats,
            "connections": self.connection_flows,
            "details": {
                "patchers": [],
                "connections": []
            }
        }
        
        # Add details for each patcher
        for node_id, node_data in self.hierarchy_graph.nodes(data=True):
            patcher_info = {
                "id": node_id,
                "name": node_data.get("name", "unnamed"),
                "type": node_data.get("type", "unknown"),
                "level": node_data.get("level", 0),
                "box_count": node_data.get("box_count", 0),
                "children": list(self.hierarchy_graph.successors(node_id)),
                "parent": list(self.hierarchy_graph.predecessors(node_id))
            }
            report["details"]["patchers"].append(patcher_info)
        
        # Add details for each connection
        for source, target, edge_data in self.connection_graph.edges(data=True):
            connection_info = {
                "source": source,
                "target": target,
                "type": edge_data.get("type", "control"),
                "source_port": edge_data.get("source_port", 0),
                "target_port": edge_data.get("target_port", 0)
            }
            report["details"]["connections"].append(connection_info)
        
        # Save report if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved hierarchy report to {output_file}")
        
        return report
    
    def export_hierarchy_to_networkx(self, patch_id: str, output_file: Optional[str] = None) -> None:
        """
        Export the hierarchy graph to a NetworkX-compatible format
        
        Args:
            patch_id: ID of the patch to export
            output_file: Optional path to save the graph
        """
        if patch_id not in self.patch_data:
            logger.error(f"Patch ID {patch_id} not found in loaded patches")
            return
        
        # Create a JSON representation of the graph
        data = nx.readwrite.json_graph.node_link_data(self.hierarchy_graph)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Exported hierarchy graph to {output_file}")
        
        return data


def main():
    """Main function to run the analyzer from command line"""
    parser = argparse.ArgumentParser(description="Analyze hierarchical structure of Max/MSP patches")
    parser.add_argument("--input", "-i", required=True, help="Path to .maxpat file")
    parser.add_argument("--output-dir", "-o", default="./output", help="Directory for output files")
    parser.add_argument("--visualize", "-v", action="store_true", help="Generate visualizations")
    parser.add_argument("--report", "-r", action="store_true", help="Generate detailed report")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = SubpatcherAnalyzer(debug=args.debug)
    
    # Load and analyze patch
    if analyzer.load_patch(args.input):
        patch_id = os.path.basename(args.input).split('.')[0]
        
        # Analyze hierarchy
        analyzer.analyze_hierarchy(patch_id)
        
        # Analyze connections
        analyzer.analyze_connections(patch_id)
        
        # Generate outputs
        if args.visualize:
            hierarchy_img = os.path.join(args.output_dir, f"{patch_id}_hierarchy.png")
            flow_img = os.path.join(args.output_dir, f"{patch_id}_flows.png")
            analyzer.visualize_hierarchy(patch_id, hierarchy_img)
            analyzer.visualize_connection_flows(patch_id, flow_img)
        
        if args.report:
            report_file = os.path.join(args.output_dir, f"{patch_id}_report.json")
            graph_file = os.path.join(args.output_dir, f"{patch_id}_graph.json")
            analyzer.generate_hierarchy_report(patch_id, report_file)
            analyzer.export_hierarchy_to_networkx(patch_id, graph_file)
        
        # Log summary
        logger.info("Analysis completed successfully")
        logger.info(f"Hierarchy depth: {analyzer.hierarchy_stats['max_depth']}")
        logger.info(f"Total patchers: {analyzer.hierarchy_stats['total_patchers']}")
        logger.info(f"Total connections: {analyzer.connection_flows['total_connections']}")
    else:
        logger.error(f"Failed to analyze {args.input}")


if __name__ == "__main__":
    main()