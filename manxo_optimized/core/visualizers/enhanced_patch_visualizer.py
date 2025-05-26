import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import networkx as nx
from patch_graph import PatchGraph
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime

class EnhancedPatchVisualizer:
    """Enhanced utility class for visualizing patch graphs with advanced features"""
    
    # Extended color schemes for different node types with more variations
    NODE_COLORS = {
        "object": "#1f77b4",     # Blue
        "message": "#ff7f0e",    # Orange
        "comment": "#2ca02c",    # Green
        "patcher": "#d62728",    # Red
        "audio": "#9467bd",      # Purple
        "visual": "#8c564b",     # Brown
        "ui": "#e377c2",         # Pink
        "control": "#17becf",    # Cyan
        "data": "#bcbd22",       # Yellow-green
        "midi": "#7f7f7f",       # Gray
        "jitter": "#c49c94",     # Light brown
        "gen": "#f7b6d2",        # Light pink
        "math": "#aec7e8",       # Light blue
        "timing": "#ffbb78",     # Light orange
        "signal": "#98df8a",     # Light green
        "filter": "#ff9896",     # Light red
        "networking": "#c5b0d5", # Light purple
        "sequencing": "#c49c94", # Tan
        "spatial": "#dbdb8d",    # Olive
        "conversion": "#9edae5", # Light cyan
        "generic": "#7f7f7f"     # Gray
    }
    
    # Extended edge colors 
    EDGE_COLORS = {
        "control": "black",
        "signal": "blue",
        "audio": "red",
        "midi": "green",
        "matrix": "purple",
        "jitter": "orange",
        "data": "brown",
        "network": "cyan",
        "virtual": "gray"
    }
    
    # Box style mapping for different node types
    BOX_STYLES = {
        "object": {"shape": "rectangle", "alpha": 0.8},
        "message": {"shape": "rectangle", "alpha": 0.8},
        "comment": {"shape": "rectangle", "alpha": 0.5, "rounded": True},
        "patcher": {"shape": "rectangle", "alpha": 0.8, "rounded": True},
        "audio": {"shape": "rectangle", "alpha": 0.8},
        "visual": {"shape": "rectangle", "alpha": 0.8},
        "ui": {"shape": "rectangle", "alpha": 0.8, "rounded": True},
        "control": {"shape": "rectangle", "alpha": 0.8},
        "data": {"shape": "rectangle", "alpha": 0.8},
        "midi": {"shape": "rectangle", "alpha": 0.8},
        "jitter": {"shape": "rectangle", "alpha": 0.8},
        "gen": {"shape": "rectangle", "alpha": 0.8, "rounded": True},
        "math": {"shape": "rectangle", "alpha": 0.8},
        "timing": {"shape": "rectangle", "alpha": 0.8},
        "signal": {"shape": "rectangle", "alpha": 0.8},
        "filter": {"shape": "rectangle", "alpha": 0.8},
        "networking": {"shape": "rectangle", "alpha": 0.8},
        "sequencing": {"shape": "rectangle", "alpha": 0.8},
        "spatial": {"shape": "rectangle", "alpha": 0.8},
        "conversion": {"shape": "rectangle", "alpha": 0.8},
        "generic": {"shape": "rectangle", "alpha": 0.7}
    }
    
    @staticmethod
    def visualize_graph(graph: PatchGraph, output_file: str = None, 
                      show_ports: bool = True, show_labels: bool = True,
                      show_node_types: bool = True, 
                      highlight_central_nodes: bool = True,
                      layout_algorithm: str = "spring") -> None:
        """
        Visualize a patch graph using matplotlib with enhanced features
        
        Args:
            graph: The PatchGraph to visualize
            output_file: Optional file path to save the visualization
            show_ports: Whether to show the inlet/outlet ports
            show_labels: Whether to show node labels
            show_node_types: Whether to show node types with labels
            highlight_central_nodes: Whether to highlight central nodes in the graph
            layout_algorithm: Graph layout algorithm ('spring', 'kamada_kawai', 'spectral', or 'original')
        """
        # Create figure with higher resolution
        plt.figure(figsize=(14, 10), dpi=100)
        
        # Create a networkx graph for layout calculation
        G = nx.DiGraph()
        
        # Add nodes with positions
        for node_id, node in graph.nodes.items():
            # Use the node's position if available, otherwise let networkx decide
            if node.position and "x" in node.position and "y" in node.position:
                pos = (node.position["x"], -node.position["y"])  # Negate y for top-down layout
            else:
                pos = None
                
            # Get node type and other attributes
            if hasattr(node, 'box_type'):
                box_type = node.box_type
            elif node.category:
                box_type = node.category
            else:
                box_type = "generic"
                
            # Get node label
            if hasattr(node, 'display_name'):
                label = node.display_name
            else:
                label = node.type
                
            # Add to graph
            G.add_node(node_id, position=pos, box_type=box_type, 
                      label=label, type=node.type, 
                      inlets=len(node.inlets), outlets=len(node.outlets))
        
        # Add edges
        for edge_id, edge in graph.edges.items():
            source = edge.source["node_id"]
            target = edge.target["node_id"]
            edge_type = edge.type
            
            # Add to graph
            G.add_edge(source, target, id=edge_id, type=edge_type, 
                      source_port=edge.source["port"], 
                      target_port=edge.target["port"])
        
        # Get positions from nodes or calculate layout
        pos = {}
        for node_id, node_data in G.nodes(data=True):
            if node_data["position"] is not None:
                pos[node_id] = node_data["position"]
        
        # Apply different layout algorithms based on parameter
        if len(pos) < len(G.nodes) or layout_algorithm != "original":
            # Create a copy of G to avoid modifying the original
            G_layout = G.copy()
            
            # Keep only nodes without fixed positions for layout calculation
            if layout_algorithm != "original":
                G_layout = G.copy()
                pos = {}  # Reset positions if not using original layout
            else:
                # Remove nodes that have fixed positions
                for node_id in list(pos.keys()):
                    G_layout.remove_node(node_id)
            
            # Calculate positions based on selected algorithm
            if layout_algorithm == "spring" or layout_algorithm == "original":
                pos_calculated = nx.spring_layout(G_layout, k=0.6, iterations=100, seed=42)
            elif layout_algorithm == "kamada_kawai":
                pos_calculated = nx.kamada_kawai_layout(G_layout)
            elif layout_algorithm == "spectral":
                pos_calculated = nx.spectral_layout(G_layout)
            else:  # Fallback to spring
                pos_calculated = nx.spring_layout(G_layout, k=0.6, iterations=100, seed=42)
            
            # Combine positions
            pos.update(pos_calculated)
        
        # Analyze graph structure to find central nodes if highlighting is enabled
        node_centrality = {}
        if highlight_central_nodes and len(G.nodes) > 1:
            try:
                # Calculate various centrality metrics
                degree_centrality = nx.degree_centrality(G)
                betweenness_centrality = nx.betweenness_centrality(G)
                
                # Combine metrics (weighted average)
                for node in G.nodes:
                    node_centrality[node] = 0.5 * degree_centrality.get(node, 0) + 0.5 * betweenness_centrality.get(node, 0)
            except:
                # Fallback to degree centrality if other metrics fail
                node_centrality = nx.degree_centrality(G)
        
        # Draw nodes with enhanced styling
        for node_id, node_data in G.nodes(data=True):
            box_type = node_data["box_type"]
            color = EnhancedPatchVisualizer.NODE_COLORS.get(box_type, EnhancedPatchVisualizer.NODE_COLORS["generic"])
            
            # Adjust alpha based on centrality if highlighting is enabled
            alpha = 0.7
            if highlight_central_nodes and node_centrality:
                # Scale centrality to affect node appearance
                centrality = node_centrality.get(node_id, 0)
                alpha = 0.5 + min(centrality * 2, 0.5)  # Alpha between 0.5 and 1.0
            
            # Get box style for this node type
            box_style = EnhancedPatchVisualizer.BOX_STYLES.get(box_type, EnhancedPatchVisualizer.BOX_STYLES["generic"])
            
            # Create a rectangular patch for the node
            x, y = pos[node_id]
            
            # Scale box size based on number of inlets/outlets
            base_width, base_height = 0.08, 0.05
            width_scale = max(1.0, (node_data["inlets"] + node_data["outlets"]) / 4)
            width = base_width * min(width_scale, 2.0)  # Cap scaling
            height = base_height
            
            # Create rectangle with rounded corners if specified
            if box_style.get("rounded", False):
                # Create rounded rectangle
                rect = patches.FancyBboxPatch(
                    (x - width/2, y - height/2), width, height, 
                    boxstyle=patches.BoxStyle("round", pad=0.02),
                    linewidth=1, edgecolor='black', facecolor=color, alpha=alpha
                )
            else:
                # Create standard rectangle
                rect = patches.Rectangle(
                    (x - width/2, y - height/2), width, height, 
                    linewidth=1, edgecolor='black', facecolor=color, alpha=alpha
                )
            
            plt.gca().add_patch(rect)
            
            # Add label with node type if requested
            if show_labels:
                if show_node_types:
                    label_text = f"{node_data['label']}\n({node_data['type']})"
                else:
                    label_text = node_data["label"]
                
                plt.text(x, y, label_text, 
                        horizontalalignment='center',
                        verticalalignment='center', 
                        fontsize=8, 
                        bbox=dict(facecolor='white', alpha=0.7, pad=1, boxstyle='round'))
            
            # Draw ports if requested
            if show_ports:
                node = graph.nodes[node_id]
                port_width = width / 8
                
                # Draw inlets on top
                num_inlets = len(node.inlets)
                if num_inlets > 0:
                    inlet_spacing = width / (num_inlets + 1)
                    for i in range(num_inlets):
                        inlet_x = x - width/2 + (i + 1) * inlet_spacing
                        inlet_y = y + height/2
                        
                        # Use different shapes for different port types
                        inlet_type = node.inlets[i]["type"] if i < len(node.inlets) else "control"
                        
                        if inlet_type == "signal":
                            # Signal inlet (tilde) - use a circle
                            inlet_shape = patches.Circle(
                                (inlet_x, inlet_y), port_width/2,
                                linewidth=1, edgecolor='black', facecolor='white'
                            )
                        else:
                            # Standard inlet - use a square
                            inlet_shape = patches.Rectangle(
                                (inlet_x - port_width/2, inlet_y - port_width/2),
                                port_width, port_width,
                                linewidth=1, edgecolor='black', facecolor='white'
                            )
                        
                        plt.gca().add_patch(inlet_shape)
                
                # Draw outlets on bottom
                num_outlets = len(node.outlets)
                if num_outlets > 0:
                    outlet_spacing = width / (num_outlets + 1)
                    for i in range(num_outlets):
                        outlet_x = x - width/2 + (i + 1) * outlet_spacing
                        outlet_y = y - height/2
                        
                        # Use different shapes for different port types
                        outlet_type = node.outlets[i]["type"] if i < len(node.outlets) else "control"
                        
                        if outlet_type == "signal":
                            # Signal outlet (tilde) - use a circle
                            outlet_shape = patches.Circle(
                                (outlet_x, outlet_y), port_width/2,
                                linewidth=1, edgecolor='black', facecolor='white'
                            )
                        else:
                            # Standard outlet - use a square
                            outlet_shape = patches.Rectangle(
                                (outlet_x - port_width/2, outlet_y - port_width/2),
                                port_width, port_width,
                                linewidth=1, edgecolor='black', facecolor='white'
                            )
                        
                        plt.gca().add_patch(outlet_shape)
        
        # Draw edges with enhanced styling
        for source, target, edge_data in G.edges(data=True):
            edge_type = edge_data.get("type", "control")
            color = EnhancedPatchVisualizer.EDGE_COLORS.get(edge_type, "black")
            linestyle = "--" if edge_type == "virtual" else "-"
            
            # Get source node attributes
            source_node = graph.nodes[source]
            source_port = edge_data.get("source_port", 0)
            target_port = edge_data.get("target_port", 0)
            
            if show_ports:
                # Calculate port positions
                source_x, source_y = pos[source]
                target_x, target_y = pos[target]
                
                # Get box dimensions
                base_width, base_height = 0.08, 0.05
                width_scale_source = max(1.0, (len(source_node.inlets) + len(source_node.outlets)) / 4)
                width_source = base_width * min(width_scale_source, 2.0)
                height_source = base_height
                
                target_node = graph.nodes[target]
                width_scale_target = max(1.0, (len(target_node.inlets) + len(target_node.outlets)) / 4)
                width_target = base_width * min(width_scale_target, 2.0)
                height_target = base_height
                
                port_width = width_source / 8
                
                # Source outlet position
                num_outlets = len(source_node.outlets)
                if num_outlets > 0 and source_port < num_outlets:
                    outlet_spacing = width_source / (num_outlets + 1)
                    outlet_x = source_x - width_source/2 + (source_port + 1) * outlet_spacing
                    outlet_y = source_y - height_source/2
                else:
                    outlet_x, outlet_y = source_x, source_y - height_source/2
                
                # Target inlet position
                num_inlets = len(target_node.inlets)
                if num_inlets > 0 and target_port < num_inlets:
                    inlet_spacing = width_target / (num_inlets + 1)
                    inlet_x = target_x - width_target/2 + (target_port + 1) * inlet_spacing
                    inlet_y = target_y + height_target/2
                else:
                    inlet_x, inlet_y = target_x, target_y + height_target/2
                
                # Create a smooth curved path for the edge
                path_patch = patches.ConnectionPatch(
                    (outlet_x, outlet_y), (inlet_x, inlet_y),
                    'data', 'data', 
                    arrowstyle='-|>', 
                    shrinkA=0, shrinkB=0,
                    mutation_scale=15,
                    connectionstyle='arc3,rad=0.1',
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.5
                )
                plt.gca().add_patch(path_patch)
            else:
                # Draw a simple edge from node center to node center
                plt.plot([pos[source][0], pos[target][0]], 
                        [pos[source][1], pos[target][1]], 
                        color=color, linestyle=linestyle, linewidth=1.5)
        
        # Set up the plot
        plt.title(f"Patch Graph Visualization: {graph.name}")
        plt.axis('off')
        
        # Add a legend for node types
        unique_box_types = set(nx.get_node_attributes(G, 'box_type').values())
        legend_patches = []
        for box_type in unique_box_types:
            color = EnhancedPatchVisualizer.NODE_COLORS.get(box_type, EnhancedPatchVisualizer.NODE_COLORS["generic"])
            legend_patches.append(patches.Patch(color=color, label=box_type))
        
        plt.legend(handles=legend_patches, loc='upper right', fontsize='small')
        
        # Save or show
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Enhanced visualization saved to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def visualize_full_patch(main_graph: PatchGraph, subpatches: Dict[str, PatchGraph], 
                            output_dir: str, base_filename: str,
                            show_ports: bool = True, show_labels: bool = True,
                            layout_algorithm: str = "spring") -> None:
        """
        Visualize a complete patch with all its subpatches
        
        Args:
            main_graph: The main PatchGraph
            subpatches: Dictionary of subpatches
            output_dir: Directory to save the visualizations
            base_filename: Base name for the output files
            show_ports: Whether to show inlet/outlet ports
            show_labels: Whether to show node labels
            layout_algorithm: Graph layout algorithm to use
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Visualize main graph
        main_output = os.path.join(output_dir, f"{base_filename}_main.png")
        EnhancedPatchVisualizer.visualize_graph(
            main_graph, main_output, 
            show_ports=show_ports, 
            show_labels=show_labels,
            layout_algorithm=layout_algorithm
        )
        
        # Visualize each subpatch
        for subpatch_id, subpatch in subpatches.items():
            subpatch_output = os.path.join(output_dir, f"{base_filename}_{subpatch_id}.png")
            EnhancedPatchVisualizer.visualize_graph(
                subpatch, subpatch_output, 
                show_ports=show_ports, 
                show_labels=show_labels,
                layout_algorithm=layout_algorithm
            )
    
    @staticmethod
    def create_html_report(main_graph: PatchGraph, subpatches: Dict[str, PatchGraph],
                          output_dir: str, base_filename: str,
                          include_stats: bool = True,
                          generate_connection_matrix: bool = True) -> str:
        """
        Create an enhanced HTML report for the patch with all visualizations and statistics
        
        Args:
            main_graph: The main PatchGraph
            subpatches: Dictionary of subpatches
            output_dir: Directory to save the report
            base_filename: Base name for the output files
            include_stats: Whether to include statistical analysis
            generate_connection_matrix: Whether to generate connection matrices
            
        Returns:
            Path to the HTML report file
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate visualizations
        EnhancedPatchVisualizer.visualize_full_patch(main_graph, subpatches, output_dir, base_filename)
        
        # Gather statistics if requested
        stats = {}
        if include_stats:
            # Main graph statistics
            main_stats = EnhancedPatchVisualizer._calculate_graph_stats(main_graph)
            stats["main"] = main_stats
            
            # Subpatches statistics
            for subpatch_id, subpatch in subpatches.items():
                stats[subpatch_id] = EnhancedPatchVisualizer._calculate_graph_stats(subpatch)
        
        # Generate connection matrices if requested
        connection_matrices = {}
        if generate_connection_matrix and len(main_graph.nodes) > 0:
            # Main graph connection matrix
            main_matrix = EnhancedPatchVisualizer._generate_connection_matrix(main_graph)
            connection_matrices["main"] = main_matrix
            
            # Subpatches connection matrices
            for subpatch_id, subpatch in subpatches.items():
                if len(subpatch.nodes) > 0:
                    connection_matrices[subpatch_id] = EnhancedPatchVisualizer._generate_connection_matrix(subpatch)
        
        # Create HTML content with enhanced styling
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enhanced Patch Report: {main_graph.name}</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                :root {{
                    --main-bg-color: #f8f9fa;
                    --header-bg-color: #343a40;
                    --header-text-color: white;
                    --section-bg-color: white;
                    --section-border-color: #dee2e6;
                    --text-color: #212529;
                    --accent-color: #007bff;
                    --hover-color: #0056b3;
                }}
                
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0;
                    padding: 0;
                    background-color: var(--main-bg-color);
                    color: var(--text-color);
                }}
                
                .container {{
                    width: 95%;
                    margin: 0 auto;
                    padding: 20px 0;
                }}
                
                .header {{
                    background-color: var(--header-bg-color);
                    color: var(--header-text-color);
                    padding: 30px 0;
                    margin-bottom: 20px;
                    text-align: center;
                }}
                
                .header h1 {{
                    margin: 0;
                    font-size: 2.5em;
                }}
                
                .header p {{
                    margin: 10px 0 0;
                    opacity: 0.8;
                }}
                
                .section {{
                    background-color: var(--section-bg-color);
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    padding: 20px;
                    margin-bottom: 30px;
                }}
                
                .section h2 {{
                    margin-top: 0;
                    padding-bottom: 10px;
                    border-bottom: 1px solid var(--section-border-color);
                }}
                
                .section h3 {{
                    color: var(--accent-color);
                }}
                
                .tabs {{
                    display: flex;
                    border-bottom: 2px solid var(--section-border-color);
                    margin-bottom: 20px;
                }}
                
                .tab {{
                    padding: 10px 15px;
                    cursor: pointer;
                    margin-right: 5px;
                    background-color: #e9ecef;
                    border-radius: 5px 5px 0 0;
                }}
                
                .tab.active {{
                    background-color: var(--accent-color);
                    color: white;
                }}
                
                .tab-content {{
                    display: none;
                }}
                
                .tab-content.active {{
                    display: block;
                }}
                
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                
                th, td {{
                    border: 1px solid var(--section-border-color);
                    padding: 12px;
                    text-align: left;
                }}
                
                th {{
                    background-color: #e9ecef;
                    font-weight: 600;
                }}
                
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                
                tr:hover {{
                    background-color: #e2e6ea;
                }}
                
                .viz-image {{
                    max-width: 100%;
                    border: 1px solid var(--section-border-color);
                    border-radius: 5px;
                    margin-top: 20px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                }}
                
                .stats-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin-bottom: 20px;
                }}
                
                .stat-box {{
                    flex: 1 0 200px;
                    background-color: var(--section-bg-color);
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    padding: 15px;
                    text-align: center;
                }}
                
                .stat-box h4 {{
                    margin-top: 0;
                    color: var(--accent-color);
                }}
                
                .stat-box p {{
                    font-size: 1.5em;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                
                .footer {{
                    text-align: center;
                    padding: 20px;
                    margin-top: 20px;
                    font-size: 0.9em;
                    color: #6c757d;
                }}
                
                /* Responsive adjustments */
                @media (max-width: 768px) {{
                    .container {{
                        width: 95%;
                    }}
                    
                    .stats-container {{
                        flex-direction: column;
                    }}
                }}
                
                /* Matrix visualization */
                .matrix-container {{
                    overflow-x: auto;
                    margin-top: 20px;
                }}
                
                .connection-matrix {{
                    border-collapse: collapse;
                    font-size: 0.8em;
                }}
                
                .connection-matrix th {{
                    writing-mode: vertical-rl;
                    transform: rotate(180deg);
                    white-space: nowrap;
                    padding: 5px;
                }}
                
                .connection-matrix td {{
                    width: 20px;
                    height: 20px;
                    text-align: center;
                }}
                
                .connection-matrix td.connected {{
                    background-color: var(--accent-color);
                }}
            </style>
            <script>
                // Simple tab functionality
                document.addEventListener('DOMContentLoaded', function() {{
                    const tabs = document.querySelectorAll('.tab');
                    
                    tabs.forEach(tab => {{
                        tab.addEventListener('click', function() {{
                            // Remove active class from all tabs
                            tabs.forEach(t => t.classList.remove('active'));
                            
                            // Add active class to clicked tab
                            this.classList.add('active');
                            
                            // Hide all tab content
                            const tabContents = document.querySelectorAll('.tab-content');
                            tabContents.forEach(content => content.classList.remove('active'));
                            
                            // Show corresponding tab content
                            const tabContentId = this.getAttribute('data-tab');
                            document.getElementById(tabContentId).classList.add('active');
                        }});
                    }});
                    
                    // Activate first tab by default
                    if(tabs.length > 0) {{
                        tabs[0].click();
                    }}
                }});
            </script>
        </head>
        <body>
            <div class="header">
                <div class="container">
                    <h1>Max/MSP Patch Visualization Report</h1>
                    <p>{main_graph.name}</p>
                    <p>Generated on {current_date}</p>
                </div>
            </div>
            
            <div class="container">
        """
        
        # Add overview section with statistics
        if include_stats and "main" in stats:
            main_stats = stats["main"]
            html_content += f"""
                <div class="section">
                    <h2>Patch Overview</h2>
                    
                    <div class="stats-container">
                        <div class="stat-box">
                            <h4>Total Objects</h4>
                            <p>{main_stats['node_count']}</p>
                        </div>
                        
                        <div class="stat-box">
                            <h4>Connections</h4>
                            <p>{main_stats['edge_count']}</p>
                        </div>
                        
                        <div class="stat-box">
                            <h4>Subpatchers</h4>
                            <p>{len(subpatches)}</p>
                        </div>
                        
                        <div class="stat-box">
                            <h4>Max Hierarchy Level</h4>
                            <p>{main_stats['max_hierarchy_level']}</p>
                        </div>
                    </div>
                    
                    <div class="stats-container">
                        <div class="stat-box">
                            <h4>Most Connected Object</h4>
                            <p>{main_stats.get('most_connected_node', 'N/A')}</p>
                            <small>Connections: {main_stats.get('most_connected_count', 0)}</small>
                        </div>
                        
                        <div class="stat-box">
                            <h4>Most Used Object Type</h4>
                            <p>{main_stats.get('most_common_type', 'N/A')}</p>
                            <small>Count: {main_stats.get('most_common_type_count', 0)}</small>
                        </div>
                    </div>
                </div>
            """
        
        # Start tabs container for main graph and subpatches
        html_content += """
                <div class="section">
                    <h2>Patch Visualizations</h2>
                    
                    <div class="tabs">
                        <div class="tab active" data-tab="tab-main">Main Patch</div>
        """
        
        # Add tabs for subpatches
        for subpatch_id, subpatch in subpatches.items():
            html_content += f"""
                        <div class="tab" data-tab="tab-{subpatch_id}">{subpatch.name}</div>
            """
        
        html_content += """
                    </div>
                    
                    <div class="tab-content active" id="tab-main">
                        <h3>Main Patch: {main_graph.name}</h3>
        """
        
        # Add main graph content
        if include_stats and "main" in stats:
            main_stats = stats["main"]
            html_content += """
                        <h4>Node Distribution by Type</h4>
                        <table>
                            <tr>
                                <th>Type</th>
                                <th>Count</th>
                                <th>Percentage</th>
                            </tr>
            """
            
            for node_type, count in main_stats.get('node_types', {}).items():
                percentage = 0
                if main_stats['node_count'] > 0:
                    percentage = (count / main_stats['node_count']) * 100
                
                html_content += f"""
                            <tr>
                                <td>{node_type}</td>
                                <td>{count}</td>
                                <td>{percentage:.1f}%</td>
                            </tr>
                """
            
            html_content += """
                        </table>
            """
        
        # Add connection matrix if available
        if generate_connection_matrix and "main" in connection_matrices:
            matrix_data = connection_matrices["main"]
            
            html_content += """
                        <h4>Node Connection Matrix</h4>
                        <div class="matrix-container">
                            <table class="connection-matrix">
                                <tr>
                                    <th></th>
            """
            
            # Add column headers
            for node_id in matrix_data["node_ids"]:
                node = main_graph.nodes.get(node_id)
                if node:
                    html_content += f"""
                                    <th title="{node.type}">{node_id}</th>
                    """
            
            html_content += """
                                </tr>
            """
            
            # Add matrix rows
            matrix = matrix_data["matrix"]
            for i, source_id in enumerate(matrix_data["node_ids"]):
                source_node = main_graph.nodes.get(source_id)
                if source_node:
                    html_content += f"""
                                <tr>
                                    <th title="{source_node.type}">{source_id}</th>
                    """
                    
                    for j, target_id in enumerate(matrix_data["node_ids"]):
                        if matrix[i][j] > 0:
                            html_content += f"""
                                    <td class="connected" title="From {source_id} to {target_id}: {matrix[i][j]} connections">{matrix[i][j]}</td>
                            """
                        else:
                            html_content += """
                                    <td></td>
                            """
                    
                    html_content += """
                                </tr>
                    """
            
            html_content += """
                            </table>
                        </div>
            """
        
        # Add main graph node list
        html_content += """
                        <h4>Node List</h4>
                        <table>
                            <tr>
                                <th>ID</th>
                                <th>Type</th>
                                <th>Category</th>
                                <th>Inlets</th>
                                <th>Outlets</th>
                            </tr>
        """
        
        # Add main graph nodes
        for node_id, node in main_graph.nodes.items():
            html_content += f"""
                            <tr>
                                <td>{node.id}</td>
                                <td>{node.type}</td>
                                <td>{node.category or "N/A"}</td>
                                <td>{len(node.inlets)}</td>
                                <td>{len(node.outlets)}</td>
                            </tr>
            """
        
        html_content += """
                        </table>
                        
                        <h4>Connection List</h4>
                        <table>
                            <tr>
                                <th>ID</th>
                                <th>Source</th>
                                <th>Target</th>
                                <th>Type</th>
                            </tr>
        """
        
        # Add main graph edges
        for edge_id, edge in main_graph.edges.items():
            source_node = main_graph.nodes.get(edge.source["node_id"])
            target_node = main_graph.nodes.get(edge.target["node_id"])
            
            source_type = source_node.type if source_node else "Unknown"
            target_type = target_node.type if target_node else "Unknown"
            
            html_content += f"""
                            <tr>
                                <td>{edge.id}</td>
                                <td>{edge.source["node_id"]} ({source_type} [port {edge.source["port"]}])</td>
                                <td>{edge.target["node_id"]} ({target_type} [port {edge.target["port"]}])</td>
                                <td>{edge.type}</td>
                            </tr>
            """
        
        html_content += f"""
                        </table>
                        
                        <h4>Visualization</h4>
                        <img src="{base_filename}_main.png" alt="Main Graph Visualization" class="viz-image">
                    </div>
        """
        
        # Add subpatch tabs content
        for subpatch_id, subpatch in subpatches.items():
            html_content += f"""
                    <div class="tab-content" id="tab-{subpatch_id}">
                        <h3>Subpatch: {subpatch.name} ({subpatch_id})</h3>
            """
            
            # Add subpatch stats if available
            if include_stats and subpatch_id in stats:
                subpatch_stats = stats[subpatch_id]
                
                html_content += """
                        <div class="stats-container">
                            <div class="stat-box">
                                <h4>Total Objects</h4>
                """
                
                html_content += f"""
                                <p>{subpatch_stats['node_count']}</p>
                            </div>
                            
                            <div class="stat-box">
                                <h4>Connections</h4>
                                <p>{subpatch_stats['edge_count']}</p>
                            </div>
                        </div>
                        
                        <h4>Node Distribution by Type</h4>
                        <table>
                            <tr>
                                <th>Type</th>
                                <th>Count</th>
                                <th>Percentage</th>
                            </tr>
                """
                
                for node_type, count in subpatch_stats.get('node_types', {}).items():
                    percentage = 0
                    if subpatch_stats['node_count'] > 0:
                        percentage = (count / subpatch_stats['node_count']) * 100
                    
                    html_content += f"""
                            <tr>
                                <td>{node_type}</td>
                                <td>{count}</td>
                                <td>{percentage:.1f}%</td>
                            </tr>
                    """
                
                html_content += """
                        </table>
                """
            
            # Add connection matrix if available
            if generate_connection_matrix and subpatch_id in connection_matrices:
                matrix_data = connection_matrices[subpatch_id]
                
                if len(matrix_data["node_ids"]) > 0:
                    html_content += """
                            <h4>Node Connection Matrix</h4>
                            <div class="matrix-container">
                                <table class="connection-matrix">
                                    <tr>
                                        <th></th>
                    """
                    
                    # Add column headers
                    for node_id in matrix_data["node_ids"]:
                        node = subpatch.nodes.get(node_id)
                        if node:
                            html_content += f"""
                                        <th title="{node.type}">{node_id}</th>
                            """
                    
                    html_content += """
                                    </tr>
                    """
                    
                    # Add matrix rows
                    matrix = matrix_data["matrix"]
                    for i, source_id in enumerate(matrix_data["node_ids"]):
                        source_node = subpatch.nodes.get(source_id)
                        if source_node:
                            html_content += f"""
                                    <tr>
                                        <th title="{source_node.type}">{source_id}</th>
                            """
                            
                            for j, target_id in enumerate(matrix_data["node_ids"]):
                                if matrix[i][j] > 0:
                                    html_content += f"""
                                        <td class="connected" title="From {source_id} to {target_id}: {matrix[i][j]} connections">{matrix[i][j]}</td>
                                    """
                                else:
                                    html_content += """
                                        <td></td>
                                    """
                            
                            html_content += """
                                    </tr>
                            """
                    
                    html_content += """
                                </table>
                            </div>
                    """
            
            # Add subpatch node list
            html_content += """
                        <h4>Node List</h4>
                        <table>
                            <tr>
                                <th>ID</th>
                                <th>Type</th>
                                <th>Category</th>
                                <th>Inlets</th>
                                <th>Outlets</th>
                            </tr>
            """
            
            # Add subpatch nodes
            for node_id, node in subpatch.nodes.items():
                html_content += f"""
                            <tr>
                                <td>{node.id}</td>
                                <td>{node.type}</td>
                                <td>{node.category or "N/A"}</td>
                                <td>{len(node.inlets)}</td>
                                <td>{len(node.outlets)}</td>
                            </tr>
                """
            
            html_content += """
                        </table>
                        
                        <h4>Connection List</h4>
                        <table>
                            <tr>
                                <th>ID</th>
                                <th>Source</th>
                                <th>Target</th>
                                <th>Type</th>
                            </tr>
            """
            
            # Add subpatch edges
            for edge_id, edge in subpatch.edges.items():
                source_node = subpatch.nodes.get(edge.source["node_id"])
                target_node = subpatch.nodes.get(edge.target["node_id"])
                
                source_type = source_node.type if source_node else "Unknown"
                target_type = target_node.type if target_node else "Unknown"
                
                html_content += f"""
                            <tr>
                                <td>{edge.id}</td>
                                <td>{edge.source["node_id"]} ({source_type} [port {edge.source["port"]}])</td>
                                <td>{edge.target["node_id"]} ({target_type} [port {edge.target["port"]}])</td>
                                <td>{edge.type}</td>
                            </tr>
                """
            
            html_content += f"""
                        </table>
                        
                        <h4>Visualization</h4>
                        <img src="{base_filename}_{subpatch_id}.png" alt="{subpatch.name} Visualization" class="viz-image">
                    </div>
            """
        
        # Close containers and add footer
        html_content += """
                </div>
                
                <div class="footer">
                    Generated using NetworkX and Matplotlib | Max/MSP Patch Visualizer
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write to file
        html_file = os.path.join(output_dir, f"{base_filename}_enhanced_report.html")
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        return html_file
    
    @staticmethod
    def _calculate_graph_stats(graph: PatchGraph) -> Dict[str, Any]:
        """
        Calculate statistics for a graph
        
        Args:
            graph: The PatchGraph to analyze
            
        Returns:
            Dictionary with graph statistics
        """
        stats = {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "max_hierarchy_level": graph.metadata.get("max_hierarchy_level", 0),
            "node_types": {},
            "most_common_type": None,
            "most_common_type_count": 0,
            "most_connected_node": None,
            "most_connected_count": 0
        }
        
        # Node type distribution
        for node_id, node in graph.nodes.items():
            node_type = node.type
            stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1
        
        # Find most common node type
        for node_type, count in stats["node_types"].items():
            if count > stats["most_common_type_count"]:
                stats["most_common_type"] = node_type
                stats["most_common_type_count"] = count
        
        # Find most connected node
        for node_id, node in graph.nodes.items():
            conn_count = len(graph.find_connected_edges(node_id))
            if conn_count > stats["most_connected_count"]:
                stats["most_connected_node"] = node_id
                stats["most_connected_count"] = conn_count
        
        return stats
    
    @staticmethod
    def _generate_connection_matrix(graph: PatchGraph) -> Dict[str, Any]:
        """
        Generate a connection matrix for the graph
        
        Args:
            graph: The PatchGraph to analyze
            
        Returns:
            Dictionary with node IDs and connection matrix
        """
        # Get all node IDs
        node_ids = list(graph.nodes.keys())
        
        # Create empty matrix
        n = len(node_ids)
        matrix = np.zeros((n, n), dtype=int)
        
        # Fill matrix with connections
        for edge_id, edge in graph.edges.items():
            source_id = edge.source["node_id"]
            target_id = edge.target["node_id"]
            
            if source_id in node_ids and target_id in node_ids:
                source_idx = node_ids.index(source_id)
                target_idx = node_ids.index(target_id)
                
                matrix[source_idx][target_idx] += 1
        
        return {
            "node_ids": node_ids,
            "matrix": matrix.tolist()
        }

if __name__ == "__main__":
    import sys
    from patch_graph import PatchGraph
    
    # Check if we have a file to visualize
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "enhanced_patch_visualization.png"
        
        # Load the patch
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Create graph from data
        graph = PatchGraph(data["id"], data["name"])
        
        # Load nodes and edges (simplified)
        for node_data in data.get("nodes", {}).values():
            # In a real implementation, properly load the nodes based on type
            graph.add_node_from_dict(node_data)
        
        for edge_data in data.get("edges", {}).values():
            # In a real implementation, properly load the edges
            graph.add_edge_from_dict(edge_data)
        
        # Visualize
        EnhancedPatchVisualizer.visualize_graph(graph, output_file)
        print(f"Enhanced visualization saved to {output_file}")
        
        # Create HTML report if requested
        if len(sys.argv) > 3 and sys.argv[3] == "--report":
            output_dir = os.path.dirname(output_file) if output_file else "."
            base_filename = os.path.splitext(os.path.basename(output_file))[0]
            
            html_file = EnhancedPatchVisualizer.create_html_report(graph, {}, output_dir, base_filename)
            print(f"HTML report saved to {html_file}")
    else:
        print("Usage: python enhanced_patch_visualizer.py <input_json_file> [output_image_file] [--report]")