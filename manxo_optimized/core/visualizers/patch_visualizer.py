import os
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import networkx as nx
from patch_graph import PatchGraph

class PatchVisualizer:
    """Utility class for visualizing patch graphs"""
    
    # Color schemes for different node types
    NODE_COLORS = {
        "object": "#1f77b4",    # Blue
        "message": "#ff7f0e",   # Orange
        "comment": "#2ca02c",   # Green
        "patcher": "#d62728",   # Red
        "audio": "#9467bd",     # Purple
        "visual": "#8c564b",    # Brown
        "ui": "#e377c2",        # Pink
        "generic": "#7f7f7f"    # Gray
    }
    
    # Color schemes for different edge types
    EDGE_COLORS = {
        "control": "black",
        "signal": "blue",
        "matrix": "purple",
        "virtual": "gray"
    }
    
    @staticmethod
    def visualize_graph(graph: PatchGraph, output_file: str = None, 
                        show_ports: bool = True, show_labels: bool = True) -> None:
        """
        Visualize a patch graph using matplotlib
        
        Args:
            graph: The PatchGraph to visualize
            output_file: Optional file path to save the visualization
            show_ports: Whether to show the inlet/outlet ports
            show_labels: Whether to show node labels
        """
        # Create figure
        plt.figure(figsize=(12, 8))
        
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
            else:
                box_type = "generic"
                
            # Get node label
            if hasattr(node, 'display_name'):
                label = node.display_name
            else:
                label = node.type
                
            # Add to graph
            G.add_node(node_id, position=pos, box_type=box_type, label=label)
        
        # Add edges
        for edge_id, edge in graph.edges.items():
            source = edge.source["node_id"]
            target = edge.target["node_id"]
            edge_type = edge.type
            
            # Add to graph
            G.add_edge(source, target, id=edge_id, type=edge_type)
        
        # Get positions, either from nodes or calculate using spring layout
        pos = {}
        for node_id, node_data in G.nodes(data=True):
            if node_data["position"] is not None:
                pos[node_id] = node_data["position"]
        
        # If we don't have positions for all nodes, use spring layout with fixed positions
        if len(pos) < len(G.nodes):
            # Create a copy of G to avoid modifying the original
            G_layout = G.copy()
            
            # Remove nodes that have fixed positions
            for node_id in pos:
                G_layout.remove_node(node_id)
                
            # Calculate positions for remaining nodes
            pos_calculated = nx.spring_layout(G_layout, k=0.5, iterations=50)
            
            # Combine positions
            pos.update(pos_calculated)
        
        # Draw nodes
        for node_id, node_data in G.nodes(data=True):
            box_type = node_data["box_type"]
            color = PatchVisualizer.NODE_COLORS.get(box_type, PatchVisualizer.NODE_COLORS["generic"])
            
            # Create a rectangular patch for the node
            x, y = pos[node_id]
            width, height = 0.08, 0.05
            
            # Create rectangle
            rect = patches.Rectangle((x - width/2, y - height/2), width, height, 
                                    linewidth=1, edgecolor='black', facecolor=color, alpha=0.7)
            plt.gca().add_patch(rect)
            
            # Add label
            if show_labels:
                plt.text(x, y, node_data["label"], horizontalalignment='center',
                        verticalalignment='center', fontsize=8)
            
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
                        inlet_rect = patches.Rectangle(
                            (inlet_x - port_width/2, inlet_y - port_width/2),
                            port_width, port_width,
                            linewidth=1, edgecolor='black', facecolor='white'
                        )
                        plt.gca().add_patch(inlet_rect)
                
                # Draw outlets on bottom
                num_outlets = len(node.outlets)
                if num_outlets > 0:
                    outlet_spacing = width / (num_outlets + 1)
                    for i in range(num_outlets):
                        outlet_x = x - width/2 + (i + 1) * outlet_spacing
                        outlet_y = y - height/2
                        outlet_rect = patches.Rectangle(
                            (outlet_x - port_width/2, outlet_y - port_width/2),
                            port_width, port_width,
                            linewidth=1, edgecolor='black', facecolor='white'
                        )
                        plt.gca().add_patch(outlet_rect)
        
        # Draw edges
        for source, target, edge_data in G.edges(data=True):
            edge_type = edge_data.get("type", "control")
            color = PatchVisualizer.EDGE_COLORS.get(edge_type, "black")
            linestyle = "--" if edge_type == "virtual" else "-"
            
            # Get source node attributes
            source_node = graph.nodes[source]
            source_port = None
            
            # Find the matching edge to get the source port
            for edge_id, edge in graph.edges.items():
                if edge.source["node_id"] == source and edge.target["node_id"] == target:
                    source_port = edge.source["port"]
                    target_port = edge.target["port"]
                    break
            
            if source_port is not None and show_ports:
                # Calculate port positions
                source_x, source_y = pos[source]
                target_x, target_y = pos[target]
                width, height = 0.08, 0.05
                port_width = width / 8
                
                # Source outlet position
                num_outlets = len(source_node.outlets)
                if num_outlets > 0:
                    outlet_spacing = width / (num_outlets + 1)
                    outlet_x = source_x - width/2 + (source_port + 1) * outlet_spacing
                    outlet_y = source_y - height/2
                else:
                    outlet_x, outlet_y = source_x, source_y - height/2
                
                # Target inlet position
                target_node = graph.nodes[target]
                num_inlets = len(target_node.inlets)
                if num_inlets > 0:
                    inlet_spacing = width / (num_inlets + 1)
                    inlet_x = target_x - width/2 + (target_port + 1) * inlet_spacing
                    inlet_y = target_y + height/2
                else:
                    inlet_x, inlet_y = target_x, target_y + height/2
                
                # Draw the edge
                plt.plot([outlet_x, inlet_x], [outlet_y, inlet_y], 
                        color=color, linestyle=linestyle, linewidth=1.5)
            else:
                # Draw a simple edge from node center to node center
                plt.plot([pos[source][0], pos[target][0]], 
                        [pos[source][1], pos[target][1]], 
                        color=color, linestyle=linestyle, linewidth=1.5)
        
        # Set up the plot
        plt.title(f"Patch Graph: {graph.name}")
        plt.axis('off')
        
        # Save or show
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def visualize_full_patch(main_graph: PatchGraph, subpatches: Dict[str, PatchGraph], 
                            output_dir: str, base_filename: str) -> None:
        """
        Visualize a complete patch with all its subpatches
        
        Args:
            main_graph: The main PatchGraph
            subpatches: Dictionary of subpatches
            output_dir: Directory to save the visualizations
            base_filename: Base name for the output files
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Visualize main graph
        main_output = os.path.join(output_dir, f"{base_filename}_main.png")
        PatchVisualizer.visualize_graph(main_graph, main_output)
        
        # Visualize each subpatch
        for subpatch_id, subpatch in subpatches.items():
            subpatch_output = os.path.join(output_dir, f"{base_filename}_{subpatch_id}.png")
            PatchVisualizer.visualize_graph(subpatch, subpatch_output)
            
    @staticmethod
    def create_html_report(main_graph: PatchGraph, subpatches: Dict[str, PatchGraph],
                          output_dir: str, base_filename: str) -> str:
        """
        Create an HTML report for the patch with all visualizations
        
        Args:
            main_graph: The main PatchGraph
            subpatches: Dictionary of subpatches
            output_dir: Directory to save the report
            base_filename: Base name for the output files
            
        Returns:
            Path to the HTML report file
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate visualizations
        PatchVisualizer.visualize_full_patch(main_graph, subpatches, output_dir, base_filename)
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Patch Report: {main_graph.name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .graph-container {{ margin-bottom: 40px; }}
                .node-list {{ margin-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .viz-image {{ max-width: 100%; border: 1px solid #ddd; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <h1>Patch Graph Report: {main_graph.name}</h1>
            
            <div class="graph-container">
                <h2>Main Graph</h2>
                
                <div class="node-list">
                    <h3>Nodes ({len(main_graph.nodes)})</h3>
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
                </div>
                
                <div class="edge-list">
                    <h3>Connections</h3>
                    <table>
                        <tr>
                            <th>ID</th>
                            <th>Source Node</th>
                            <th>Source Port</th>
                            <th>Target Node</th>
                            <th>Target Port</th>
                            <th>Type</th>
                        </tr>
        """
        
        # Add main graph edges
        for edge_id, edge in main_graph.edges.items():
            html_content += f"""
                        <tr>
                            <td>{edge.id}</td>
                            <td>{edge.source["node_id"]}</td>
                            <td>{edge.source["port"]}</td>
                            <td>{edge.target["node_id"]}</td>
                            <td>{edge.target["port"]}</td>
                            <td>{edge.type}</td>
                        </tr>
            """
        
        html_content += f"""
                    </table>
                </div>
                
                <div class="visualization">
                    <h3>Visualization</h3>
                    <img src="{base_filename}_main.png" alt="Main Graph Visualization" class="viz-image">
                </div>
            </div>
        """
        
        # Add subpatches
        for subpatch_id, subpatch in subpatches.items():
            html_content += f"""
            <div class="graph-container">
                <h2>Subpatch: {subpatch.name} ({subpatch_id})</h2>
                
                <div class="node-list">
                    <h3>Nodes ({len(subpatch.nodes)})</h3>
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
                </div>
                
                <div class="edge-list">
                    <h3>Connections</h3>
                    <table>
                        <tr>
                            <th>ID</th>
                            <th>Source Node</th>
                            <th>Source Port</th>
                            <th>Target Node</th>
                            <th>Target Port</th>
                            <th>Type</th>
                        </tr>
            """
            
            # Add subpatch edges
            for edge_id, edge in subpatch.edges.items():
                html_content += f"""
                        <tr>
                            <td>{edge.id}</td>
                            <td>{edge.source["node_id"]}</td>
                            <td>{edge.source["port"]}</td>
                            <td>{edge.target["node_id"]}</td>
                            <td>{edge.target["port"]}</td>
                            <td>{edge.type}</td>
                        </tr>
                """
            
            html_content += f"""
                    </table>
                </div>
                
                <div class="visualization">
                    <h3>Visualization</h3>
                    <img src="{base_filename}_{subpatch_id}.png" alt="{subpatch.name} Visualization" class="viz-image">
                </div>
            </div>
            """
        
        # Close HTML
        html_content += """
        </body>
        </html>
        """
        
        # Write to file
        html_file = os.path.join(output_dir, f"{base_filename}_report.html")
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        return html_file

if __name__ == "__main__":
    import sys
    from patch_graph import PatchGraph
    
    # Check if we have a file to visualize
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "patch_visualization.png"
        
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
        PatchVisualizer.visualize_graph(graph, output_file)
        print(f"Visualization saved to {output_file}")
    else:
        print("Usage: python patch_visualizer.py <input_json_file> [output_image_file]")