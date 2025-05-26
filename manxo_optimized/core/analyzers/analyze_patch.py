import os
import json
import sys
from patch_graph import PatchGraph
from node import Node
from edge import Edge

class PatchAnalyzer:
    """Simple utility to analyze Max/MSP patches"""
    
    @staticmethod
    def load_maxpat_or_amxd(file_path):
        """Load a Max patch or Ableton device file"""
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                return data
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                return None
    
    @staticmethod
    def analyze_patch(patch_data):
        """Analyze the patch structure"""
        if not patch_data:
            return None
        
        # Extract basic patch info
        patcher = patch_data.get("patcher", {})
        boxes = patcher.get("boxes", [])
        lines = patcher.get("lines", [])
        
        # Collect node types
        node_types = {}
        for box in boxes:
            # Extract the object type from the box
            box_type = "unknown"
            if "box" in box:
                box_type = "unknown container"  # Special container
            elif "maxclass" in box:
                box_type = box["maxclass"]
            elif "text" in box:
                # For message boxes, the type is in the text
                text = box["text"]
                if isinstance(text, str):
                    words = text.split()
                    if words:
                        box_type = words[0]
            
            # Count this type
            if box_type in node_types:
                node_types[box_type] += 1
            else:
                node_types[box_type] = 1
        
        # Connection types
        connection_types = {}
        for line in lines:
            conn_type = "unknown"
            if "patchline" in line:
                patchline = line["patchline"]
                if "type" in patchline:
                    conn_type = patchline["type"]
                else:
                    conn_type = "control"  # Default
            
            if conn_type in connection_types:
                connection_types[conn_type] += 1
            else:
                connection_types[conn_type] = 1
        
        # Subpatchers
        subpatchers = []
        for i, box in enumerate(boxes):
            if "box" in box or ("maxclass" in box and box["maxclass"] == "newobj" and 
                               "text" in box and isinstance(box["text"], str) and 
                               ("p " in box["text"] or "patcher" in box["text"])):
                subpatchers.append({
                    "index": i,
                    "id": box.get("id", f"box_{i}"),
                    "type": "subpatcher"
                })
            elif "patcher" in box:
                subpatchers.append({
                    "index": i,
                    "id": box.get("id", f"box_{i}"),
                    "type": "embedded patcher"
                })
        
        # Collect results
        result = {
            "file_info": {
                "name": os.path.basename(file_path),
                "path": file_path,
                "type": "maxpat" if file_path.endswith(".maxpat") else "amxd"
            },
            "patch_stats": {
                "node_count": len(boxes),
                "connection_count": len(lines),
                "subpatcher_count": len(subpatchers)
            },
            "node_types": node_types,
            "connection_types": connection_types,
            "subpatchers": subpatchers
        }
        
        return result
    
    @staticmethod
    def print_analysis(analysis):
        """Print analysis results in a readable format"""
        if not analysis:
            print("No analysis data available.")
            return
        
        print("\n===== PATCH ANALYSIS =====")
        print(f"File: {analysis['file_info']['name']} ({analysis['file_info']['type']})")
        print(f"Path: {analysis['file_info']['path']}")
        print("\n--- Stats ---")
        print(f"Total nodes: {analysis['patch_stats']['node_count']}")
        print(f"Total connections: {analysis['patch_stats']['connection_count']}")
        print(f"Subpatchers: {analysis['patch_stats']['subpatcher_count']}")
        
        print("\n--- Node Types ---")
        sorted_nodes = sorted(analysis['node_types'].items(), key=lambda x: x[1], reverse=True)
        for node_type, count in sorted_nodes:
            print(f"  {node_type}: {count}")
        
        print("\n--- Connection Types ---")
        for conn_type, count in analysis['connection_types'].items():
            print(f"  {conn_type}: {count}")
        
        if analysis['subpatchers']:
            print("\n--- Subpatchers ---")
            for sp in analysis['subpatchers']:
                print(f"  {sp['id']} ({sp['type']})")
        
        print("\n===========================\n")

# Main function for standalone usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_patch.py <path_to_maxpat_or_amxd>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    data = PatchAnalyzer.load_maxpat_or_amxd(file_path)
    analysis = PatchAnalyzer.analyze_patch(data)
    PatchAnalyzer.print_analysis(analysis)