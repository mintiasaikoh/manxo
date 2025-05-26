import os
import sys
import json
import struct
from collections import Counter
import re

def extract_json_from_amxd(file_path):
    """
    Extract JSON patcher data from an .amxd file
    
    Args:
        file_path: Path to the .amxd file
        
    Returns:
        Dictionary containing the extracted JSON data, or None if extraction failed
    """
    try:
        with open(file_path, 'rb') as f:
            # Read the file
            data = f.read()
            
            # Look for JSON start (typically after "mx@c")
            json_start = data.find(b'{')
            if json_start == -1:
                print("Could not find JSON start marker")
                return None
            
            # Extract the JSON data
            json_data = data[json_start:]
            
            # Try to decode it
            try:
                # Find the end of the JSON object
                brace_count = 0
                end_pos = 0
                in_string = False
                escape_next = False
                
                for i, byte in enumerate(json_data):
                    char = chr(byte)
                    
                    if escape_next:
                        escape_next = False
                        continue
                    
                    if char == '\\':
                        escape_next = True
                    elif char == '"' and not escape_next:
                        in_string = not in_string
                    elif not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i + 1
                                break
                
                if end_pos > 0:
                    json_data = json_data[:end_pos]
                
                # Decode and parse the JSON
                json_str = json_data.decode('utf-8')
                json_obj = json.loads(json_str)
                return json_obj
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                print(f"Error decoding JSON: {e}")
                
                # Try to find the patcher section directly
                patcher_start = data.find(b'"patcher"')
                if patcher_start > 0:
                    # Find the opening brace after "patcher"
                    brace_start = data.find(b'{', patcher_start)
                    if brace_start > 0:
                        # Extract just the patcher portion
                        patcher_data = data[brace_start:]
                        
                        # Use the same brace counting logic
                        brace_count = 0
                        end_pos = 0
                        in_string = False
                        escape_next = False
                        
                        for i, byte in enumerate(patcher_data):
                            char = chr(byte)
                            
                            if escape_next:
                                escape_next = False
                                continue
                            
                            if char == '\\':
                                escape_next = True
                            elif char == '"' and not escape_next:
                                in_string = not in_string
                            elif not in_string:
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        end_pos = i + 1
                                        break
                        
                        if end_pos > 0:
                            patcher_data = patcher_data[:end_pos]
                        
                        try:
                            patcher_str = b'{"patcher":' + patcher_data + b'}'
                            patcher_obj = json.loads(patcher_str.decode('utf-8'))
                            return patcher_obj
                        except Exception as e2:
                            print(f"Error extracting patcher data: {e2}")
                            return None
                
                return None
                
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def get_object_type(box):
    """Extract the most descriptive object type from a box"""
    # First check if it has a maxclass
    if "maxclass" in box:
        return box["maxclass"]
    
    # Check for text (usually used for object name/type)
    if "text" in box:
        text = box["text"]
        if isinstance(text, str):
            # For Max objects, the first word is typically the object type
            words = text.split()
            if words:
                return words[0]
    
    # Fall back to box class if available
    if "box" in box and isinstance(box["box"], dict):
        box_data = box["box"]
        if "maxclass" in box_data:
            return box_data["maxclass"]
        elif "text" in box_data:
            text = box_data["text"]
            if isinstance(text, str):
                words = text.split()
                if words:
                    return words[0]
    
    # Last resort
    return "unknown"

def get_connection_info(line):
    """Extract connection type and endpoints from a line"""
    connection = {
        "type": "control",  # Default type
        "source": {"id": "unknown", "index": 0},
        "target": {"id": "unknown", "index": 0}
    }
    
    # Extract connection type
    if "patchline" in line and isinstance(line["patchline"], dict):
        patchline = line["patchline"]
        if "type" in patchline:
            connection["type"] = patchline["type"]
    
    # Extract source and target
    if "source" in line and isinstance(line["source"], dict):
        source = line["source"]
        if "id" in source:
            connection["source"]["id"] = source["id"]
        if "index" in source:
            connection["source"]["index"] = source["index"]
    
    if "destination" in line and isinstance(line["destination"], dict):
        dest = line["destination"]
        if "id" in dest:
            connection["target"]["id"] = dest["id"]
        if "index" in dest:
            connection["target"]["index"] = dest["index"]
    
    return connection

def extract_subpatcher(box):
    """Extract a subpatcher from a box if it exists"""
    if "patcher" in box:
        return box["patcher"]
    elif "box" in box and isinstance(box["box"], dict) and "patcher" in box["box"]:
        return box["box"]["patcher"]
    return None

def analyze_patch_structure(patch_data, file_path):
    """Perform detailed analysis of patch structure"""
    if not patch_data:
        return None
    
    # Extract patcher data
    patcher = patch_data.get("patcher", {})
    
    # Basic info
    patch_info = {
        "file_info": {
            "name": os.path.basename(file_path),
            "path": file_path,
            "type": "maxpat" if file_path.endswith(".maxpat") else "amxd"
        },
        "app_version": patcher.get("appversion", {}),
        "presentation_mode": patcher.get("openinpresentation", 0),
        "global_name": patcher.get("globalpatchername", "")
    }
    
    # Analyze boxes (nodes)
    boxes = patcher.get("boxes", [])
    box_types = Counter()
    ui_objects = []
    audio_objects = []
    key_objects = []
    
    for box in boxes:
        box_type = get_object_type(box)
        box_types[box_type] += 1
        
        # Collect object ID
        obj_id = box.get("id", "")
        if not obj_id and "box" in box and isinstance(box["box"], dict):
            obj_id = box["box"].get("id", "")
        
        # Collect text/arguments if available
        text = box.get("text", "")
        if not text and "box" in box and isinstance(box["box"], dict):
            text = box["box"].get("text", "")
        
        # Identify different object categories
        if box_type in ["live.dial", "live.slider", "live.numbox", "live.toggle", 
                       "live.menu", "flonum", "slider", "toggle", "bpatcher"]:
            ui_objects.append({
                "id": obj_id,
                "type": box_type,
                "text": text
            })
        
        elif box_type.endswith("~") or box_type in ["adc~", "dac~", "gain~", "plugin~", 
                                                  "plugout~", "vst~", "sfplay~"]:
            audio_objects.append({
                "id": obj_id,
                "type": box_type,
                "text": text
            })
        
        # Extract key objects
        if box_type in ["plugout~", "plugin~", "vst~", "pfft~", "gen~", "jit.gen", 
                       "loadbang", "pattrstorage", "live.object", "p"]:
            key_objects.append({
                "id": obj_id,
                "type": box_type,
                "text": text
            })
    
    # Analyze lines (connections)
    lines = patcher.get("lines", [])
    connection_types = Counter()
    
    for line in lines:
        connection = get_connection_info(line)
        connection_types[connection["type"]] += 1
    
    # Analyze subpatchers
    subpatchers = []
    for i, box in enumerate(boxes):
        sp = extract_subpatcher(box)
        if sp:
            # Get basic subpatcher info
            sp_name = sp.get("name", f"subpatcher_{i}")
            sp_boxes = sp.get("boxes", [])
            sp_lines = sp.get("lines", [])
            
            subpatchers.append({
                "id": f"subpatcher_{i}",
                "name": sp_name,
                "node_count": len(sp_boxes),
                "connection_count": len(sp_lines)
            })
    
    # Assemble results
    result = {
        "file_info": patch_info["file_info"],
        "patch_info": {
            "app_version": patch_info["app_version"],
            "presentation_mode": patch_info["presentation_mode"],
            "global_name": patch_info["global_name"]
        },
        "structure": {
            "node_count": len(boxes),
            "connection_count": len(lines),
            "subpatcher_count": len(subpatchers),
            "node_types": dict(box_types),
            "connection_types": dict(connection_types)
        },
        "objects": {
            "ui_objects": ui_objects,
            "audio_objects": audio_objects,
            "key_objects": key_objects
        },
        "subpatchers": subpatchers
    }
    
    return result

def print_detailed_analysis(analysis):
    """Print detailed analysis in a readable format"""
    if not analysis:
        print("No analysis data available.")
        return
    
    file_info = analysis["file_info"]
    patch_info = analysis["patch_info"]
    structure = analysis["structure"]
    objects = analysis["objects"]
    
    print("\n===== DETAILED PATCH ANALYSIS =====")
    print(f"File: {file_info['name']} ({file_info['type']})")
    print(f"Path: {file_info['path']}")
    
    # App version
    app_version = patch_info["app_version"]
    if app_version:
        print(f"\nCreated with Max version: {app_version.get('major', '?')}.{app_version.get('minor', '?')}.{app_version.get('revision', '?')}")
    
    print(f"Presentation mode: {'Enabled' if patch_info['presentation_mode'] else 'Disabled'}")
    if patch_info['global_name']:
        print(f"Global patcher name: {patch_info['global_name']}")
    
    # Structure stats
    print("\n--- Structure ---")
    print(f"Total nodes: {structure['node_count']}")
    print(f"Total connections: {structure['connection_count']}")
    print(f"Subpatchers: {structure['subpatcher_count']}")
    
    # Node types
    print("\n--- Top Node Types ---")
    sorted_nodes = sorted(structure['node_types'].items(), key=lambda x: x[1], reverse=True)
    for i, (node_type, count) in enumerate(sorted_nodes[:15]):  # Show top 15
        print(f"  {node_type}: {count}")
    if len(sorted_nodes) > 15:
        print(f"  ... and {len(sorted_nodes) - 15} more types")
    
    # Connection types
    print("\n--- Connection Types ---")
    for conn_type, count in structure['connection_types'].items():
        print(f"  {conn_type}: {count}")
    
    # UI objects
    if objects['ui_objects']:
        print("\n--- UI Objects ---")
        for ui in objects['ui_objects'][:10]:  # Show first 10
            print(f"  {ui['type']} (ID: {ui['id']}): {ui['text']}")
        if len(objects['ui_objects']) > 10:
            print(f"  ... and {len(objects['ui_objects']) - 10} more UI objects")
    
    # Audio objects
    if objects['audio_objects']:
        print("\n--- Audio Objects ---")
        for audio in objects['audio_objects'][:10]:  # Show first 10
            print(f"  {audio['type']} (ID: {audio['id']}): {audio['text']}")
        if len(objects['audio_objects']) > 10:
            print(f"  ... and {len(objects['audio_objects']) - 10} more audio objects")
    
    # Key objects
    if objects['key_objects']:
        print("\n--- Key Objects ---")
        for key in objects['key_objects']:
            print(f"  {key['type']} (ID: {key['id']}): {key['text']}")
    
    # Subpatchers
    if analysis['subpatchers']:
        print("\n--- Subpatchers ---")
        for i, sp in enumerate(analysis['subpatchers'][:10]):  # Show first 10
            print(f"  {sp['name']} ({sp['id']}): {sp['node_count']} nodes, {sp['connection_count']} connections")
        if len(analysis['subpatchers']) > 10:
            print(f"  ... and {len(analysis['subpatchers']) - 10} more subpatchers")
    
    print("\n====================================\n")

# Main function
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python amxd_analyzer.py <path_to_maxpat_or_amxd>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    # Extract and analyze
    json_data = extract_json_from_amxd(file_path)
    if json_data:
        # Analyze in detail
        analysis = analyze_patch_structure(json_data, file_path)
        print_detailed_analysis(analysis)
        
        # Save original data and analysis
        if len(sys.argv) > 2 and sys.argv[2] == "--save":
            output_dir = "/Users/mymac/manxo/analysis"
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Save extracted JSON
            json_output = os.path.join(output_dir, f"{base_name}_extracted.json")
            with open(json_output, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            # Save analysis
            analysis_output = os.path.join(output_dir, f"{base_name}_analysis.json")
            with open(analysis_output, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            print(f"Extracted JSON saved to: {json_output}")
            print(f"Analysis saved to: {analysis_output}")
    else:
        print(f"Failed to extract data from {file_path}")
        sys.exit(1)