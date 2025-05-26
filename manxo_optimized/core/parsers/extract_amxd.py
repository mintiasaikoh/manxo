import os
import sys
import json
import struct
import binascii

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
        print("Usage: python extract_amxd.py <path_to_amxd>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    # Extract JSON from amxd
    json_data = extract_json_from_amxd(file_path)
    
    if json_data:
        # Analyze the extracted data
        analysis = analyze_patch(json_data)
        print_analysis(analysis)
        
        # Optionally save the extracted JSON
        output_dir = "/Users/mymac/manxo/analysis"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_extracted.json")
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Extracted JSON saved to: {output_file}")
    else:
        print(f"Failed to extract JSON from {file_path}")
        sys.exit(1)