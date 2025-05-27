"""Analyze Max/MSP patch connections and save to database."""

import json
import sys
from pathlib import Path
from db_connector import DatabaseConnector

def analyze_patch(patch_file: str):
    """Analyze a single patch file and store connections in database."""
    
    # Load patch file
    with open(patch_file, 'r') as f:
        patch_data = json.load(f)
    
    # Connect to database
    db = DatabaseConnector('scripts/db_settings.ini')
    db.connect()
    
    # Extract patcher
    patcher = patch_data.get('patcher', {})
    boxes = patcher.get('boxes', [])
    lines = patcher.get('lines', [])
    
    print(f"Analyzing {patch_file}")
    print(f"Found {len(boxes)} objects and {len(lines)} connections")
    
    # Process each connection
    for line in lines:
        source = line.get('source', [None, None])
        dest = line.get('destination', [None, None])
        
        # Find source and target objects
        source_obj = next((b for b in boxes if b.get('box', {}).get('id') == source[0]), None)
        target_obj = next((b for b in boxes if b.get('box', {}).get('id') == dest[0]), None)
        
        if source_obj and target_obj:
            source_box = source_obj.get('box', {})
            target_box = target_obj.get('box', {})
            
            # Extract object types and values
            source_type = source_box.get('maxclass', 'unknown')
            target_type = target_box.get('maxclass', 'unknown')
            
            # Extract values from text or other attributes
            source_value = None
            target_value = None
            
            if source_type in ['message', 'flonum', 'number', 'comment']:
                source_value = source_box.get('text', '')
            elif source_type == 'newobj':
                text = source_box.get('text', '')
                source_type = text.split()[0] if text else 'newobj'
                source_value = text
            
            if target_type in ['message', 'flonum', 'number', 'comment']:
                target_value = target_box.get('text', '')
            elif target_type == 'newobj':
                text = target_box.get('text', '')
                target_type = text.split()[0] if text else 'newobj'
                target_value = text
            
            # Insert into database
            query = """
            INSERT INTO object_connections 
            (source_object_type, source_port, target_object_type, target_port, 
             patch_file, file_type, source_value, target_value)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source_object_type, source_port, target_object_type, 
                        target_port, patch_file) 
            DO UPDATE SET 
                source_value = EXCLUDED.source_value,
                target_value = EXCLUDED.target_value
            """
            
            file_type = 'amxd' if patch_file.endswith('.amxd') else 'maxpat'
            params = (
                source_type, source[1] or 0, 
                target_type, dest[1] or 0,
                Path(patch_file).name, file_type,
                source_value, target_value
            )
            
            db.execute_update(query, params)
    
    db.disconnect()
    print(f"Analysis complete for {patch_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_patch_connections.py <patch_file>")
        sys.exit(1)
    
    analyze_patch(sys.argv[1])