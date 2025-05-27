#!/usr/bin/env python3
"""Verify database contents and show statistics."""

import sys
from db_connector import DatabaseConnector
from typing import Dict, List


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def verify_database() -> None:
    """Verify database contents and display statistics."""
    db = DatabaseConnector('scripts/db_settings.ini')
    
    try:
        db.connect()
        print("✅ Database connection successful")
        
        # Check tables
        print_section("Database Tables")
        tables = db.execute_query("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        
        for table in tables:
            table_name = table['table_name']
            count_result = db.execute_query(f"SELECT COUNT(*) as count FROM {table_name}")
            count = count_result[0]['count'] if count_result else 0
            print(f"  - {table_name}: {count:,} rows")
        
        # Connection statistics
        print_section("Connection Statistics")
        
        # Total connections
        total = db.execute_query("SELECT COUNT(*) as count FROM object_connections")
        print(f"Total connections: {total[0]['count']:,}")
        
        # Connections with values
        with_values = db.execute_query("""
            SELECT COUNT(*) as count 
            FROM object_connections 
            WHERE source_value IS NOT NULL OR target_value IS NOT NULL
        """)
        print(f"Connections with values: {with_values[0]['count']:,}")
        
        # File types
        print("\nFile type distribution:")
        file_types = db.execute_query("""
            SELECT file_type, COUNT(*) as count 
            FROM object_connections 
            GROUP BY file_type 
            ORDER BY count DESC
        """)
        for ft in file_types:
            print(f"  - {ft['file_type']}: {ft['count']:,}")
        
        # Top object types
        print_section("Top 20 Object Types")
        top_objects = db.execute_query("""
            SELECT source_object_type as object_type, COUNT(*) as count 
            FROM object_connections 
            GROUP BY source_object_type 
            ORDER BY count DESC 
            LIMIT 20
        """)
        
        for i, obj in enumerate(top_objects, 1):
            print(f"{i:2d}. {obj['object_type']:<20} {obj['count']:>10,} connections")
        
        # Common patterns
        print_section("Top 10 Connection Patterns")
        patterns = db.execute_query("""
            SELECT 
                source_object_type || ' → ' || target_object_type as pattern,
                COUNT(*) as frequency
            FROM object_connections
            GROUP BY source_object_type, target_object_type
            ORDER BY frequency DESC
            LIMIT 10
        """)
        
        for i, pattern in enumerate(patterns, 1):
            print(f"{i:2d}. {pattern['pattern']:<40} {pattern['frequency']:>8,} times")
        
        # Sample connections with values
        print_section("Sample Connections with Values")
        samples = db.execute_query("""
            SELECT source_object_type, source_value, target_object_type, target_value
            FROM object_connections
            WHERE source_value IS NOT NULL 
               AND target_value IS NOT NULL
               AND source_value != ''
               AND target_value != ''
            LIMIT 5
        """)
        
        for i, conn in enumerate(samples, 1):
            print(f"\n{i}. {conn['source_object_type']}('{conn['source_value'][:50]}...')")
            print(f"   → {conn['target_object_type']}('{conn['target_value'][:50]}...')")
        
        # Unique patches
        print_section("Patch Files")
        unique_patches = db.execute_query("""
            SELECT COUNT(DISTINCT patch_file) as count 
            FROM object_connections
        """)
        print(f"Total unique patch files: {unique_patches[0]['count']:,}")
        
        # Sample patch files
        print("\nSample patch files:")
        sample_patches = db.execute_query("""
            SELECT patch_file, COUNT(*) as connections
            FROM object_connections
            GROUP BY patch_file
            ORDER BY connections DESC
            LIMIT 5
        """)
        
        for patch in sample_patches:
            print(f"  - {patch['patch_file']}: {patch['connections']} connections")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        db.disconnect()


if __name__ == "__main__":
    print("MANXO Database Verification Tool")
    print("================================")
    verify_database()
    print("\n✅ Verification complete!")