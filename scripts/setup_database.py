#!/usr/bin/env python3
"""Setup MANXO database with initial schema."""

import os
import sys
import psycopg2
from pathlib import Path
from db_connector import DatabaseConnector

def setup_database():
    """Create database and tables for MANXO."""
    
    # Read SQL file
    sql_file = Path(__file__).parent / 'create_tables.sql'
    if not sql_file.exists():
        print(f"Error: {sql_file} not found")
        sys.exit(1)
    
    with open(sql_file, 'r') as f:
        sql_commands = f.read()
    
    # Connect to database
    db = DatabaseConnector('scripts/db_settings.ini')
    
    try:
        db.connect()
        print("Connected to database")
        
        # Execute SQL commands
        db.cursor.execute(sql_commands)
        db.connection.commit()
        print("Database schema created successfully")
        
        # Verify tables
        db.cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        
        tables = db.cursor.fetchall()
        print("\nCreated tables:")
        for table in tables:
            print(f"  - {table[0]}")
        
        # Check for existing data
        db.cursor.execute("SELECT COUNT(*) FROM object_connections")
        count = db.cursor.fetchone()[0]
        if count > 0:
            print(f"\nFound {count:,} existing connections in database")
        else:
            print("\nDatabase is empty. Run analyze_patch_connections.py to populate it.")
            
    except Exception as e:
        print(f"Error setting up database: {e}")
        sys.exit(1)
    finally:
        db.disconnect()

if __name__ == "__main__":
    setup_database()