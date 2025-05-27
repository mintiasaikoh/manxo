"""Database connector for MANXO project."""

import psycopg2
import configparser
from typing import Dict, List, Any, Optional

class DatabaseConnector:
    """PostgreSQL database connector for Max/MSP analysis data."""
    
    def __init__(self, config_file: str):
        """Initialize database connector with config file."""
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.connection = None
        self.cursor = None
    
    def connect(self):
        """Establish database connection."""
        try:
            self.connection = psycopg2.connect(
                host=self.config['database']['host'],
                port=self.config['database']['port'],
                database=self.config['database']['database'],
                user=self.config['database']['user'],
                password=self.config['database']['password']
            )
            self.cursor = self.connection.cursor()
            print(f"Connected to database: {self.config['database']['database']}")
        except Exception as e:
            print(f"Database connection error: {e}")
            raise
    
    def disconnect(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            print("Database connection closed")
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute SELECT query and return results as list of dicts."""
        if not self.cursor:
            raise Exception("No database connection")
        
        self.cursor.execute(query, params)
        columns = [desc[0] for desc in self.cursor.description]
        results = []
        for row in self.cursor.fetchall():
            results.append(dict(zip(columns, row)))
        return results
    
    def execute_update(self, query: str, params: Optional[tuple] = None):
        """Execute INSERT/UPDATE/DELETE query."""
        if not self.cursor:
            raise Exception("No database connection")
        
        self.cursor.execute(query, params)
        self.connection.commit()
        return self.cursor.rowcount

# Quick test if run directly
if __name__ == "__main__":
    db = DatabaseConnector('db_settings.ini')
    db.connect()
    
    # Test query
    try:
        result = db.execute_query("SELECT COUNT(*) as count FROM object_connections")
        print(f"Total connections: {result[0]['count']}")
    except Exception as e:
        print(f"Query error: {e}")
    
    db.disconnect()