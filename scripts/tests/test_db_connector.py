"""Tests for database connector."""

import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_connector import DatabaseConnector


class TestDatabaseConnector:
    """Test cases for DatabaseConnector class."""
    
    @patch('psycopg2.connect')
    def test_connect_success(self, mock_connect):
        """Test successful database connection."""
        # Setup mock
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection
        
        # Test connection
        db = DatabaseConnector('db_settings.ini')
        db.connect()
        
        # Verify
        assert db.connection is not None
        assert db.cursor is not None
        mock_connect.assert_called_once()
    
    @patch('psycopg2.connect')
    def test_connect_failure(self, mock_connect):
        """Test database connection failure."""
        # Setup mock to raise exception
        mock_connect.side_effect = Exception("Connection failed")
        
        # Test connection
        db = DatabaseConnector('db_settings.ini')
        
        with pytest.raises(Exception) as exc_info:
            db.connect()
        
        assert "Connection failed" in str(exc_info.value)
    
    def test_execute_query_without_connection(self):
        """Test query execution without connection."""
        db = DatabaseConnector('db_settings.ini')
        
        with pytest.raises(Exception) as exc_info:
            db.execute_query("SELECT 1")
        
        assert "No database connection" in str(exc_info.value)
    
    @patch('psycopg2.connect')
    def test_execute_query_success(self, mock_connect):
        """Test successful query execution."""
        # Setup mock
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.description = [('count',)]
        mock_cursor.fetchall.return_value = [(100,)]
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection
        
        # Test query
        db = DatabaseConnector('db_settings.ini')
        db.connect()
        results = db.execute_query("SELECT COUNT(*) as count FROM test")
        
        # Verify
        assert len(results) == 1
        assert results[0]['count'] == 100
        mock_cursor.execute.assert_called_once_with("SELECT COUNT(*) as count FROM test", None)
    
    @patch('psycopg2.connect')
    def test_disconnect(self, mock_connect):
        """Test database disconnection."""
        # Setup mock
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection
        
        # Test disconnect
        db = DatabaseConnector('db_settings.ini')
        db.connect()
        db.disconnect()
        
        # Verify
        mock_cursor.close.assert_called_once()
        mock_connection.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])