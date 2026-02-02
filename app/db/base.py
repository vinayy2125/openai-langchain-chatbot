"""
Centralized database connection module.

This module provides a unified interface for database connections across the entire application.
All database operations should use get_db_conn() to ensure consistent connection handling.
"""
import psycopg2
import os
from dotenv import load_dotenv
from typing import Optional
import logging

# Load environment variables
load_dotenv()

# Database configuration
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")  # Default to localhost
DB_PORT = os.getenv("DB_PORT", "5432")      # Default PostgreSQL port

# Setup logger for this module
logger = logging.getLogger(__name__)


def get_db_conn():
    """
    Get a connection to the PostgreSQL database.
    
    DEPRECATED: This function now uses connection pooling for better performance.
    Consider using PooledDatabaseConnection context manager instead.
    
    WARNING: Connections from this function MUST be explicitly closed by the caller.
    The connection will NOT be automatically returned to the pool.
    
    Returns:
        psycopg2.connection: A PostgreSQL database connection from the pool
        
    Raises:
        ConnectionError: If database connection fails
        ValueError: If required environment variables are missing
    """
    try:
        from app.db.pool import get_pooled_connection
        conn = get_pooled_connection()
        logger.debug("Database connection acquired from pool")
        return conn
    except Exception as e:
        logger.error(f"Failed to get pooled connection: {e}")
        raise ConnectionError(f"Failed to get database connection: {e}") from e


def get_db_connection_info() -> dict:
    """
    Get database connection information (without sensitive data).
    
    Returns:
        dict: Database connection details for debugging/monitoring
    """
    return {
        "host": DB_HOST,
        "port": DB_PORT,
        "database": DB_NAME,
        "user": DB_USER,
        "has_password": bool(DB_PASSWORD)
    }


def return_db_conn(conn):
    """
    Return a connection to the pool.
    
    This should be called after using get_db_conn() to ensure the connection
    is properly returned to the pool instead of being closed.
    
    Args:
        conn: The connection to return to the pool
    """
    try:
        from app.db.pool import return_pooled_connection
        return_pooled_connection(conn)
        logger.debug("Database connection returned to pool")
    except Exception as e:
        logger.error(f"Failed to return connection to pool: {e}")



class DatabaseConnection:
    """
    Context manager for database connections with automatic cleanup.
    
    Now uses connection pooling for better performance.
    
    Usage:
        with DatabaseConnection() as (conn, cursor):
            cursor.execute("SELECT * FROM users")
            # Connection automatically returned to pool
    """
    
    def __init__(self):
        """Initialize using the pooled connection manager."""
        from app.db.pool import PooledDatabaseConnection
        self._pooled_context = PooledDatabaseConnection()
    
    def __enter__(self):
        """Acquire connection from pool."""
        return self._pooled_context.__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Return connection to pool."""
        return self._pooled_context.__exit__(exc_type, exc_val, exc_tb)


def test_database_connection() -> bool:
    """
    Test the database connection and return True if successful.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        conn = get_db_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        logger.info("Database connection test successful")
        return result is not None and result[0] == 1
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False
