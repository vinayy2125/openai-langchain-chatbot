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
    
    This is the centralized method for all database connections in the application.
    It includes proper error handling and connection validation.
    
    Returns:
        psycopg2.connection: A PostgreSQL database connection
        
    Raises:
        ConnectionError: If database connection fails
        ValueError: If required environment variables are missing
    """
    # Validate required environment variables
    missing_vars = []
    if not DB_NAME:
        missing_vars.append("DB_NAME")
    if not DB_USER:
        missing_vars.append("DB_USER")
    if not DB_PASSWORD:
        missing_vars.append("DB_PASSWORD")
    
    if missing_vars:
        raise ValueError(f"Missing required database environment variables: {', '.join(missing_vars)}")
    
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER, 
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            options="-c client_encoding=UTF8"
        )
        logger.debug("Database connection established successfully")
        return conn
        
    except psycopg2.Error as e:
        logger.error(f"Database connection failed: {e}")
        raise ConnectionError(f"Failed to connect to database: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during database connection: {e}")
        raise ConnectionError(f"Unexpected database connection error: {e}") from e


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


class DatabaseConnection:
    """
    Context manager for database connections with automatic cleanup.
    
    Usage:
        with DatabaseConnection() as (conn, cursor):
            cursor.execute("SELECT * FROM users")
            # Connection and cursor automatically closed
    """
    
    def __init__(self):
        self.conn = None
        self.cursor = None
    
    def __enter__(self):
        self.conn = get_db_conn()
        self.cursor = self.conn.cursor()
        return self.conn, self.cursor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            # If there was an exception, rollback the transaction
            if self.conn:
                try:
                    self.conn.rollback()
                    logger.warning("Transaction rolled back due to exception")
                except Exception as e:
                    logger.error(f"Error during rollback: {e}")
        else:
            # If no exception, commit the transaction
            if self.conn:
                try:
                    self.conn.commit()
                except Exception as e:
                    logger.error(f"Error during commit: {e}")
        
        # Close cursor and connection
        if self.cursor:
            try:
                self.cursor.close()
            except Exception as e:
                logger.error(f"Error closing cursor: {e}")
        
        if self.conn:
            try:
                self.conn.close()
                logger.debug("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing connection: {e}")


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
