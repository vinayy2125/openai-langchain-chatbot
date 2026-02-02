"""
Database connection pool manager for improved performance.

This module provides a thread-safe connection pool to reduce the overhead
of creating new database connections for every request.
"""
import psycopg2
from psycopg2 import pool
import os
from dotenv import load_dotenv
from typing import Optional
import logging
import threading
import atexit

# Load environment variables
load_dotenv()

# Database configuration
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# Pool configuration
MIN_CONNECTIONS = int(os.getenv("DB_POOL_MIN", "2"))
MAX_CONNECTIONS = int(os.getenv("DB_POOL_MAX", "20"))

# Setup logger
logger = logging.getLogger(__name__)

# Global connection pool instance
_connection_pool: Optional[pool.ThreadedConnectionPool] = None
_pool_lock = threading.Lock()


class ConnectionPoolManager:
    """
    Manages a thread-safe connection pool for PostgreSQL.
    
    This class implements the singleton pattern to ensure only one pool
    exists throughout the application lifecycle.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Ensure only one instance of the pool manager exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the connection pool if not already initialized."""
        if self._initialized:
            return
            
        with _pool_lock:
            if self._initialized:
                return
                
            # Validate required environment variables
            missing_vars = []
            if not DB_NAME:
                missing_vars.append("DB_NAME")
            if not DB_USER:
                missing_vars.append("DB_USER")
            if not DB_PASSWORD:
                missing_vars.append("DB_PASSWORD")
            
            if missing_vars:
                raise ValueError(
                    f"Missing required database environment variables: {', '.join(missing_vars)}"
                )
            
            try:
                self.pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=MIN_CONNECTIONS,
                    maxconn=MAX_CONNECTIONS,
                    dbname=DB_NAME,
                    user=DB_USER,
                    password=DB_PASSWORD,
                    host=DB_HOST,
                    port=DB_PORT,
                    options="-c client_encoding=UTF8"
                )
                logger.info(
                    f"Database connection pool initialized: "
                    f"min={MIN_CONNECTIONS}, max={MAX_CONNECTIONS}"
                )
                self._initialized = True
                
                # Register cleanup on exit
                atexit.register(self.close_all_connections)
                
            except psycopg2.Error as e:
                logger.error(f"Failed to create connection pool: {e}")
                raise ConnectionError(f"Failed to create connection pool: {e}") from e
    
    def get_connection(self):
        """
        Get a connection from the pool.
        
        Returns:
            psycopg2.connection: A database connection from the pool
            
        Raises:
            ConnectionError: If unable to get a connection from the pool
        """
        if not self._initialized or not self.pool:
            raise ConnectionError("Connection pool not initialized")
        
        try:
            conn = self.pool.getconn()
            if conn:
                logger.debug("Connection acquired from pool")
                return conn
            else:
                raise ConnectionError("Failed to get connection from pool")
        except psycopg2.pool.PoolError as e:
            logger.error(f"Pool error when getting connection: {e}")
            raise ConnectionError(f"Pool error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error getting connection: {e}")
            raise ConnectionError(f"Unexpected error: {e}") from e
    
    def return_connection(self, conn):
        """
        Return a connection to the pool.
        
        Args:
            conn: The connection to return to the pool
        """
        if not self._initialized or not self.pool:
            logger.warning("Attempted to return connection to uninitialized pool")
            return
        
        try:
            self.pool.putconn(conn)
            logger.debug("Connection returned to pool")
        except Exception as e:
            logger.error(f"Error returning connection to pool: {e}")
    
    def close_all_connections(self):
        """Close all connections in the pool."""
        if self._initialized and self.pool:
            try:
                self.pool.closeall()
                logger.info("All database connections closed")
            except Exception as e:
                logger.error(f"Error closing connection pool: {e}")
    
    def get_pool_status(self) -> dict:
        """
        Get the current status of the connection pool.
        
        Returns:
            dict: Pool status information
        """
        if not self._initialized or not self.pool:
            return {
                "initialized": False,
                "min_connections": MIN_CONNECTIONS,
                "max_connections": MAX_CONNECTIONS,
            }
        
        # Note: ThreadedConnectionPool doesn't expose current connection count
        # This is a limitation of psycopg2's pool implementation
        return {
            "initialized": True,
            "min_connections": MIN_CONNECTIONS,
            "max_connections": MAX_CONNECTIONS,
            "host": DB_HOST,
            "port": DB_PORT,
            "database": DB_NAME,
        }


class PooledDatabaseConnection:
    """
    Context manager for pooled database connections with automatic cleanup.
    
    Usage:
        with PooledDatabaseConnection() as (conn, cursor):
            cursor.execute("SELECT * FROM users")
            # Connection automatically returned to pool
    """
    
    def __init__(self, auto_commit: bool = True):
        """
        Initialize the context manager.
        
        Args:
            auto_commit: Whether to auto-commit on successful exit (default: True)
        """
        self.conn = None
        self.cursor = None
        self.pool_manager = ConnectionPoolManager()
        self.auto_commit = auto_commit
    
    def __enter__(self):
        """Acquire connection from pool and create cursor."""
        self.conn = self.pool_manager.get_connection()
        self.cursor = self.conn.cursor()
        return self.conn, self.cursor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Handle transaction and return connection to pool."""
        if exc_type:
            # If there was an exception, rollback the transaction
            if self.conn:
                try:
                    self.conn.rollback()
                    logger.warning("Transaction rolled back due to exception")
                except Exception as e:
                    logger.error(f"Error during rollback: {e}")
        else:
            # If no exception and auto_commit is True, commit the transaction
            if self.conn and self.auto_commit:
                try:
                    self.conn.commit()
                except Exception as e:
                    logger.error(f"Error during commit: {e}")
        
        # Close cursor
        if self.cursor:
            try:
                self.cursor.close()
            except Exception as e:
                logger.error(f"Error closing cursor: {e}")
        
        # Return connection to pool (don't close it!)
        if self.conn:
            try:
                self.pool_manager.return_connection(self.conn)
            except Exception as e:
                logger.error(f"Error returning connection to pool: {e}")


def get_pooled_connection():
    """
    Get a connection from the pool.
    
    This is a convenience function for getting a pooled connection.
    Remember to return it using return_pooled_connection() when done.
    
    Returns:
        psycopg2.connection: A database connection from the pool
    """
    pool_manager = ConnectionPoolManager()
    return pool_manager.get_connection()


def return_pooled_connection(conn):
    """
    Return a connection to the pool.
    
    Args:
        conn: The connection to return
    """
    pool_manager = ConnectionPoolManager()
    pool_manager.return_connection(conn)


def get_pool_status() -> dict:
    """
    Get the current status of the connection pool.
    
    Returns:
        dict: Pool status information
    """
    pool_manager = ConnectionPoolManager()
    return pool_manager.get_pool_status()


def initialize_pool():
    """
    Explicitly initialize the connection pool.
    
    This can be called at application startup to ensure the pool
    is ready before handling requests.
    """
    ConnectionPoolManager()
    logger.info("Connection pool initialization requested")


def close_pool():
    """
    Close all connections in the pool.
    
    This should be called at application shutdown.
    """
    pool_manager = ConnectionPoolManager()
    pool_manager.close_all_connections()
