import psycopg2
from config import DB_CONFIG
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def check_and_kill_locks():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        cursor = conn.cursor()
        
        logger.info("Checking for blocking queries...")
        
        # Check for blocking queries
        cursor.execute("""
            SELECT pid, usename, state, query, query_start
            FROM pg_stat_activity
            WHERE state != 'idle'
            AND pid != pg_backend_pid()
            AND (query LIKE '%assistant_instructions%' OR query LIKE '%LOCK%');
        """)
        
        rows = cursor.fetchall()
        
        if not rows:
            logger.info("No blocking queries found.")
        else:
            logger.info(f"Found {len(rows)} potentially blocking queries:")
            for row in rows:
                pid, user, state, query, start = row
                logger.info(f"PID: {pid}, User: {user}, State: {state}, Start: {start}")
                logger.info(f"Query: {query}")
                
                # Kill the query
                logger.info(f"Attempting to terminate process {pid}...")
                try:
                    cursor.execute(f"SELECT pg_terminate_backend({pid});")
                    logger.info(f"Successfully terminated process {pid}")
                except Exception as e:
                    logger.error(f"Failed to terminate process {pid}: {str(e)}")
                    
        conn.close()
        
    except Exception as e:
        logger.error(f"Error checking locks: {str(e)}")

if __name__ == "__main__":
    check_and_kill_locks()
