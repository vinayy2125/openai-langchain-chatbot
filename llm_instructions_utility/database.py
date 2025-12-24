import sys
import os
# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import psycopg2
from psycopg2 import sql, extras
from typing import List, Dict, Optional
from datetime import datetime
import logging
import time
from app.db.pool import PooledDatabaseConnection

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages PostgreSQL database operations for assistant_instructions table"""
    
    def __init__(self):
        """
        Initialize database manager.
        Uses the application's connection pool, so no credentials needed here.
        """
        self.ensure_table_exists()
    
    def ensure_table_exists(self):
        """Create assistant_instructions table if it doesn't exist"""
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS assistant_instructions (
            id SERIAL PRIMARY KEY,
            assistant_name VARCHAR(255) NOT NULL,
            assistant_instruction TEXT NOT NULL,
            active_state BOOLEAN DEFAULT TRUE,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            modified_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE OR REPLACE FUNCTION update_modified_date_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.modified_date = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        
        DROP TRIGGER IF EXISTS update_assistant_instructions_modified_date ON assistant_instructions;
        
        CREATE TRIGGER update_assistant_instructions_modified_date
        BEFORE UPDATE ON assistant_instructions
        FOR EACH ROW
        EXECUTE FUNCTION update_modified_date_column();
        """
        
        try:
            logger.info("Verifying table schema...")
            # Use connection pool context manager
            with PooledDatabaseConnection() as (conn, cursor):
                # Check if table exists first
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'assistant_instructions'
                    );
                """)
                table_exists = cursor.fetchone()[0]
                
                if table_exists:
                    logger.debug("Table already exists. Skipping creation.")
                else:
                    logger.info("Table does not exist. Creating...")
                    cursor.execute(create_table_query)
                    # Context manager auto-commits on success
                    logger.info("Table created successfully")
                    
        except Exception as e:
            logger.error(f"Failed to create table: {str(e)}", exc_info=True)
            raise Exception(f"Failed to create table: {str(e)}")
    
    def get_all_instructions(self) -> List[Dict]:
        """
        Retrieve all instructions from the database
        """
        query = """
        SELECT id, assistant_name, assistant_instruction, active_state,
               created_date, modified_date
        FROM assistant_instructions
        ORDER BY id;
        """
        
        try:
            start_time = time.time()
            with PooledDatabaseConnection() as (conn, cursor):
                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                results = cursor.fetchall()
                
                logger.info(f"Retrieved {len(results)} instructions in {time.time() - start_time:.4f}s")
                
                # Convert to list of dicts
                instructions = []
                for row in results:
                    row_dict = dict(zip(columns, row))
                    instructions.append({
                        'id': row_dict['id'],
                        'assistant_name': row_dict['assistant_name'],
                        'assistant_instruction': row_dict['assistant_instruction'],
                        'active_state': row_dict['active_state'],
                        'created_date': row_dict['created_date'].isoformat() if row_dict['created_date'] else None,
                        'modified_date': row_dict['modified_date'].isoformat() if row_dict['modified_date'] else None
                    })
                return instructions
                
        except Exception as e:
            logger.error(f"Failed to retrieve instructions: {str(e)}", exc_info=True)
            raise Exception(f"Failed to retrieve instructions: {str(e)}")
    
    def get_instruction_by_id(self, instruction_id: int) -> Optional[Dict]:
        """
        Retrieve a specific instruction by ID
        """
        query = """
        SELECT id, assistant_name, assistant_instruction, active_state,
               created_date, modified_date
        FROM assistant_instructions
        WHERE id = %s;
        """
        
        try:
            with PooledDatabaseConnection() as (conn, cursor):
                cursor.execute(query, (instruction_id,))
                row = cursor.fetchone()
                
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    row_dict = dict(zip(columns, row))
                    return {
                        'id': row_dict['id'],
                        'assistant_name': row_dict['assistant_name'],
                        'assistant_instruction': row_dict['assistant_instruction'],
                        'active_state': row_dict['active_state'],
                        'created_date': row_dict['created_date'].isoformat() if row_dict['created_date'] else None,
                        'modified_date': row_dict['modified_date'].isoformat() if row_dict['modified_date'] else None
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve instruction {instruction_id}: {str(e)}", exc_info=True)
            raise Exception(f"Failed to retrieve instruction: {str(e)}")
    
    def add_instruction(self, assistant_name: str, assistant_instruction: str, 
                       active_state: bool = True) -> int:
        """
        Add a new instruction to the database
        """
        # Input validation
        if not assistant_name or not assistant_name.strip():
            raise ValueError("Assistant name cannot be empty")
        if len(assistant_name) > 255:
            raise ValueError("Assistant name too long (max 255 characters)")
        if not assistant_instruction or not assistant_instruction.strip():
            raise ValueError("Instruction cannot be empty")

        query = """
        INSERT INTO assistant_instructions (assistant_name, assistant_instruction, active_state)
        VALUES (%s, %s, %s)
        RETURNING id;
        """
        
        try:
            with PooledDatabaseConnection() as (conn, cursor):
                cursor.execute(query, (assistant_name, assistant_instruction, active_state))
                new_id = cursor.fetchone()[0]
                # Context manager commits on exit
                logger.info(f"Added new instruction with ID {new_id}")
                return new_id
        except Exception as e:
            logger.error(f"Failed to add instruction: {str(e)}", exc_info=True)
            raise Exception(f"Failed to add instruction: {str(e)}")
    
    def update_instruction(self, instruction_id: int, assistant_name: str = None, 
                          assistant_instruction: str = None, active_state: bool = None):
        """
        Update an existing instruction
        """
        # Input validation
        if assistant_name is not None:
            if not assistant_name.strip():
                raise ValueError("Assistant name cannot be empty")
            if len(assistant_name) > 255:
                raise ValueError("Assistant name too long (max 255 characters)")
        
        if assistant_instruction is not None and not assistant_instruction.strip():
            raise ValueError("Instruction cannot be empty")

        # Build dynamic update query based on provided parameters
        update_fields = []
        params = []
        
        if assistant_name is not None:
            update_fields.append("assistant_name = %s")
            params.append(assistant_name)
        
        if assistant_instruction is not None:
            update_fields.append("assistant_instruction = %s")
            params.append(assistant_instruction)
        
        if active_state is not None:
            update_fields.append("active_state = %s")
            params.append(active_state)
        
        if not update_fields:
            return  # Nothing to update
        
        params.append(instruction_id)
        
        query = f"""
        UPDATE assistant_instructions
        SET {', '.join(update_fields)}
        WHERE id = %s;
        """
        
        try:
            with PooledDatabaseConnection() as (conn, cursor):
                cursor.execute(query, params)
                # Context manager commits
                logger.info(f"Updated instruction {instruction_id}")
        except Exception as e:
            logger.error(f"Failed to update instruction {instruction_id}: {str(e)}", exc_info=True)
            raise Exception(f"Failed to update instruction: {str(e)}")
    
    def delete_instruction(self, instruction_id: int):
        """
        Delete an instruction from the database
        """
        query = "DELETE FROM assistant_instructions WHERE id = %s;"
        
        try:
            with PooledDatabaseConnection() as (conn, cursor):
                cursor.execute(query, (instruction_id,))
                # Context manager commits
                logger.info(f"Deleted instruction {instruction_id}")
        except Exception as e:
            logger.error(f"Failed to delete instruction {instruction_id}: {str(e)}", exc_info=True)
            raise Exception(f"Failed to delete instruction: {str(e)}")
