"""
Migration script to update assistant_instructions table schema
This will safely migrate from old schema to new schema
"""

import psycopg2
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def migrate_schema():
    """Migrate the assistant_instructions table to new schema"""
    
    # Get database credentials
    host = os.getenv('DB_HOST')
    port = os.getenv('DB_PORT')
    database = os.getenv('DB_NAME')
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    
    print(f"Connecting to database: {database} at {host}:{port}")
    
    try:
        # Connect to database
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        conn.autocommit = False
        cursor = conn.cursor()
        
        print("Connected successfully!")
        
        # Check if table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'assistant_instructions'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            print("Table doesn't exist. Creating new table with correct schema...")
            create_new_table(cursor)
            conn.commit()
            print("✅ New table created successfully!")
        else:
            print("Table exists. Checking schema...")
            
            # Check current columns
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'assistant_instructions'
                ORDER BY ordinal_position;
            """)
            current_columns = [row[0] for row in cursor.fetchall()]
            print(f"Current columns: {current_columns}")
            
            # Check if migration is needed
            if 'assistant_name' in current_columns:
                print("✅ Table already has the new schema!")
            else:
                print("Migration needed. Starting migration...")
                migrate_existing_table(cursor, current_columns)
                conn.commit()
                print("✅ Migration completed successfully!")
        
        cursor.close()
        conn.close()
        print("\n🎉 Schema migration completed!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        if conn:
            conn.rollback()
        raise

def create_new_table(cursor):
    """Create new table with correct schema"""
    cursor.execute("""
        CREATE TABLE assistant_instructions (
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
        
        CREATE TRIGGER update_assistant_instructions_modified_date
        BEFORE UPDATE ON assistant_instructions
        FOR EACH ROW
        EXECUTE FUNCTION update_modified_date_column();
    """)

def migrate_existing_table(cursor, current_columns):
    """Migrate existing table to new schema"""
    
    print("Step 1: Creating backup table...")
    cursor.execute("""
        CREATE TABLE assistant_instructions_backup AS 
        SELECT * FROM assistant_instructions;
    """)
    
    print("Step 2: Dropping old table...")
    cursor.execute("DROP TABLE assistant_instructions CASCADE;")
    
    print("Step 3: Creating new table with correct schema...")
    create_new_table(cursor)
    
    print("Step 4: Migrating data...")
    
    # Determine migration strategy based on old columns
    if 'name' in current_columns and 'instruction' in current_columns:
        # Old schema: name, category, instruction, description, created_at, updated_at
        print("Migrating from old schema (name, instruction, etc.)...")
        cursor.execute("""
            INSERT INTO assistant_instructions 
                (assistant_name, assistant_instruction, active_state, created_date, modified_date)
            SELECT 
                name,
                instruction,
                TRUE,  -- Default all to active
                COALESCE(created_at, CURRENT_TIMESTAMP),
                COALESCE(updated_at, CURRENT_TIMESTAMP)
            FROM assistant_instructions_backup;
        """)
    else:
        print("Unknown old schema. Please manually migrate data.")
        return
    
    # Get count of migrated records
    cursor.execute("SELECT COUNT(*) FROM assistant_instructions;")
    count = cursor.fetchone()[0]
    print(f"✅ Migrated {count} records")
    
    print("Step 5: Dropping backup table...")
    cursor.execute("DROP TABLE assistant_instructions_backup;")

if __name__ == "__main__":
    print("=" * 60)
    print("Assistant Instructions Table Migration")
    print("=" * 60)
    print()
    
    response = input("This will migrate your assistant_instructions table. Continue? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        migrate_schema()
    else:
        print("Migration cancelled.")
