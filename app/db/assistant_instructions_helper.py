from app.db.base import DatabaseConnection

def get_assistant_instruction_by_key(key: str):
    """
    Fetch assistant_name and assistant_instruction from assistant_instructions table for a given key (id or name).
    Returns None if not found.
    """
    # Try integer id match if possible, else fallback to assistant_name
    try:
        int_key = int(key)
        query = """
            SELECT assistant_name, assistant_instruction
            FROM assistant_instructions
            WHERE active_state = TRUE AND id = %s
            LIMIT 1
        """
        params = (int_key,)
    except (ValueError, TypeError):
        query = """
            SELECT assistant_name, assistant_instruction
            FROM assistant_instructions
            WHERE active_state = TRUE AND assistant_name = %s
            LIMIT 1
        """
        params = (key,)
    with DatabaseConnection() as (conn, cursor):
        cursor.execute(query, params)
        row = cursor.fetchone()
        if row:
            return {"assistant_name": row[0], "assistant_instruction": row[1]}
        return None
