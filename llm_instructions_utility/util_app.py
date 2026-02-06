import streamlit as st
import redis
import sys
import os
import json
import logging
from datetime import datetime
import pandas as pd

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from main app
try:
    from app.config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD
except ImportError:
    REDIS_HOST = 'localhost'
    REDIS_PORT = 6379
    REDIS_PASSWORD = None

# Import our custom modules
from database import DatabaseManager
from logger import ChangeLogger

# Constants
TEXTAREA_HEIGHT = 400
MAX_RECENT_CHANGES = 10

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis connection (singleton)
@st.cache_resource(show_spinner=False)
def get_redis_client():
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=0,
        password=REDIS_PASSWORD,
        decode_responses=True
    )

# Sync all prompt sections (instructions) from DB to Redis
def sync_prompt_sections_to_redis():
    redis_client = get_redis_client()
    if st.session_state.db_manager:
        try:
            # Fetch all instructions from DB
            data = st.session_state.db_manager.get_all_instructions()
            # Map instructions to the required structure (all as strings)
            prompt_json = {}
            for row in data:
                name = row.get('assistant_name')
                value = row.get('assistant_instruction', '')
                prompt_json[name] = value
            logger.info(f"Writing to Redis as chat_prompt_json mapping: {prompt_json}")
            # Store as a JSON object if RedisJSON is available, else fallback to string
            try:
                redis_client.json().set('chat_prompt_json', '$', prompt_json)
            except Exception as json_exc:
                logger.warning(f"RedisJSON not available, storing as string. {json_exc}")
                redis_client.set('chat_prompt_json', json.dumps(prompt_json, ensure_ascii=False))
            logger.info(f"Synced chat_prompt_json mapping to Redis.")
        except Exception as e:
            logger.error(f"Failed to sync prompt sections to Redis: {str(e)}", exc_info=True)

# Page configuration
st.set_page_config(
    page_title="LLM Instructions Manager",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .success-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    h1 {
        color: #2c3e50;
        font-weight: 700;
    }
    h2 {
        color: #34495e;
        font-weight: 600;
    }
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = None
if 'logger' not in st.session_state:
    st.session_state.logger = None
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'current_data' not in st.session_state:
    st.session_state.current_data = None

def initialize_connection():
    """Initialize database connection"""
    logger.info("Initializing database connection")
    try:
        st.session_state.db_manager = DatabaseManager()
        st.session_state.logger = ChangeLogger()
        st.session_state.connected = True
        logger.info("Successfully connected to database")
        return True, "✅ Successfully connected to database!"
    except Exception as e:
        st.session_state.connected = False
        logger.error(f"Connection failed: {str(e)}", exc_info=True)
        return False, f"❌ Connection failed: {str(e)}"

def load_current_data():
    """Load current data from database"""
    if st.session_state.db_manager:
        try:
            logger.debug("Loading current data from database")
            data = st.session_state.db_manager.get_all_instructions()
            st.session_state.current_data = data
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}", exc_info=True)
            st.error(f"Error loading data: {str(e)}")
            return None
    return None

def main():
    # Header
    st.title("🤖 LLM Instructions Manager")
    st.markdown("---")
    
    # Sidebar for connection status and controls
    with st.sidebar:
        st.header("📊 Status & Controls")
        
        # Auto-connect on first load
        if not st.session_state.connected and st.session_state.db_manager is None:
            with st.spinner("Connecting to database..."):
                success, message = initialize_connection()
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        # Connection status
        if st.session_state.connected:
            st.success("🟢 Database Connected")
            
            # Show connection details
            with st.expander("Connection Details"):
                try:
                    from app.db.base import DB_HOST, DB_PORT, DB_NAME, DB_USER
                    st.text(f"Host: {DB_HOST}")
                    st.text(f"Port: {DB_PORT}")
                    st.text(f"Database: {DB_NAME}")
                    st.text(f"User: {DB_USER}")
                except ImportError:
                    st.text("Using Application Pool")
            
            # Refresh button
            if st.button("🔄 Refresh Data"):
                logger.info("User requested data refresh")
                load_current_data()
                st.rerun()
            
            # Reconnect button (in case of connection issues)
            if st.button("🔌 Reconnect"):
                logger.info("User requested reconnection")
                st.session_state.db_manager = None
                st.session_state.connected = False
                st.rerun()
        else:
            st.error("🔴 Not Connected")
            st.warning("Please check your database configuration in the main app")
            
            if st.button("🔄 Retry Connection"):
                st.rerun()
        
        st.markdown("---")
        
        # Log file info
        st.header("📋 Log Information")
        if st.session_state.logger:
            log_count = st.session_state.logger.get_log_count()
            st.info(f"Total changes logged: {log_count}")
    
    # Main content area
    if not st.session_state.connected:
        st.info("👈 Please connect to the database using the sidebar")
        return
    
    # Load data if not already loaded
    if st.session_state.current_data is None:
        load_current_data()
    
    # Tabs for different operations
    tab1, tab2, tab3, tab4 = st.tabs(["📝 View & Edit", "➕ Add New", "🔙 Undo Changes", "📊 Change History"])
    
    # Tab 1: View and Edit
    with tab1:
        st.header("View & Edit Instructions")
        
        if st.session_state.current_data:
            # Display as a table
            df = pd.DataFrame(st.session_state.current_data)
            st.dataframe(df, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Edit Instruction")
            
            # Select instruction to edit
            instruction_ids = [row['id'] for row in st.session_state.current_data]
            selected_id = st.selectbox("Select Instruction ID", instruction_ids)
            
            if selected_id:
                # Find the selected instruction
                selected_instruction = next(
                    (item for item in st.session_state.current_data if item['id'] == selected_id),
                    None
                )
                
                if selected_instruction:
                    with st.form("edit_form"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            new_assistant_name = st.text_input(
                                "Assistant Name",
                                value=selected_instruction.get('assistant_name', '')
                            )
                        
                        with col2:
                            new_active_state = st.checkbox(
                                "Active",
                                value=selected_instruction.get('active_state', True)
                            )
                        
                        new_assistant_instruction = st.text_area(
                            "Assistant Instruction",
                            value=selected_instruction.get('assistant_instruction', ''),
                            height=TEXTAREA_HEIGHT
                        )
                        
                        submit_edit = st.form_submit_button("💾 Save Changes")
                        
                        if submit_edit:
                            try:
                                logger.info(f"Updating instruction {selected_id}")
                                # Log the change before updating
                                st.session_state.logger.log_change(
                                    operation="UPDATE",
                                    record_id=selected_id,
                                    old_data=selected_instruction,
                                    new_data={
                                        'id': selected_id,
                                        'assistant_name': new_assistant_name,
                                        'assistant_instruction': new_assistant_instruction,
                                        'active_state': new_active_state
                                    }
                                )
                                
                                # Update in database
                                st.session_state.db_manager.update_instruction(
                                    selected_id,
                                    assistant_name=new_assistant_name,
                                    assistant_instruction=new_assistant_instruction,
                                    active_state=new_active_state
                                )
                                sync_prompt_sections_to_redis()
                                st.success("✅ Instruction updated successfully!")
                                load_current_data()
                                st.rerun()
                            except Exception as e:
                                logger.error(f"Error updating instruction: {str(e)}", exc_info=True)
                                st.error(f"❌ Error updating instruction: {str(e)}")
        else:
            st.info("No data available. Please check your database connection.")
    
    # Tab 2: Add New
    with tab2:
        st.header("Add New Instruction")
        
        with st.form("add_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                add_assistant_name = st.text_input("Assistant Name")
            
            with col2:
                add_active_state = st.checkbox("Active", value=True)
            
            add_assistant_instruction = st.text_area("Assistant Instruction", height=TEXTAREA_HEIGHT)
            
            submit_add = st.form_submit_button("➕ Add Instruction")
            
            if submit_add:
                if not add_assistant_name or not add_assistant_instruction:
                    st.error("❌ Assistant Name and Instruction are required fields!")
                else:
                    try:
                        logger.info("Adding new instruction")
                        new_id = st.session_state.db_manager.add_instruction(
                            assistant_name=add_assistant_name,
                            assistant_instruction=add_assistant_instruction,
                            active_state=add_active_state
                        )
                        
                        # Log the addition
                        st.session_state.logger.log_change(
                            operation="INSERT",
                            record_id=new_id,
                            old_data=None,
                            new_data={
                                'id': new_id,
                                'assistant_name': add_assistant_name,
                                'assistant_instruction': add_assistant_instruction,
                                'active_state': add_active_state
                            }
                        )
                        
                        sync_prompt_sections_to_redis()
                        st.success(f"✅ Instruction added successfully! ID: {new_id}")
                        load_current_data()
                        st.rerun()
                    except Exception as e:
                        logger.error(f"Error adding instruction: {str(e)}", exc_info=True)
                        st.error(f"❌ Error adding instruction: {str(e)}")
    
    # Tab 3: Undo Changes
    with tab3:
        st.header("Undo Recent Changes")
        
        recent_changes = st.session_state.logger.get_recent_changes(MAX_RECENT_CHANGES)
        
        if recent_changes:
            st.info(f"📋 Showing last {len(recent_changes)} changes")
            
            for i, change in enumerate(recent_changes):
                with st.expander(
                    f"{change['timestamp']} - {change['operation']} (ID: {change['record_id']})"
                ):
                    st.json(change)
                    
                    if st.button(f"🔙 Undo This Change", key=f"undo_{i}"):
                        try:
                            logger.info(f"Undoing change {change['operation']} for record {change['record_id']}")
                            # Perform undo operation
                            if change['operation'] == 'UPDATE':
                                # Revert to old data
                                old_data = change['old_data']
                                st.session_state.db_manager.update_instruction(
                                    change['record_id'],
                                    assistant_name=old_data.get('assistant_name'),
                                    assistant_instruction=old_data.get('assistant_instruction'),
                                    active_state=old_data.get('active_state')
                                )
                                
                                # Log the undo operation
                                st.session_state.logger.log_change(
                                    operation="UNDO_UPDATE",
                                    record_id=change['record_id'],
                                    old_data=change['new_data'],
                                    new_data=change['old_data']
                                )
                                
                                st.success("✅ Change reverted successfully!")
                                load_current_data()
                                st.rerun()
                            
                            elif change['operation'] == 'INSERT':
                                # Delete the inserted record
                                st.session_state.db_manager.delete_instruction(change['record_id'])
                                sync_prompt_sections_to_redis()
                                # Log the undo operation
                                st.session_state.logger.log_change(
                                    operation="UNDO_INSERT",
                                    record_id=change['record_id'],
                                    old_data=change['new_data'],
                                    new_data=None
                                )
                                st.success("✅ Insertion reverted successfully!")
                                load_current_data()
                                st.rerun()
                            
                            elif change['operation'] == 'DELETE':
                                # Re-insert the deleted record
                                old_data = change['old_data']
                                st.session_state.db_manager.add_instruction(
                                    assistant_name=old_data.get('assistant_name'),
                                    assistant_instruction=old_data.get('assistant_instruction'),
                                    active_state=old_data.get('active_state', True)
                                )
                                sync_prompt_sections_to_redis()
                                # Log the undo operation
                                st.session_state.logger.log_change(
                                    operation="UNDO_DELETE",
                                    record_id=change['record_id'],
                                    old_data=None,
                                    new_data=change['old_data']
                                )
                                st.success("✅ Deletion reverted successfully!")
                                load_current_data()
                                st.rerun()
                        
                        except Exception as e:
                            logger.error(f"Error undoing change: {str(e)}", exc_info=True)
                            st.error(f"❌ Error undoing change: {str(e)}")
        else:
            st.info("No changes to undo")
    
    # Tab 4: Change History
    with tab4:
        st.header("Complete Change History")
        
        all_changes = st.session_state.logger.get_all_changes()
        
        if all_changes:
            # Create DataFrame for better visualization
            history_df = pd.DataFrame(all_changes)
            
            # Add filters
            col1, col2 = st.columns(2)
            
            with col1:
                operation_filter = st.multiselect(
                    "Filter by Operation",
                    options=history_df['operation'].unique(),
                    default=history_df['operation'].unique()
                )
            
            with col2:
                date_filter = st.date_input(
                    "Filter by Date",
                    value=None
                )
            
            # Apply filters
            filtered_df = history_df[history_df['operation'].isin(operation_filter)]
            
            if date_filter:
                filtered_df = filtered_df[
                    pd.to_datetime(filtered_df['timestamp']).dt.date == date_filter
                ]
            
            st.dataframe(filtered_df, use_container_width=True)
            
            # Download button for logs
            if st.button("📥 Download Change History"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"change_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No change history available")

if __name__ == "__main__":
    main()
