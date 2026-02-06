import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import logging
from logging.handlers import RotatingFileHandler

# Hardcoded config for simplification as per requirements
LOG_FILE = "instruction_changes.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

logger = logging.getLogger(__name__)

class ChangeLogger:
    """Manages logging of all changes to assistant_instructions table"""
    
    def __init__(self, log_file: str = LOG_FILE):
        """
        Initialize the change logger
        
        Args:
            log_file: Path to the log file (for data persistence, not application logs)
        """
        self.log_file = log_file
        self.ensure_log_file_exists()
        logger.info(f"ChangeLogger initialized with file: {log_file}")
    
    def ensure_log_file_exists(self):
        """Create log file if it doesn't exist"""
        if not os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    f.write("")  # Create empty file
                logger.info(f"Created new change log file: {self.log_file}")
            except Exception as e:
                logger.error(f"Failed to create log file: {str(e)}", exc_info=True)
                raise
    
    def log_change(self, operation: str, record_id: int, 
                   old_data: Optional[Dict] = None, 
                   new_data: Optional[Dict] = None):
        """
        Log a change to the database
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = {
            "timestamp": timestamp,
            "operation": operation,
            "record_id": record_id,
            "old_data": old_data,
            "new_data": new_data
        }
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            logger.info(f"Logged {operation} for record {record_id}")
        except Exception as e:
            logger.error(f"Failed to write to log file: {str(e)}", exc_info=True)
            raise Exception(f"Failed to write to log file: {str(e)}")
    
    def get_all_changes(self) -> List[Dict]:
        """
        Retrieve all logged changes
        """
        changes = []
        
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                changes.append(json.loads(line))
                            except json.JSONDecodeError:
                                logger.warning(f"Skipping malformed log line: {line[:50]}...")
                                continue
            
            # Return in reverse chronological order (newest first)
            return list(reversed(changes))
        except Exception as e:
            logger.error(f"Failed to read log file: {str(e)}", exc_info=True)
            raise Exception(f"Failed to read log file: {str(e)}")
    
    def get_recent_changes(self, limit: int = 10) -> List[Dict]:
        """
        Retrieve the most recent changes
        """
        all_changes = self.get_all_changes()
        return all_changes[:limit]
    
    def get_changes_by_record_id(self, record_id: int) -> List[Dict]:
        """
        Retrieve all changes for a specific record
        """
        all_changes = self.get_all_changes()
        return [
            change for change in all_changes 
            if change.get('record_id') == record_id
        ]
    
    def get_changes_by_operation(self, operation: str) -> List[Dict]:
        """
        Retrieve all changes of a specific operation type
        """
        all_changes = self.get_all_changes()
        return [
            change for change in all_changes 
            if change.get('operation') == operation
        ]
    
    def get_changes_by_date(self, date: str) -> List[Dict]:
        """
        Retrieve all changes from a specific date
        """
        all_changes = self.get_all_changes()
        return [
            change for change in all_changes 
            if change.get('timestamp', '').startswith(date)
        ]
    
    def get_log_count(self) -> int:
        """
        Get the total number of logged changes
        """
        return len(self.get_all_changes())
    
    def clear_logs(self):
        """Clear all log entries (use with caution!)"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("")
            logger.warning("All change logs cleared")
        except Exception as e:
            logger.error(f"Failed to clear log file: {str(e)}", exc_info=True)
            raise Exception(f"Failed to clear log file: {str(e)}")
    
    def export_logs(self, output_file: str):
        """
        Export logs to a different file
        """
        try:
            all_changes = self.get_all_changes()
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_changes, f, indent=2, ensure_ascii=False)
            logger.info(f"Logs exported to {output_file}")
        except Exception as e:
            logger.error(f"Failed to export logs: {str(e)}", exc_info=True)
            raise Exception(f"Failed to export logs: {str(e)}")
    
    def get_last_state_before_timestamp(self, record_id: int, 
                                       timestamp: str) -> Optional[Dict]:
        """
        Get the last known state of a record before a specific timestamp
        """
        changes = self.get_changes_by_record_id(record_id)
        
        for change in changes:
            if change.get('timestamp') < timestamp:
                if change.get('operation') in ['INSERT', 'UPDATE', 'UNDO_DELETE']:
                    return change.get('new_data')
                elif change.get('operation') in ['DELETE', 'UNDO_INSERT']:
                    return None
        
        return None
