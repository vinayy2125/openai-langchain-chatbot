
import pytest
from app.orchestrator.llm_adapter import ConstrainedLLMAdapter
from app.orchestrator.uc1_config import load_uc1_config

class TestSanitizerIDs:
    def test_strip_internal_ids(self):
        """Verify that UC1-A and similar IDs are stripped from response."""
        config = load_uc1_config()
        adapter = ConstrainedLLMAdapter(config)
        
        leaked_response = "I can help with the UC1-A area of product development."
        cleaned = adapter._sanitize_response(leaked_response)
        
        print(f"Original: {leaked_response}")
        print(f"Cleaned:  {cleaned}")
        
        assert "UC1-A" not in cleaned
        assert "UC1-A area" not in cleaned # Likely removes 'UC1-A', leaving ' area' or double space
        
    def test_strip_generic_uc_ids(self):
        """Verify generic UC IDs are stripped."""
        config = load_uc1_config()
        adapter = ConstrainedLLMAdapter(config)
        
        leaked = "Reference ID: UC2-EXIT."
        cleaned = adapter._sanitize_response(leaked)
        assert "UC2-EXIT" not in cleaned
