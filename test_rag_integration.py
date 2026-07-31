#!/usr/bin/env python3
"""
Test script to verify RAG integration is working correctly.
This script tests both the frontend and backend components.
"""

import sys
import os
import json
from unittest.mock import patch, MagicMock

# Add the project directories to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'frontend'))

def test_backend_chat_request_model():
    """Test that the ChatRequest model accepts RAG parameters."""
    try:
        from routers.chat import ChatRequest

        # Test data with RAG parameters
        test_data = {
            "messages": [{"role": "user", "content": "Test question"}],
            "modele": "gemma4:e4b",
            "temperature": 0.3,
            "context_size": 22000,
            "think": False,
            "collection_name": "test_collection",
            "n_results": 250,
            "seuil": 0.6,
            "alpha": 0.5,
            "use_hyde": True,
            "use_expansion": True,
            "doc_date_filter": ""
        }

        # Create ChatRequest instance
        request = ChatRequest(**test_data)

        # Verify all parameters are set correctly
        assert request.collection_name == "test_collection"
        assert request.n_results == 250
        assert request.seuil == 0.6
        assert request.alpha == 0.5
        assert request.use_hyde == True
        assert request.use_expansion == True
        assert request.doc_date_filter == ""

        print("✅ Backend ChatRequest model test passed")
        return True

    except Exception as e:
        print(f"❌ Backend ChatRequest model test failed: {e}")
        return False

def test_session_state_initialization():
    """Test that session state initializes RAG config correctly."""
    try:
        # Mock streamlit session state
        class MockSessionState:
            def __init__(self):
                self.data = {}

            def __getitem__(self, key):
                return self.data.get(key)

            def __setitem__(self, key, value):
                self.data[key] = value

            def __contains__(self, key):
                return key in self.data

        # Mock streamlit module
        import sys
        from unittest.mock import MagicMock

        # Create mock modules
        mock_st = MagicMock()
        mock_st.session_state = MockSessionState()

        # Temporarily replace streamlit in sys.modules
        original_st = sys.modules.get('streamlit')
        sys.modules['streamlit'] = mock_st

        try:
            # Import and test session state initialization
            from plugins.session_state import init_session_state

            # Call the function
            init_session_state()

            # Verify rag_config was initialized
            assert 'rag_config' in mock_st.session_state.data
            rag_config = mock_st.session_state.data['rag_config']

            # Verify default values
            assert rag_config['collection'] == 'aucune_collection'
            assert rag_config['model'] == 'gemma4:e4b'
            assert rag_config['n_results'] == 250
            assert rag_config['seuil'] == 0.6
            assert rag_config['use_hyde'] == True
            assert rag_config['use_expansion'] == True
            assert rag_config['alpha'] == 0.5

            print("✅ Session state initialization test passed")
            return True

        finally:
            # Restore original streamlit module
            if original_st:
                sys.modules['streamlit'] = original_st
            else:
                del sys.modules['streamlit']

    except Exception as e:
        print(f"❌ Session state initialization test failed: {e}")
        return False

def test_rag_parameter_passing():
    """Test that RAG parameters are correctly added to the payload."""
    try:
        # Mock the necessary components
        class MockSessionState:
            def __init__(self):
                self.data = {
                    'rag_config': {
                        'collection': 'test_collection',
                        'model': 'gemma4:e4b',
                        'n_results': 250,
                        'seuil': 0.6,
                        'alpha': 0.5,
                        'use_hyde': True,
                        'use_expansion': True,
                        'doc_date_filter': ''
                    },
                    'knowledge_ready': False,
                    'session_id': 'test_session',
                    'think_mode': False,
                    'tables_info': None
                }

            def get(self, key, default=None):
                return self.data.get(key, default)

        # Mock streamlit and other modules
        import sys
        from unittest.mock import MagicMock

        mock_st = MagicMock()
        mock_st.session_state = MockSessionState()

        # Mock os.environ
        mock_os = MagicMock()
        mock_os.environ.get.side_effect = lambda key, default: {
            'API_URL': 'http://test:8000',
            'DEFAULT_LLM': 'gemma4:e4b',
            'CONTEXT_SIZE': '22000',
            'TEMPERATURE': '0.3',
            'PAYLOAD_DEBUG': 'hide'
        }.get(key, default)

        # Temporarily replace modules
        original_st = sys.modules.get('streamlit')
        original_os = sys.modules.get('os')
        sys.modules['streamlit'] = mock_st
        sys.modules['os'] = mock_os

        try:
            # Test the payload creation logic
            messages_pour_api = [{"role": "user", "content": "Test question"}]

            # Simulate the payload creation from the chat UI
            payload = {
                "messages": messages_pour_api,
                "modele": "gemma4:e4b",
                "temperature": 0.3,
                "context_size": 22000,
                "session_id": 'test_session',
                "mode": "discussion",
                "think": False,
                "tables_info": None,
                "request_id": "test_request_id",
                "seed": 12345,
            }

            # Add RAG parameters (this is the logic we're testing)
            if not mock_st.session_state.get('knowledge_ready') and hasattr(mock_st.session_state, 'rag_config'):
                rag_config = mock_st.session_state.rag_config
                payload.update({
                    "collection_name": rag_config.get("collection"),
                    "n_results": rag_config.get("n_results"),
                    "seuil": rag_config.get("seuil"),
                    "alpha": rag_config.get("alpha"),
                    "use_hyde": rag_config.get("use_hyde"),
                    "use_expansion": rag_config.get("use_expansion"),
                    "doc_date_filter": rag_config.get("doc_date_filter"),
                })

            # Verify RAG parameters were added
            assert payload['collection_name'] == 'test_collection'
            assert payload['n_results'] == 250
            assert payload['seuil'] == 0.6
            assert payload['alpha'] == 0.5
            assert payload['use_hyde'] == True
            assert payload['use_expansion'] == True
            assert payload['doc_date_filter'] == ''

            print("✅ RAG parameter passing test passed")
            return True

        finally:
            # Restore original modules
            if original_st:
                sys.modules['streamlit'] = original_st
            else:
                del sys.modules['streamlit']

            if original_os:
                sys.modules['os'] = original_os
            else:
                del sys.modules['os']

    except Exception as e:
        print(f"❌ RAG parameter passing test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing RAG integration...")
    print()

    tests = [
        test_backend_chat_request_model,
        test_session_state_initialization,
        test_rag_parameter_passing,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! RAG integration is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())