import streamlit as st
import requests
import json
import httpx
import asyncio
import markdown
import time
import pandas as pd
import uuid
import re
from datetime import datetime
from typing import Dict, Any, List, Optional

# --- 1. Configuration Constants ---
API_URL = "http://localhost:8000/stream_query/"
METRICS_URL = "http://localhost:8000/metrics"

# --- 2. Page Config (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="NetSage",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'About': "# NetSage\nYour intelligent network assistant."
    }
)

# --- 3. Custom CSS (Immediately after page_config) ---
st.markdown("""
<style>
    html, body {
        height: 100%;
        margin: 0;
        padding: 0;
    }
    .main {
        height: 100vh; display: flex; flex-direction: column; background-color: #1A1B26;
        color: #D8DEE9; max-width: 1200px; margin: 0 auto; padding: 0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .main .block-container {
        flex-grow: 1; overflow-y: auto; padding-top: 1rem !important;
        padding-bottom: 140px !important; display: flex; flex-direction: column;
    }
    section[data-testid="stSidebar"] {
        background-color: #16161E; border-right: 1px solid rgba(255,255,255,0.1);
    }
    h1 {
        font-size: 1.75rem !important; font-weight: 700 !important; color: #F8F8F2 !important;
        text-align: center; margin-bottom: 0.5rem !important; letter-spacing: -0.01em;
    }
    .caption {
        text-align: center; color: #BFBFD5; margin-bottom: 2.5rem; font-size: 1rem;
    }
    .chat-messages-wrapper {
        flex-grow: 1; overflow-y: auto; padding: 1rem 0.5rem;
        display: flex; flex-direction: column-reverse;
    }
    .chat-messages-inner {
        display: flex; flex-direction: column; gap: 1rem;
        width: 100%; max-width: 800px; margin: 0 auto;
    }
    .chat-message {
        display: flex; margin-bottom: 1rem; width: 100%; animation: fadeIn 0.3s ease-out;
    }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .chat-message.user { justify-content: flex-end; }
    .chat-message.assistant { justify-content: flex-start; }
    .avatar {
        width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center;
        justify-content: center; font-size: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        flex-shrink: 0;
    }
    .chat-message.user .avatar { margin-left: 10px; }
    .chat-message.assistant .avatar { margin-right: 10px; }
    .avatar.assistant { background-color: #5E81AC; color: white; }
    .avatar.user { background-color: #434C5E; color: white; }
    .message-content {
        max-width: 75%; padding: 10px 15px; border-radius: 18px; position: relative;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); line-height: 1.55; word-wrap: break-word;
    }
    .chat-message.user .message-content { background-color: #4C566A; color: #ECEFF4; border-bottom-right-radius: 5px; }
    .chat-message.assistant .message-content { background-color: #3B4252; color: #E5E9F0; border-bottom-left-radius: 5px; }
    .message-text { margin: 0; white-space: pre-wrap; word-wrap: break-word; }
    .message-text p:last-child { margin-bottom: 0; }
    .message-text.error-text { color: #FF9999; }
    .message-text pre { margin: 0.5em 0; font-size: 0.9em; }
    .sources-box {
        margin-top: 10px; padding: 8px 12px; background-color: rgba(255,255,255,0.05);
        border-radius: 6px; font-size: 0.85rem; border: 1px solid rgba(255,255,255,0.1);
    }
    .sources-box strong { color: #88C0D0; display: block; margin-bottom: 5px; font-weight: 600; }
    .sources-box ul { margin: 0; padding-left: 18px; }
    .sources-box li { margin-bottom: 3px; color: #A3ABB2; }
    .timestamp { font-size: 0.7rem; color: rgba(236, 239, 244, 0.4); margin-top: 6px; display: block; }
    .chat-message.user .timestamp { text-align: right; }
    .chat-message.assistant .timestamp { text-align: left; }
    @keyframes blink { 0% { opacity: 1; } 50% { opacity: 0; } 100% { opacity: 1; } }
    .cursor { display: inline-block; animation: blink 1s infinite; margin-left: 1px; user-select: none; }
    .input-container {
        position: fixed !important; bottom: 0 !important; left: 0 !important; right: 0 !important;
        width: 100% !important; padding: 1rem !important; background-color: rgba(26, 27, 38, 0.97) !important;
        backdrop-filter: blur(12px) !important; -webkit-backdrop-filter: blur(12px) !important;
        border-top: 1px solid rgba(255,255,255,0.1) !important; z-index: 1000 !important;
        display: flex !important; justify-content: center !important; align-items: center !important;
        box-shadow: 0 -3px 15px rgba(0, 0, 0, 0.25) !important;
    }
    .input-container form { width: 100% !important; max-width: 750px; margin: 0 !important; padding: 0 !important; }
    .input-container .stForm { border: none !important; background: transparent !important; padding: 0 !important; margin: 0 !important; }
    .stTextInput > div > div > input {
        background-color: #2E3440 !important; color: #ECEFF4 !important;
        border: 1px solid rgba(255,255,255,0.1) !important; padding: 0.7rem 1.1rem !important;
        font-size: 0.95rem !important; line-height: 1.5 !important; box-shadow: none !important;
        transition: all 0.2s ease !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #88C0D0 !important; box-shadow: 0 0 0 2.5px rgba(136, 192, 208, 0.25) !important;
    }
    .stButton>button { border-radius: 25px !important; font-weight: 500 !important; transition: all 0.2s ease !important;
        height: calc(0.7rem * 2 + 1.1rem * 1.5 + 2px); /* Approx match input height */
    }
    .input-container .stButton>button {
        background-color: #88C0D0 !important; color: #2E3440 !important;
        padding: 0 1.2rem !important; min-width: 80px !important;
    }
    .input-container .stButton>button:hover { background-color: #81A1C1 !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .stTabs [data-baseweb="tab-list"] { justify-content: center; }
    .sidebar-header { padding: 1.5rem 1rem 1rem 1rem; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 1rem; }
    .sidebar-header h3 { color: #88C0D0; font-size: 1.25rem; font-weight: 600; margin: 0; letter-spacing: 0.2px; }
    div[data-testid="stSidebarNav"] + div div[data-testid="stButton"] button {
        width: 100%; background-color: rgba(255,255,255,0.03) !important; color: #D8DEE9 !important;
        border-radius: 6px !important; padding: 0.65rem 0.85rem !important;
        border-left: 3px solid transparent !important; justify-content: flex-start !important;
        font-weight: 400 !important; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;
        font-size: 0.9rem;
    }
    div[data-testid="stSidebarNav"] + div div[data-testid="stButton"] button:hover { background-color: rgba(255,255,255,0.07) !important; }
    div[data-testid="stSidebarNav"] + div div[data-testid="stButton"] button:disabled {
        background-color: rgba(136, 192, 208, 0.15) !important; border-left-color: #88C0D0 !important;
        color: #ECEFF4 !important; font-weight: 500 !important;
    }
    div[data-testid="stSidebarNav"] + div .stMultiColumn div[data-testid="stButton"] button {
        background: transparent !important; border: none !important; color: #A3ABB2 !important;
        padding: 0.2rem 0.3rem !important; font-size: 0.9rem !important; min-width: auto !important;
    }
    div[data-testid="stSidebarNav"] + div .stMultiColumn div[data-testid="stButton"] button:hover {
        background-color: rgba(255,255,255,0.1) !important; color: #ECEFF4 !important;
    }
    .rename-input-container .stTextInput input {
        background-color: #3B4252 !important; color: #ECEFF4 !important; border-radius: 6px !important;
        border: 1px solid rgba(255,255,255,0.2) !important; padding: 0.5rem 0.75rem !important;
        font-size: 0.9rem !important; margin-bottom: 0.5rem;
    }
    .rename-input-container .stButton button {
        padding: 0.4rem 0.6rem !important; height: auto !important; min-height: 0 !important; font-size: 0.85rem;
    }
    .stDataFrame { width: 100% !important; }
    .stDataFrame table { width: 100% !important; border-collapse: collapse; margin-top: 1rem; }
    .stDataFrame th { background-color: #2E3440; color: #ECEFF4; padding: 0.7rem 0.9rem; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.15); font-size: 0.9rem;}
    .stDataFrame td { background-color: #232733; color: #D8DEE9; padding: 0.7rem 0.9rem; border-bottom: 1px solid rgba(255,255,255,0.07); font-size: 0.85rem;}
    .stDataFrame tr:hover td { background-color: #2C3043; }
    .stMetric { border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)

# --- 4. Function Definitions ---
def initialize_session_state():
    defaults = {
        "conversations": {}, "current_conversation_id": None, "messages": [],
        "user_query_for_input_widget": "", "editing_conversation_id": None,
        "temp_conversation_title": "", "metrics_data": None, "active_tab": "Chat",
        "_trigger_async_call": False, "_async_error_message": None,
        # Temporary storage for async call parameters across Streamlit's implicit reruns
        "_query_to_send_after_callback": None,
        "_assistant_message_ref_after_callback": None,
        "_history_ref_after_callback": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def new_conversation(): # Called by on_click, no st.rerun() here
    conv_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    existing_titles = [conv["title"] for conv in st.session_state.conversations.values()]
    new_chat_number = 1
    while f"New Chat {new_chat_number}" in existing_titles: new_chat_number += 1
    st.session_state.conversations[conv_id] = {"title": f"New Chat {new_chat_number}", "history": [], "created_at": timestamp}
    st.session_state.current_conversation_id = conv_id
    st.session_state.messages = []
    st.session_state.user_query_for_input_widget = ""
    st.session_state.editing_conversation_id = None
    # Streamlit automatically reruns after this callback completes

def load_conversation(conv_id: str): # Called by on_click, no st.rerun() here
    if conv_id in st.session_state.conversations:
        st.session_state.current_conversation_id = conv_id
        st.session_state.messages = list(st.session_state.conversations[conv_id]["history"])
        st.session_state.user_query_for_input_widget = ""
        st.session_state.editing_conversation_id = None
    # Streamlit automatically reruns

def delete_conversation(conv_id: str): # Called by on_click, no st.rerun() here
    if conv_id in st.session_state.conversations:
        del st.session_state.conversations[conv_id]
        if st.session_state.current_conversation_id == conv_id:
            st.session_state.current_conversation_id = None
            st.session_state.messages = []
            st.session_state.user_query_for_input_widget = ""
            if st.session_state.conversations:
                most_recent_conv_id = sorted(st.session_state.conversations.items(), key=lambda item: item[1]["created_at"], reverse=True)[0][0]
                # We can't call load_conversation directly and expect an immediate UI update *within* this same callback flow if it also modifies state for rerun.
                # Instead, set the state that the next rerun will use.
                st.session_state.current_conversation_id = most_recent_conv_id # This will be picked up by the auto-rerun
                st.session_state.messages = list(st.session_state.conversations[most_recent_conv_id]["history"])
            # else: no other conversations, state is already cleared.
    # Streamlit automatically reruns

def start_rename_conversation(conv_id: str): # Called by on_click
    st.session_state.editing_conversation_id = conv_id
    st.session_state.temp_conversation_title = st.session_state.conversations[conv_id]["title"]
    # Streamlit automatically reruns

def save_rename_conversation(conv_id: str): # Called by on_click
    key = f"rename_input_{conv_id}"
    new_title = st.session_state.get(key, "").strip()
    if new_title and conv_id in st.session_state.conversations: st.session_state.conversations[conv_id]["title"] = new_title
    st.session_state.editing_conversation_id = None
    st.session_state.temp_conversation_title = ""
    if key in st.session_state: del st.session_state[key]
    # Streamlit automatically reruns

def cancel_rename_conversation(conv_id: str): # Called by on_click
    key = f"rename_input_{conv_id}"
    st.session_state.editing_conversation_id = None
    st.session_state.temp_conversation_title = ""
    if key in st.session_state: del st.session_state[key]
    # Streamlit automatically reruns

def format_timestamp(ts: float) -> str: return datetime.fromtimestamp(ts).strftime("%H:%M")

def render_message(msg_data: Dict[str, Any]):
    role, content = msg_data["role"], msg_data["content"]
    sources, timestamp = msg_data.get("sources"), msg_data.get("timestamp")
    is_thinking, is_error = msg_data.get("is_thinking", False), msg_data.get("is_error", False)
    avatar_icon = "🤖" if role == "assistant" else "👤"
    message_class = "assistant" if role == "assistant" else "user"
    display_content = content + ('<span class="cursor">▍</span>' if is_thinking else '')
    html_content = markdown.markdown(display_content, extensions=['fenced_code', 'codehilite'])
    avatar_html = f'<div class="avatar {message_class}">{avatar_icon}</div>'
    sources_html = ""
    if sources:
        source_items = "".join([f"<li>{s.get('source', 'N/A')} (Page {s.get('page', 'N/A')})</li>" for s in sources])
        sources_html = f"<div class='sources-box'><strong>Sources:</strong><ul>{source_items}</ul></div>"
    timestamp_html = f"<div class='timestamp'>{format_timestamp(timestamp)}</div>" if timestamp else ""
    content_div = f"<div class='message-content'><div class='message-text {'error-text' if is_error else ''}'>{html_content}</div>{sources_html}{timestamp_html}</div>"
    full_message_html = f'<div class="chat-message {message_class}">{content_div if role == "user" else avatar_html} {avatar_html if role == "user" else content_div}</div>'
    st.markdown(full_message_html, unsafe_allow_html=True)

async def stream_api_response(query: str, assistant_message_ui_ref: Dict[str, Any], current_conversation_history_ref: List[Dict[str, Any]]):
    api_chat_history = [{"role": msg["role"], "content": msg["content"]} for msg in current_conversation_history_ref[:-1]] # Excludes current user query from history sent to API
    request_data = {"query": query, "chat_history": api_chat_history}
    assistant_response_content, retrieved_sources, error_occurred = "", None, False
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", API_URL, json=request_data) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    chunk_str = chunk.decode('utf-8', errors='replace')
                    for line in chunk_str.splitlines():
                        if line.startswith('data:'):
                            json_data = line[len('data:'):].strip()
                            if json_data:
                                try:
                                    event = json.loads(json_data)
                                    if event["type"] == "token":
                                        assistant_response_content += event["data"]
                                        assistant_message_ui_ref["content"] = assistant_response_content # Modify ref directly
                                    elif event["type"] == "sources":
                                        retrieved_sources = event["data"]
                                        assistant_message_ui_ref["sources"] = retrieved_sources # Modify ref
                                    elif event["type"] == "error":
                                        assistant_response_content += f"\n\nError from API: {event['data']}"
                                        error_occurred = True; break
                                except json.JSONDecodeError:
                                    assistant_response_content += f"\n\nError: Malformed data: {json_data}"
                                    error_occurred = True; break
                    if error_occurred: break
    except httpx.HTTPStatusError as e: assistant_response_content, error_occurred = f"API error: {e.response.status_code} - {e.response.text}", True
    except httpx.RequestError as e: assistant_response_content, error_occurred = f"Connection error: {e}", True
    except Exception as e: assistant_response_content, error_occurred = f"An unexpected error occurred: {e}", True
    finally:
        assistant_message_ui_ref["content"], assistant_message_ui_ref["is_thinking"] = assistant_response_content, False
        assistant_message_ui_ref["is_error"], assistant_message_ui_ref["timestamp"] = error_occurred, time.time()
        if retrieved_sources: assistant_message_ui_ref["sources"] = retrieved_sources
        # The assistant_message_ui_ref is already part of st.session_state.messages, so it's updated.
        # Now update the permanent history record (which is a separate list in st.session_state.conversations)
        final_history_msg = {"role": "assistant", "content": assistant_response_content, "sources": retrieved_sources,
                             "timestamp": assistant_message_ui_ref["timestamp"], "is_error": error_occurred}
        current_conversation_history_ref.append(final_history_msg)
        # No st.rerun() here; the calling logic (async trigger block) will handle the final rerun.

def handle_submit(): # Called by on_click from form submission
    query = st.session_state.user_query_for_input_widget.strip()
    if not query: return

    if not st.session_state.current_conversation_id:
        conv_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title = query[:35].strip() + "..." if len(query) > 35 else query
        if not title: title = f"Chat {len(st.session_state.conversations) + 1}"
        st.session_state.conversations[conv_id] = {"title": title, "history": [], "created_at": timestamp}
        st.session_state.current_conversation_id = conv_id
        st.session_state.messages = [] # Clear UI messages for new conversation

    current_conv = st.session_state.conversations[st.session_state.current_conversation_id]
    
    # Add user message to UI and history
    user_msg = {"role": "user", "content": query, "timestamp": time.time()}
    st.session_state.messages.append(user_msg)
    current_conv["history"].append(user_msg)
    
    # Add assistant's "Thinking..." message to UI
    assistant_msg_ui = {"role": "assistant", "content": "Thinking", "is_thinking": True, "timestamp": time.time()}
    st.session_state.messages.append(assistant_msg_ui)
    
    st.session_state.user_query_for_input_widget = "" # Clear the input field's session state binding
    
    # Set up parameters for the async call that will happen in the *next* script run
    st.session_state._query_to_send_after_callback = query
    st.session_state._assistant_message_ref_after_callback = assistant_msg_ui # This is the actual dict object
    st.session_state._history_ref_after_callback = current_conv["history"] # This is the actual list object
    st.session_state._trigger_async_call = True
    # No st.rerun() here. Streamlit will automatically rerun after this callback completes.
    # The UI will show the user message and "Thinking..." due to state changes made above.

@st.cache_data(ttl=60)
def fetch_metrics_cached():
    try:
        response = requests.get(METRICS_URL); response.raise_for_status(); return response.json()
    except requests.exceptions.RequestException as e:
        # If metrics fail, we don't want to block the chat. Store error to display, return None.
        st.session_state._async_error_message = f"Error fetching metrics: {e}"
        return None

# --- 5. Session State Initialization Call ---
initialize_session_state()

# --- 6. Logic for Async Call Triggered by Rerun (after a callback) ---
if st.session_state.get("_trigger_async_call", False):
    # Retrieve parameters stored by the callback
    query_to_send = st.session_state.pop("_query_to_send_after_callback", None)
    assistant_message_ref = st.session_state.pop("_assistant_message_ref_after_callback", None)
    history_ref = st.session_state.pop("_history_ref_after_callback", None)
    
    # Reset the trigger immediately
    st.session_state._trigger_async_call = False

    if query_to_send and assistant_message_ref and history_ref:
        # Sanity check: ensure the assistant_message_ref (which is a dict) is still in st.session_state.messages
        # This is important as st.session_state.messages could have been cleared by another action (e.g. new_conversation)
        # if multiple interactions happen very quickly or state management is complex.
        # For a simple submit -> auto-rerun -> this block, the reference should still be valid.
        ref_is_valid_in_current_messages = any(msg is assistant_message_ref for msg in st.session_state.messages)

        if ref_is_valid_in_current_messages:
            asyncio.run(stream_api_response(query_to_send, assistant_message_ref, history_ref))
        else:
            # Fallback: If the exact object reference is lost, try to find the "Thinking" message.
            # This is less reliable as there should only be one "Thinking" state.
            recovered_ref = None
            for msg_in_list in st.session_state.messages:
                if msg_in_list.get("role") == "assistant" and msg_in_list.get("is_thinking"):
                    recovered_ref = msg_in_list # Use the object from the current list
                    asyncio.run(stream_api_response(query_to_send, recovered_ref, history_ref))
                    break
            if not recovered_ref:
                st.session_state._async_error_message = "Critical Error: Assistant message placeholder lost for API call."
    else:
        st.session_state._async_error_message = "Error: Incomplete data for scheduled API call."
    
    st.rerun() # This rerun IS necessary here in the main script body to show the final API response.

# --- 7. Main UI Rendering ---
st.title("NetSage")
st.caption("Your intelligent network assistant.")

if st.session_state.get("_async_error_message"):
    st.error(st.session_state._async_error_message)
    st.session_state._async_error_message = None # Clear after displaying

tab_chat, tab_metrics = st.tabs(["Chat", "Metrics"])

with tab_chat:
    st.markdown('<div class="chat-messages-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="chat-messages-inner">', unsafe_allow_html=True)
    if st.session_state.current_conversation_id or st.session_state.messages:
        for msg_data in st.session_state.messages: render_message(msg_data)
    elif not st.session_state.conversations: st.info("Welcome! Start a new conversation or type a question.")
    else: st.info("Select or start a new conversation.")
    st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    with st.form(key="chat_input_form", clear_on_submit=True): # clear_on_submit handles visual clearing
        cols = st.columns([1, 0.12])
        with cols[0]: st.text_input("Ask a question:", key="user_query_for_input_widget", placeholder="e.g., What is TCP?", label_visibility="collapsed")
        with cols[1]: st.form_submit_button("Send", on_click=handle_submit) # on_click sets up the async trigger
    st.markdown('</div>', unsafe_allow_html=True)

with tab_metrics:
    st.subheader("Performance Metrics")
    if st.button("🔄 Refresh Metrics", key="refresh_metrics_button"): # This is an on_click
        st.cache_data.clear()
        st.session_state.metrics_data = fetch_metrics_cached()
        # Streamlit automatically reruns after this.
    if 'metrics_data' not in st.session_state or st.session_state.metrics_data is None:
        st.session_state.metrics_data = fetch_metrics_cached()

    if st.session_state.metrics_data:
        metrics = st.session_state.metrics_data
        m_col1, m_col2 = st.columns(2)
        with m_col1: st.metric("Total Queries", metrics.get("total_queries", 0))
        with m_col2: avg_time = metrics.get("avg_response_time"); st.metric("Avg Response Time", f"{avg_time:.2f}s" if avg_time is not None else "N/A")
        recent_queries = metrics.get("recent_queries", [])
        if recent_queries:
            st.subheader("Recent Queries")
            df = pd.DataFrame(recent_queries)
            if "timestamp" in df.columns: df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
            for col in ["total_time", "time_to_first_token", "sources_time"]:
                if col in df.columns: df[col] = df[col].apply(lambda x: f"{x:.2f}s" if pd.notnull(x) and isinstance(x, (int, float)) else "N/A")
            st.dataframe(df, use_container_width=True)
    elif not st.session_state.get("_async_error_message"):
        st.info("Metrics data not available. Try refreshing.")

with st.sidebar:
    st.markdown('<div class="sidebar-header"><h3>Conversations</h3></div>', unsafe_allow_html=True)
    if st.button("➕ New Conversation", on_click=new_conversation, use_container_width=True, type="primary"): pass # on_click handles it
    st.markdown("---")
    if st.session_state.conversations:
        sorted_conversations = sorted(st.session_state.conversations.items(), key=lambda item: item[1].get("created_at", "1970-01-01"), reverse=True)
        for conv_id, conv_data in sorted_conversations:
            is_active = conv_id == st.session_state.current_conversation_id
            if st.session_state.editing_conversation_id == conv_id:
                st.markdown('<div class="rename-input-container">', unsafe_allow_html=True)
                st.text_input("New title:", value=st.session_state.temp_conversation_title, key=f"rename_input_{conv_id}", label_visibility="collapsed")
                r_cols = st.columns([1,1]);
                with r_cols[0]: st.button("💾 Save", on_click=save_rename_conversation, args=(conv_id,), key=f"save_rename_{conv_id}", use_container_width=True)
                with r_cols[1]: st.button("❌ Cancel", on_click=cancel_rename_conversation, args=(conv_id,), key=f"cancel_rename_{conv_id}", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                item_cols = st.columns([0.7, 0.15, 0.15])
                with item_cols[0]: st.button(f"{conv_data['title']}", key=f"load_conv_{conv_id}", on_click=load_conversation, args=(conv_id,), disabled=is_active, use_container_width=True)
                with item_cols[1]: st.button("✏️", key=f"edit_conv_{conv_id}", on_click=start_rename_conversation, args=(conv_id,), help="Rename")
                with item_cols[2]: st.button("🗑️", key=f"delete_conv_{conv_id}", on_click=delete_conversation, args=(conv_id,), help="Delete")
            st.markdown("---", unsafe_allow_html=True)
    else: st.caption("No conversations yet.")

st.markdown("""
<script>
    const chatWrapper = document.querySelector('.chat-messages-wrapper');
    function scrollToBottom() { if (chatWrapper) { chatWrapper.scrollTop = 0; } }
    scrollToBottom(); // Initial scroll
    const observer = new MutationObserver((mutationsList, observer) => {
        for(const mutation of mutationsList) { if (mutation.type === 'childList' && mutation.addedNodes.length > 0) { scrollToBottom(); break; } }
    });
    if (chatWrapper) { observer.observe(chatWrapper, { childList: true, subtree: true }); }
</script>
""", unsafe_allow_html=True)