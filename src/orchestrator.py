import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# ==========================================
# 1. SETUP & PATH RESOLUTION
# ==========================================
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[0]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agent_tools import query_telemetry_db, fetch_corridor_conditions, search_compliance_sop

load_dotenv(project_root / ".env")

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ==========================================
# 2. FACTORY INITIALIZATION: AGENT REASONER LLM
# ==========================================

AGENT_LLM_SETTING = os.getenv("Agent_llm", "GROQ").strip().upper()

if AGENT_LLM_SETTING == "OPENAI":
    print("🤖 Brain Mode: Utilizing Cloud OpenAI Reasoner (gpt-4o)...")
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

elif AGENT_LLM_SETTING in ["GROQ"]:
    groq_api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    groq_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip()
    if not groq_model:
        groq_model = "qwen/qwen3.8-27b"

    print(f"⚡ Brain Mode: Utilizing Groq Cloud Reasoner ({groq_model})...")
    effective_key = groq_api_key if groq_api_key else "placeholder_groq_key"

    llm = None
    # 1. Primary: native langchain-groq
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model=groq_model,
            temperature=0,
            groq_api_key=effective_key,
        )
        print(f"✅ Groq (langchain-groq) initialized successfully.")
    except Exception as e:
        print(f"ℹ️ langchain-groq unavailable ({e}). Trying OpenAI-compatible Groq endpoint...")

    # 2. Secondary: Groq's OpenAI-compatible endpoint via langchain-openai
    if llm is None:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=groq_model,
                temperature=0,
                api_key=effective_key,
                openai_api_key=effective_key,
                base_url="https://api.groq.com/openai/v1",
            )
            print(f"✅ Groq (OpenAI-compatible endpoint) initialized successfully.")
        except Exception as e:
            print(f"⚠️ Cloud Groq initialization failed ({e}). Falling back to local Ollama (qwen3:8b)...")
            from langchain_community.chat_models import ChatOllama
            llm = ChatOllama(model="qwen3:8b", temperature=0, num_predict=1024)

elif AGENT_LLM_SETTING in ["GEMINI", "GOOGLE"]:
    gemini_api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
    if not gemini_model:
        gemini_model = "gemini-3.8-flash"

    print(f"♊ Brain Mode: Utilizing Google Gemini Cloud Reasoner ({gemini_model})...")
    effective_key = gemini_api_key if gemini_api_key else "placeholder_gemini_key"

    llm = None
    # 1. Primary: native langchain-google-genai
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=gemini_model,
            temperature=0,
            google_api_key=effective_key,
        )
    except Exception as e:
        print(f"ℹ️ langchain-google-genai unavailable ({e}). Trying OpenAI-compatible Google endpoint...")

    # 2. Secondary: Google's official OpenAI-compatible endpoint via langchain-openai
    if llm is None:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=gemini_model,
                temperature=0,
                api_key=effective_key,
                openai_api_key=effective_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        except Exception as e:
            print(f"⚠️ Cloud Gemini initialization failed ({e}). Falling back to local Ollama (qwen3:8b)...")
            from langchain_community.chat_models import ChatOllama
            llm = ChatOllama(model="qwen3:8b", temperature=0, num_predict=1024)

elif AGENT_LLM_SETTING == "DEEPSEEK":
    print("🐳 Brain Mode: Utilizing DeepSeek Cloud Reasoner...")
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model="deepseek-v4-flash",
        temperature=0,
        openai_api_key=os.getenv("DEEPSEEK_API_KEY", "placeholder_key"),
        base_url="https://api.deepseek.com",
        max_tokens=2048,
    )

else:  # FALLBACK / DEFAULT RUNNER MODE
    print("🤗 Brain Mode: Local Fallback Activated. Binding Local Ollama (qwen3:8b)...")
    from langchain_community.chat_models import ChatOllama
    llm = ChatOllama(model="qwen3:8b", temperature=0, num_predict=1024)

fde_tools = [query_telemetry_db, fetch_corridor_conditions, search_compliance_sop]
try:
    llm_with_tools = llm.bind_tools(fde_tools)
except Exception as e:
    print(f"⚠️ Tool binding error ({e}), re-trying standard binding...")
    llm_with_tools = llm

# ==========================================
# 3. GRAPH ARCHITECTURE ASSEMBLY
# ==========================================
def reasoning_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

print("⚙️ Compiling LangGraph FDE Orchestrator...")
graph_builder = StateGraph(AgentState)
graph_builder.add_node("reasoner", reasoning_node)
graph_builder.add_node("tools", ToolNode(fde_tools))

graph_builder.add_edge(START, "reasoner")
graph_builder.add_conditional_edges("reasoner", tools_condition)
graph_builder.add_edge("tools", "reasoner")

fde_agent = graph_builder.compile(checkpointer=MemorySaver())

# ==========================================
# 4. CHAT LOOP TESTING PANEL
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*55)
    print("🚀 FDE Supply Chain Orchestrator State Machine Online")
    print(f"   Configured Execution: [LLM: {AGENT_LLM_SETTING}] -> [Embeddings: {os.getenv('Embeddings_model', 'LOCAL')}]")
    print("="*55 + "\n")
    
    # Load the business-structured system prompt from the external file
    prompt_path = project_root / "src" / "prompts" / "system_prompt.txt"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            system_instructions = f.read()
    except FileNotFoundError:
        print(f"Error: Could not find {prompt_path}")
        system_instructions = "You are a helpful AI assistant." # Basic fallback

    system_prompt = SystemMessage(content=system_instructions)
    
    thread_config = {"configurable": {"thread_id": "production_test_1"}}
    fde_agent.invoke({"messages": [system_prompt]}, config=thread_config)
    
    while True:
        user_input = input("\nDispatcher > ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        events = fde_agent.stream({"messages": [("user", user_input)]}, config=thread_config, stream_mode="updates")
        for event in events:
            for node_name, node_state in event.items():
                if node_name == "tools":
                    print("   [System] 🔄 Retrieving external data elements via ToolNode...")
                elif node_name == "reasoner":
                    latest_msg = node_state["messages"][-1]
                    if latest_msg.content:
                        print(f"\n🤖 FDE Agent:\n{latest_msg.content}")