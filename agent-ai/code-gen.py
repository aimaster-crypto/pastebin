from langgraph.graph import StateGraph, END
import os
import subprocess
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import json
import re

# ---------- PROJECT ROOT ----------

PROJECT_ROOT = "generated_project"
os.makedirs(PROJECT_ROOT, exist_ok=True)

# ---------- TOOLS ----------

def create_folder(path: str):
    full_path = os.path.join(PROJECT_ROOT, path)
    os.makedirs(full_path, exist_ok=True)
    return f"Folder created: {full_path}"

def create_file(path: str, content: str):
    full_path = os.path.join(PROJECT_ROOT, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)
    return f"File created: {full_path}"

def read_file(path: str):
    full_path = os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(full_path):
        return f"[ERROR] File not found: {full_path}"
    with open(full_path, "r") as f:
        return f.read()

def show_file_tree(path=PROJECT_ROOT):
    tree = []
    for root, dirs, files in os.walk(path):
        for d in dirs:
            tree.append(os.path.relpath(os.path.join(root, d), PROJECT_ROOT) + "/")
        for f in files:
            tree.append(os.path.relpath(os.path.join(root, f), PROJECT_ROOT))
    return "\n".join(tree)

def run_maven(project_dir: str, goals: str = "clean compile"):
    try:
        result = subprocess.run(
            ["mvn"] + goals.split(),
            cwd=os.path.join(PROJECT_ROOT, project_dir),
            capture_output=True,
            text=True
        )
        return result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return str(e)

tools = {
    "create_folder": create_folder,
    "create_file": create_file,
    "read_file": read_file,
    "show_file_tree": show_file_tree,
    "run_maven": run_maven
}

tool_descriptions = {
    "create_folder": "Creates a folder at the specified path. Args: {'path': 'folder_path'}",
    "create_file": "Creates a file with given content. Args: {'path': 'file_path', 'content': 'file_content'}",
    "read_file": "Returns content of the specified file. Args: {'path': 'file_path'}",
    "show_file_tree": "Returns the current project file tree. Args: {}",
    "run_maven": "Runs specified Maven goals in the project directory. Args: {'project_dir': 'directory_path', 'goals': 'clean compile'}"
}

tool_list_text = "\n".join([f"- {name}: {desc}" for name, desc in tool_descriptions.items()])

# ---------- LLM SETUP ----------

load_dotenv()
GRQO_API_KEY = os.getenv("GRQO_API_KEY")

llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    openai_api_key=GRQO_API_KEY,
    openai_api_base="https://api.groq.com/openai/v1",
    temperature=0.2
)

def call_llm(prompt: str, state: dict) -> str:
    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content
    state.setdefault("llm_calls", []).append({"prompt": prompt, "response": text})
    print(f"\n[LLM PROMPT]: {prompt}\n[LLM RESPONSE]: {text}\n")
    return text

# ---------- NODES ----------

def _extract_json(text: str):
    """Extract JSON from a code block or plain text"""
    try:
        match = re.search(r"```json(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()
    except:
        return text.strip()

def extract_json(text: str):
    new_text = text.replace("\"", '*$quote*')
    new_text = new_text.replace("'", '"')
    print(new_text)
    return 
    

def planner(state: dict) -> dict:
    user_request = state.get("input", "")
    current_tree = show_file_tree()
    prompt = f"""
You are a Spring Boot coding assistant.
Available tools:
{tool_list_text}

Current project file tree:
{current_tree}

Break down the following request into the next actionable Spring Boot task. 
Respond in JSON format, either a single object or an array of objects.
Use SINGLE QUOTES for JSON keys and string values for tool.
If the project is complete, respond with 'DONE'.
User request: '{user_request}'
""" 
    next_task = call_llm(prompt, state)
    next_task_json = extract_json(next_task)
    state["tasks"] = [next_task_json] if next_task_json.strip().upper() != "DONE" else []
    return state

def executor_loop(state: dict) -> dict:
    state.setdefault("results", [])
    
    while state.get("tasks"):
        task_json = state["tasks"].pop(0)
        try:
            parsed = json.loads(task_json)
            if isinstance(parsed, list):
                task_list = parsed
            else:
                task_list = [parsed]
        except Exception as e:
            state["results"].append(f"[ERROR PARSING TASK]: {task_json} -> {e}")
            continue

        for task in task_list:
            tool_name = task.get("tool")
            args = task.get("args", {})

            if tool_name in tools:
                result = tools[tool_name](**args)
                print(f"[TOOL EXECUTED]: {tool_name} -> {str(result)[:200]}")
                state["results"].append(f"{tool_name}: {result}")
            else:
                state["results"].append(f"[UNKNOWN TOOL]: {tool_name}")

        # Ask LLM for next task
        current_tree = show_file_tree()
        prompt = f"""
Respond in JSON format (single object or array of objects).
If the project is complete, respond with 'DONE'.
Use SINGLE QUOTES for JSON keys and string values for tool.
The last task was executed: '{tool_name}' with args {args}.
Available tools:
{tool_list_text}

Current project file tree:
{current_tree}


What is the next task to continue building the Spring Boot project?
"""
# """
# Example response:
# ```json
# [{'tool': 'create_folder', 'args': {'path': 'src/main/java/com/example/app'}}]

# Return 'DONE' if the project is complete.
# Also do not reutrn anything else than json or DONE
# """
        next_task = call_llm(prompt, state)
        next_task_json = extract_json(next_task)
        if next_task_json.strip().upper() != "DONE":
            state["tasks"].append(next_task_json)

    return state

# ---------- LANGGRAPH WORKFLOW ----------

workflow = StateGraph(dict)
workflow.add_node("planner", planner)
workflow.add_node("executor", executor_loop)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", END)

app = workflow.compile()

# ---------- RUN ----------

if __name__ == "__main__":
    user_input = "Build me a Spring Boot app with a REST API for books"
    initial_state = {
        "input": user_input,
        "tasks": [],
        "results": [],
        "llm_calls": []
    }
    final_state = app.invoke(initial_state)

    print("\n--- FINAL RESULTS ---")
    for r in final_state["results"]:
        print(r)
