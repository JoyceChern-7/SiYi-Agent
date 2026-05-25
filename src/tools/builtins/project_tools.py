from __future__ import annotations

from .common import *

def _agents_path(context: ToolContext) -> Path:
    return _state_dir(context) / "agents.json"

def _teams_path(context: ToolContext) -> Path:
    return _state_dir(context) / "teams.json"

def _task_records_path(context: ToolContext) -> Path:
    return _state_dir(context) / "task_records.json"

def _load_records(path: Path) -> list[JsonObject]:
    data = _read_json_file(path, [])
    return data if isinstance(data, list) else []

class AgentTool(BaseTool):
    name = "Agent"
    description = "Create, list, get, or delete a named sub-agent record."
    input_schema = _schema(
        {
            "action": {"type": "string", "enum": ["create", "list", "get", "delete"], "default": "create"},
            "agent_id": {"type": "string"},
            "name": {"type": "string"},
            "role": {"type": "string"},
            "prompt": {"type": "string"},
        },
    )

    def is_read_only(self, raw_input: JsonObject) -> bool:
        return str(raw_input.get("action") or "create") in {"list", "get"}

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        path = _agents_path(context)
        agents = _load_records(path)
        action = str(raw_input.get("action") or "create")
        if action == "list":
            return _ok(json.dumps(agents, ensure_ascii=False, indent=2), context, {"agents": agents})
        if action == "get":
            agent = next((item for item in agents if item.get("agent_id") == raw_input.get("agent_id")), None)
            if agent is None:
                return _error(f"Agent not found: {raw_input.get('agent_id')}", context)
            return _ok(json.dumps(agent, ensure_ascii=False, indent=2), context, agent)
        if action == "delete":
            before = len(agents)
            agents = [item for item in agents if item.get("agent_id") != raw_input.get("agent_id")]
            _write_json_file(path, agents)
            return _ok(f"deleted {before - len(agents)} agent(s)", context, {"deleted": before - len(agents)})

        agent_id = new_id("agent")
        agent = {
            "agent_id": agent_id,
            "name": raw_input.get("name") or agent_id,
            "role": raw_input.get("role") or "general",
            "prompt": raw_input.get("prompt") or "",
            "messages": [],
            "created_at": time.time(),
        }
        agents.append(agent)
        _write_json_file(path, agents)
        return _ok(json.dumps(agent, ensure_ascii=False, indent=2), context, agent)

def _siyi_command(prompt: str, context: ToolContext) -> list[str]:
    exe = shutil.which("siyi")
    if exe:
        return [exe, "--cwd", context.project_root, "--internal-worker-prompt", prompt]
    env_pythonpath = os.environ.get("PYTHONPATH")
    src_path = str(Path(__file__).resolve().parents[1])
    os.environ["PYTHONPATH"] = f"{src_path}{os.pathsep}{env_pythonpath}" if env_pythonpath else src_path
    return [sys.executable, "-m", "app.cli", "--cwd", context.project_root, "--internal-worker-prompt", prompt]

class SendMessageTool(BaseTool):
    name = "SendMessage"
    description = "Append a message to an agent or team record."
    input_schema = _schema(
        {
            "target_id": {"type": "string"},
            "message": {"type": "string"},
            "target_type": {"type": "string", "enum": ["agent", "team"], "default": "agent"},
        },
        required=["target_id", "message"],
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        target_type = str(raw_input.get("target_type") or "agent")
        path = _teams_path(context) if target_type == "team" else _agents_path(context)
        records = _load_records(path)
        id_key = "team_id" if target_type == "team" else "agent_id"
        target = next((item for item in records if item.get(id_key) == raw_input["target_id"]), None)
        if target is None:
            return _error(f"{target_type} not found: {raw_input['target_id']}", context)
        target.setdefault("messages", []).append(
            {"role": "user", "content": raw_input["message"], "created_at": time.time()}
        )
        _write_json_file(path, records)
        return _ok(f"sent message to {target_type} {raw_input['target_id']}", context, target)

class TeamCreateTool(BaseTool):
    name = "TeamCreate"
    description = "Create a team record containing agent ids."
    input_schema = _schema(
        {
            "name": {"type": "string"},
            "agent_ids": {"type": "array", "items": {"type": "string"}},
        },
        required=["name"],
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        teams = _load_records(_teams_path(context))
        team = {
            "team_id": new_id("team"),
            "name": raw_input["name"],
            "agent_ids": raw_input.get("agent_ids") or [],
            "messages": [],
            "created_at": time.time(),
        }
        teams.append(team)
        _write_json_file(_teams_path(context), teams)
        return _ok(json.dumps(team, ensure_ascii=False, indent=2), context, team)

class TeamDeleteTool(BaseTool):
    name = "TeamDelete"
    description = "Delete a team record."
    input_schema = _schema({"team_id": {"type": "string"}}, required=["team_id"])

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        teams = _load_records(_teams_path(context))
        before = len(teams)
        teams = [item for item in teams if item.get("team_id") != raw_input["team_id"]]
        _write_json_file(_teams_path(context), teams)
        return _ok(f"deleted {before - len(teams)} team(s)", context, {"deleted": before - len(teams)})

class TaskCreateTool(BaseTool):
    name = "TaskCreate"
    description = "Create a task record."
    input_schema = _schema(
        {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "status": {"type": "string", "default": "pending"},
        },
        required=["title"],
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        records = _load_records(_task_records_path(context))
        task = {
            "task_id": new_id("taskrec"),
            "title": raw_input["title"],
            "description": raw_input.get("description") or "",
            "status": raw_input.get("status") or "pending",
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        records.append(task)
        _write_json_file(_task_records_path(context), records)
        return _ok(json.dumps(task, ensure_ascii=False, indent=2), context, task)

class TaskGetTool(BaseTool):
    name = "TaskGet"
    description = "Get a task record by id."
    read_only = True
    input_schema = _schema({"task_id": {"type": "string"}}, required=["task_id"])

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        records = _load_records(_task_records_path(context))
        task = next((item for item in records if item.get("task_id") == raw_input["task_id"]), None)
        if task is None:
            return _error(f"Task record not found: {raw_input['task_id']}", context)
        return _ok(json.dumps(task, ensure_ascii=False, indent=2), context, task)

class TaskListTool(BaseTool):
    name = "TaskList"
    description = "List task records."
    read_only = True
    input_schema = _schema({})

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        del raw_input
        records = _load_records(_task_records_path(context))
        data = {"records": records}
        return _ok(json.dumps(data, ensure_ascii=False, indent=2), context, data)

class TaskUpdateTool(BaseTool):
    name = "TaskUpdate"
    description = "Update fields on a task record."
    input_schema = _schema(
        {
            "task_id": {"type": "string"},
            "status": {"type": "string"},
            "title": {"type": "string"},
            "description": {"type": "string"},
            "metadata": {"type": "object"},
        },
        required=["task_id"],
    )

    async def run(self, raw_input: JsonObject, context: ToolContext) -> ToolResult:
        records = _load_records(_task_records_path(context))
        task = next((item for item in records if item.get("task_id") == raw_input["task_id"]), None)
        if task is None:
            return _error(f"Task record not found: {raw_input['task_id']}", context)
        for key in ("status", "title", "description", "metadata"):
            if key in raw_input and raw_input[key] is not None:
                task[key] = raw_input[key]
        task["updated_at"] = time.time()
        _write_json_file(_task_records_path(context), records)
        return _ok(json.dumps(task, ensure_ascii=False, indent=2), context, task)
