from typing import Callable, TypedDict, Dict, Any
from langgraph.graph import StateGraph, END

from agent.prompts import USER_TEMPLATE, CRITIC_TEMPLATE
from agent.tools import run_code_with_tests
from agent.utils import (
    extract_or_fallback,
    autopatch_typing,
    autopatch_unbound_local,
)


class AgentState(TypedDict):
    """State for the agent graph."""
    imports: str
    buggy_body: str
    entry_point: str
    tests: str
    declaration: str

    retries_left: int
    last_step: str
    raw: str
    code: str
    passed: bool
    error: str


def _build_user_prompt(imports: str, buggy_body: str, entry_point: str, declaration: str) -> str:
    """Builds the initial prompt."""
    return USER_TEMPLATE.format(
        entry_point=entry_point, imports=imports or "(none)", buggy_body=buggy_body or "(missing)", declaration=declaration or "(missing)"
    )


def _build_critic_prompt(error: str, prev_code: str) -> str:
    """Builds the reflection prompt."""
    return CRITIC_TEMPLATE.format(error=error or "(no error?)", prev_code=prev_code or "")

def solve_node(state: AgentState, model: Callable[[str], str]) -> AgentState:
    """Generates a potential solution."""
    prompt = _build_user_prompt(state["imports"], state["buggy_body"], state["entry_point"], state["declaration"])
    raw = model(prompt)
    code = extract_or_fallback(raw, state["entry_point"])
    state.update({"raw": raw, "code": code, "retries_left": state["retries_left"] - 1, "last_step": "solve"})
    return state


def reflect_node(state: AgentState, model: Callable[[str], str]) -> AgentState:
    """Reflects on the error and provides a new solution."""
    prompt = _build_critic_prompt(state["error"], state["code"])
    raw = model(prompt)
    code = extract_or_fallback(raw, state["entry_point"])
    state.update({"raw": raw, "code": code, "retries_left": state["retries_left"] - 1, "last_step": "reflect"})
    return state


def test_node(state: AgentState) -> AgentState:
    """Tests the generated code and auto-patches common errors."""
    ok, err = run_code_with_tests(state["code"], state["tests"], state["imports"])

    if not ok:
        ok2, err2, patched = autopatch_typing(state["code"], state["tests"], state["imports"])
        if ok2:
            state.update({"code": patched, "passed": True, "error": ""})
            return state
        err = err2 or err

        ok3, err3, patched2 = autopatch_unbound_local(state["code"], state["tests"], state["imports"], err)
        if ok3:
            state.update({"code": patched2, "passed": True, "error": ""})
            return state
        err = err3 or err

    state.update({"passed": ok, "error": err})
    return state


def _route(state: AgentState) -> str:
    """Routes the agent to the next step."""
    if state["passed"] or state["retries_left"] <= 0:
        return "end"
    return "reflect" if state.get("last_step") == "solve" else "solve"


def create_agent(model: Callable[[str], str]):
    """Creates the LangGraph agent."""
    g = StateGraph(AgentState)

    def _solve(state: AgentState) -> AgentState:
        return solve_node(state, model)

    def _reflect(state: AgentState) -> AgentState:
        return reflect_node(state, model)

    g.add_node("solve", _solve)
    g.add_node("reflect", _reflect)
    g.add_node("test", test_node)

    g.set_entry_point("solve")
    g.add_edge("solve", "test")
    g.add_edge("reflect", "test")
    g.add_conditional_edges("test", _route, {"solve": "solve", "reflect": "reflect", "end": END})

    return g.compile()


def run_agent(
    agent,
    *,
    imports: str,
    buggy_body: str,
    entry_point: str,
    declaration: str,
    tests: str,
    max_retries: int = 4,
) -> Dict[str, Any]:
    """Runs the agent."""
    init: AgentState = {
        "imports": imports,
        "buggy_body": buggy_body,
        "entry_point": entry_point,
        "declaration": declaration,
        "tests": tests,
        "retries_left": max_retries,
        "last_step": "",
        "raw": "",
        "code": "",
        "passed": False,
        "error": "",
    }
    out = agent.invoke(init)
    return {"program": out["code"], "passed": out["passed"], "error": out["error"], "raw": out["raw"]}
