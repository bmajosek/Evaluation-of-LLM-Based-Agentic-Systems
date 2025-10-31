# Evaluation of LLM-Based Agentic Systems for Software Development Tasks

This project is a solution to the task of creating an LLM-based AI agent that fixes buggy Python code. The agent is evaluated on the HumanEvalFix dataset.

## Task Description

The goal was to implement an LLM-based AI agent to fix buggy Python code, using any agentic framework and any LLM. The solution needed to include code for running the agent, obtaining benchmark scores (pass@1), and providing instructions for running the code.

## Solution Overview

The implemented agent uses LangGraph to create a ReAct-style agent. It follows a "generate, test, reflect" loop:

1.  **Solve**: The agent generates an initial fix for the buggy code.
2.  **Test**: The fix is tested in a sandboxed environment.
3.  **Reflect**: If the tests fail, the agent reflects on the error and generates a new solution.

This project uses the `Qwen/Qwen2.5-0.5B-Instruct` model by default.

## Getting Started

### 1. Setup

```bash
pip install -r requirements.txt
```

### 2. Run the Demo & Evaluation

The `results.ipynb` notebook contains a step-by-step demonstration of the agent, a small-scale evaluation, and detailed explanations.

To run the full evaluation on a subset of the HumanEvalFix dataset, execute:

```bash
python main.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --max-new-tokens 256 \
  --retries 3
```

## Results

The evaluation script saves detailed per-task results to `runs/results.jsonl`. The file included in this repository contains the results for the default `Qwen/Qwen2.5-0.5B-Instruct` model.

The `results.ipynb` notebook also contains a direct comparison with a larger `Qwen/Qwen2.5-1.5B-Instruct` model, demonstrating how performance can vary with model size. Based on a small-scale experiment, the larger model shows stronger performance even with fewer retry attempts.

## Project Structure

```
.
├── agent/            # Core agent logic
│   ├── agent.py      # LangGraph state machine
│   ├── prompts.py    # Prompts for the LLM
│   ├── tools.py      # Sandboxed code execution
│   └── utils.py      # Helper functions
├── evaluation/       # Evaluation scripts
│   └── humaneval_eval.py
├── main.py           # Main script to run the evaluation
├── results.ipynb     # Demo notebook
└── requirements.txt
```

