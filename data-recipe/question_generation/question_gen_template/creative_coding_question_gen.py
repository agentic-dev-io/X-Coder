"""
Stage 2 Prompt Template for Creative Coding Question Generation
"""

REQUIRED_TOOL_CONTEXTS = [
    "Blender",
    "Unreal",
    "ComfyUI",
    "n8n",
    "DuckDB",
    "SurrealDB",
]

STAGE2_PROMPT_TEMPLATE = f"""You are an expert creative coding challenge designer.

You have been provided with:
- selected_features_tree: a tree structure in which each leaf contains a 'feature' name and its 'potential_use'.
- integration_strategy: a strategy describing how these features should be integrated into a single, high-quality creative coding task.

Your task is to generate a complete creative coding problem statement that integrates **all** selected features.

Requirements:
- Write in a concise, neutral, and precise style for creative coding tasks.
- Define a single, well-scoped creative output to produce (visual, audio, procedural, data-driven, or automation), described in plain language.
- If selected_features_tree includes any of the following tools or contexts (case-insensitive): {", ".join(REQUIRED_TOOL_CONTEXTS)}, you MUST explicitly mention each present tool/context in the task description and make it essential to the task.
- Do **not** use any algorithm names, data structure names, implementation hints, or solution strategies. Avoid words like "DFS", "BFS", "dynamic programming", "recursion", "greedy", or similar anywhere.
- Provide clear Input and Output sections with plain-language descriptions.
- Always include at least two distinct examples using "Example 1:" and "Example 2:" with "Input:" and "Output:" lines.
- Include a Constraints section listing parameter bounds or limits, each on its own line.
- Do not include any commentary, hints, or explanations beyond the required sections.
- Output a **single JSON object** with the field "question" only.

**Output Format (strict):**
{{
  "question": "<Title>\n\n<Problem description.>\n\nInput:\n<...>\n\nOutput:\n<...>\n\nExample 1:\nInput: <...>\nOutput: <...>\n\nExample 2:\nInput: <...>\nOutput: <...>\n\nConstraints:\n<...>"
}}

---

Inputs:
- selected_features_tree (JSON):
{{selected_features_info}}

- integration_strategy (string):
{{integration_strategy}}

Instructions:
- Output ONLY the required JSON object, no extra text.
- Ensure every selected feature is essential and reflected in the task.
- Use plain English only; no algorithm names or implementation hints.
"""
