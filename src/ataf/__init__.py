"""ATAF — Agent Tool Acquisition Framework.

Runtime tool acquisition for AI agents. When an agent's LLM needs a
capability that isn't in the current tool catalog, the LLM writes the
missing tool as Python code. The ATAF server hot-deploys that code as
a live FastAPI endpoint, and the agent re-prompts the LLM with the
refreshed catalog.

See DESIGN.md at the repo root for the full architecture.
"""

__version__ = "0.1.0.dev0"
