"""rpi_agent - RPI CLI Agent using Google ADK and GenAI.mil.

Executes the Research, Plan, Iterate, Implement (RPI) methodology against local
Git repositories using Google Agent Development Kit (ADK), LiteLLM OpenAI-compatible
gateways, local skills discovered in ~/.agents/skills, and prompt_toolkit.

Version 0.6.0 adds token-conscious structural code outlining (Python, Ansible,
Terraform, Shell) and automated session history pruning to operate sustainably
within daily quota constraints.

Version 0.7.0 maintains functionality, but expands script to a multi-file Python
project, adds unit tests, type checking, coverage analysis, and distribution
builds.
"""

__version__ = "0.7.0"
