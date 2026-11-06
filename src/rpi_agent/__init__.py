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

Version 0.7.1 fixes skill discovery crashing on malformed frontmatter, nested
path tab-completion, Python method outlining, and non-UTF-8 handling in
replace_in_file.

Version 0.7.2 survives quota exhaustion: rate-limit aware retries, daily-quota
detection with resume instructions, persisted sessions (--resume, /sessions),
and checkpointed sub-agent results reused on resume.
"""

__version__ = "0.7.2"
