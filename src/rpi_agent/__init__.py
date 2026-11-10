"""rpi_agent - RPI CLI Agent using Google ADK and OpenAI-compatible gateways.

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

Version 0.8.0 adds Universal and Exuberant Ctags indexing and symbol navigation
tools (`search_symbols`, `list_symbols`, and `generate_ctags`).

Version 0.9.0 defaults to standard OpenAI API endpoints, adds `--gemini` for
Google Gemini's OpenAI-compatible endpoint, and supports user configuration in
`~/.config/rpi-agent.yml`.

Version 0.10.0 adds structured logging with `structlog`, OpenTelemetry tracing
and HTTP request/response telemetry capture, an interactive split-screen REPL
log pane with scrollback, and on-demand or persistent log export at configurable
verbosity levels.

Version 0.10.1 opens the log pane automatically only for the DEBUG and INFO
levels (not ERROR or CRITICAL) and declares `httpx` as a direct dependency.
"""

__version__ = "0.10.1"
