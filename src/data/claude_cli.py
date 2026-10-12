"""Headless Claude Code (``claude -p``) as a one-shot text model for the HaluEval answer rewrite.

Used when no API key is available: the calls run on the logged-in Claude subscription.

Each call is ONE subprocess with no tools, no MCP servers, no saved session and a fixed system prompt;
the prompt goes in on stdin. The process runs in an empty temporary working directory so no project
CLAUDE.md is loaded into the model's context. Processes share nothing, so several can run in parallel.

The model id is read back from the JSON output (``modelUsage``) and must equal the pinned id: a response
served by any other model (e.g. an automatic fallback) raises instead of being recorded. A non-zero exit,
a timeout, an error result, empty output or a rate-limit message is retried with exponential backoff, then
raises ``ClaudeCliError``. There is no fallback to another model or provider.

``claude -p`` has no temperature or max-token control, so outputs are not reproducible from the code alone;
the committed jsonl is the record.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
import time
from typing import Callable

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You rewrite answers exactly as instructed. Output only the sentence."
# Usage/rate-limit notices that can come back as a "successful" result instead of a rewrite.
_LIMIT_RE = re.compile(r"usage limit|rate[ _-]?limit|limit (reached|exceeded)|hit your limit|overloaded|too many requests", re.IGNORECASE)


class ClaudeCliError(RuntimeError):
    """A ``claude -p`` call failed after all retries, or returned something that must not be recorded."""


def build_command(model: str) -> list[str]:
    return [
        "claude", "-p",
        "--model", model,
        "--tools", "",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--system-prompt", SYSTEM_PROMPT,
        "--output-format", "json",
    ]


def cli_version(run: Callable | None = None) -> str:
    p = (run or subprocess.run)(["claude", "--version"], capture_output=True, text=True, timeout=60)
    if p.returncode != 0 or not p.stdout.strip():
        raise ClaudeCliError(f"`claude --version` failed (exit {p.returncode}): {p.stderr.strip()[:300]}")
    return p.stdout.strip()


class ClaudeCliClient:
    """One prompt in, one answer out, one fresh ``claude -p`` process per call."""

    def __init__(self, model: str, timeout_s: float = 180, max_retries: int = 5, backoff_s: float = 15,
                 run: Callable | None = None, sleep: Callable | None = None):
        if not model:
            raise ClaudeCliError("claude_cli: no model pinned in configs/providers.yaml")
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._run = run or subprocess.run
        self._sleep = sleep or time.sleep
        self.version = cli_version(self._run)
        self._cwd = tempfile.mkdtemp(prefix="claude_cli_empty_")  # no CLAUDE.md here

    def complete(self, prompt: str) -> tuple[str, str]:
        """Return (text, served model id) for ``prompt``; raise ``ClaudeCliError`` once retries are used up."""
        last = ""
        for attempt in range(self.max_retries + 1):
            if attempt:
                delay = self.backoff_s * 2 ** (attempt - 1)
                logger.warning("claude -p failed (%s); retry %d/%d in %.0fs", last, attempt, self.max_retries, delay)
                self._sleep(delay)
            try:
                p = self._run(build_command(self.model), input=prompt, capture_output=True, text=True,
                              timeout=self.timeout_s, cwd=self._cwd)
            except subprocess.TimeoutExpired:
                last = f"timeout after {self.timeout_s:.0f}s"
                continue
            text, last = self._parse(p)
            if text is not None:
                return text, self.model
        raise ClaudeCliError(f"claude -p failed after {self.max_retries + 1} attempts: {last}")

    def _parse(self, p: subprocess.CompletedProcess) -> tuple[str | None, str]:
        """(text, "") on success; (None, reason) for a retryable failure. Raises on a wrong model."""
        if p.returncode != 0:
            return None, f"exit {p.returncode}: {(p.stderr or p.stdout or '').strip()[:300]}"
        try:
            d = json.loads(p.stdout)
        except (json.JSONDecodeError, TypeError):
            return None, f"unparseable output: {(p.stdout or '').strip()[:300]}"
        result = (d.get("result") or "").strip()
        if d.get("is_error") or d.get("subtype") != "success" or d.get("api_error_status"):
            return None, f"error result ({d.get('subtype')}, {d.get('api_error_status')}): {result[:300]}"
        if not result:
            return None, "empty output"
        models = sorted((d.get("modelUsage") or {}).keys())
        if models != [self.model]:
            raise ClaudeCliError(f"claude -p answered with model(s) {models}, expected only {self.model!r}; refusing to record it")
        if _LIMIT_RE.search(result):
            return None, f"rate-limit message: {result[:300]}"
        return result, ""
