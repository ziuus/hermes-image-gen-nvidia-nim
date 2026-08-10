"""Test setup for hermes-image-gen-nvidia-nim.

The plugin implements the ``agent.image_gen_provider.ImageGenProvider`` ABC and
reads secrets/config through Hermes internals, so the Hermes Agent source tree
must be importable when the tests run.

Point ``HERMES_AGENT_REPO`` at a hermes-agent checkout, or rely on the default
location used by ``hermes setup`` (``~/.hermes/hermes-agent``).
"""

from __future__ import annotations

import os
import sys

import pytest


def _hermes_agent_repo() -> str:
    env = os.environ.get("HERMES_AGENT_REPO")
    if env:
        return env
    default = os.path.expanduser("~/.hermes/hermes-agent")
    if os.path.isdir(default):
        return default
    return env or default


@pytest.fixture(scope="session", autouse=True)
def _hermes_sys_path() -> None:
    repo = _hermes_agent_repo()
    if not os.path.isdir(repo):
        pytest.skip(
            "hermes-agent source tree not found (set HERMES_AGENT_REPO to its path)"
        )
    if repo not in sys.path:
        sys.path.insert(0, repo)
