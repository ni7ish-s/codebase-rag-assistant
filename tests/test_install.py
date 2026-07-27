"""Tests for ``coderag install`` (:mod:`coderag.install`).

Everything runs against an isolated tmp ``$HOME`` and cwd so no real agent config is
touched. The wizard is driven by feeding scripted answers to ``input``.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
import yaml

from coderag import install as inst


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    return tmp_path


# --- claude (.mcp.json) ---------------------------------------------------------------


def test_claude_creates_and_is_idempotent(home):
    r = inst.install("claude")
    assert r.action == "created"
    data = json.loads(Path(r.path).read_text())
    assert "mcp" in data["mcpServers"]["coderag"]["args"]
    assert inst.install("claude").action == "unchanged"


def test_claude_merges_existing_and_backs_up(home):
    p = Path.cwd() / ".mcp.json"
    p.write_text(json.dumps({"mcpServers": {"other": {"command": "x", "args": []}}}))
    r = inst.install("claude")
    assert r.action == "updated"
    data = json.loads(p.read_text())
    assert {"other", "coderag"} <= set(data["mcpServers"])
    assert p.with_suffix(".json.bak").exists()


# --- hermes (~/.hermes/config.yaml) ---------------------------------------------------


def test_hermes_writes_yaml_with_tools(home):
    tools = ["search_code", "search_files"]
    r = inst.install("hermes", watched_dir=Path.cwd(), tools=tools)
    assert r.action == "created"
    data = yaml.safe_load(Path(r.path).read_text())
    entry = data["mcp_servers"]["coderag"]
    assert entry["tools"]["include"] == tools
    assert "--watched-dir" in entry["args"]
    assert (
        inst.install("hermes", watched_dir=Path.cwd(), tools=tools).action
        == "unchanged"
    )


def test_hermes_manual_without_pyyaml(home, monkeypatch):
    monkeypatch.setattr(inst, "yaml", None)
    r = inst.install("hermes")
    assert r.action == "manual" and "coderag" in r.detail


# --- codex (~/.codex/config.toml) -----------------------------------------------------


def test_codex_appends_and_is_idempotent(home):
    p = Path.home() / ".codex" / "config.toml"
    p.parent.mkdir()
    p.write_text("[other]\nx = 1\n")
    r = inst.install("codex")
    assert r.action == "appended"
    data = tomllib.loads(p.read_text())
    assert "other" in data
    assert data["mcp_servers"]["coderag"]["args"][0] == "mcp"
    assert p.with_suffix(".toml.bak").exists()
    assert inst.install("codex").action == "unchanged"


def test_codex_conflict_is_manual(home):
    p = Path.home() / ".codex" / "config.toml"
    p.parent.mkdir()
    p.write_text('[mcp_servers.coderag]\ncommand = "old"\nargs = []\n')
    assert inst.install("codex").action == "manual"


# --- shared behaviour -----------------------------------------------------------------


def test_dry_run_writes_nothing(home):
    r = inst.install("claude", dry_run=True)
    assert r.action == "would-write"
    assert not (Path.cwd() / ".mcp.json").exists()


def test_unknown_target_errors(home):
    assert inst.install("emacs").action == "error"


def test_detect_targets(home, monkeypatch):
    monkeypatch.setattr(inst.shutil, "which", lambda *_: None)
    assert inst.detect_targets() == []
    (Path.home() / ".hermes").mkdir()
    (Path.home() / ".codex").mkdir()
    assert set(inst.detect_targets()) == {"hermes", "codex"}


def test_wizard_collects_choices(home, monkeypatch):
    # answers: target "2" (hermes), keep default workspace, expose all tools "y"
    answers = iter(["2", "", "y"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    plans = inst.run_wizard([], Path.cwd())
    assert len(plans) == 1
    assert plans[0].target == "hermes"
    assert plans[0].tools == inst.DEFAULT_TOOLS


# --- workspace-scope guidance (large trees are supported) -----------------------------


def test_default_workspace_prefers_git_root(home):
    repo = Path.cwd()
    (repo / ".git").mkdir()
    deep = repo / "pkg" / "deep"
    deep.mkdir(parents=True)
    # Run from a subdirectory: the natural scope is the whole repo, not the subdir.
    assert inst.default_workspace(deep) == repo.resolve()


def test_default_workspace_falls_back_to_start(home):
    start = Path.cwd() / "loose"
    start.mkdir()
    assert inst.default_workspace(start) == start.resolve()


def test_is_broad_root_flags_home_and_system(home):
    assert inst._is_broad_root(Path("/"))  # filesystem root
    assert inst._is_broad_root(Path("/usr"))
    assert inst._is_broad_root(Path.home())  # the user's whole home
    assert not inst._is_broad_root(Path.cwd())  # a normal project dir


def test_wizard_describes_large_tree_support(home, monkeypatch, capsys):
    # Choosing "/" is a legitimate large-tree choice: the wizard sets expectations
    # (background, takes longer) and flags the /proc footgun, without discouraging it.
    answers = iter(["1", "/", ""])  # claude, watched=/, (no tools prompt for claude)
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    inst.run_wizard([], Path.cwd())
    out = capsys.readouterr().out
    assert "large tree" in out
    assert "/proc" in out  # the one genuine footgun for "/"
    assert "almost always what you want" not in out  # no longer discourages


# --- launcher resolution (the venv-activation footgun) --------------------------------
#
# An agent launches the server from its own shell, so the command written into its config
# must resolve there — not merely wherever `coderag install` happened to run.


def _patch_venv(monkeypatch, *, prefix: str, base_prefix: str) -> None:
    monkeypatch.setattr(inst.sys, "prefix", prefix)
    monkeypatch.setattr(inst.sys, "base_prefix", base_prefix)


def test_invocation_bare_for_system_install(monkeypatch):
    # Not in a venv and `coderag` on PATH → a durable bare command.
    _patch_venv(monkeypatch, prefix="/usr", base_prefix="/usr")
    monkeypatch.setattr(inst.shutil, "which", lambda *_a, **_k: "/usr/bin/coderag")
    assert inst._server_invocation(None) == ("coderag", ["mcp"])


def test_invocation_absolute_inside_activated_venv(monkeypatch, tmp_path):
    # The only `coderag` on PATH is the active venv's own bin/coderag, which disappears
    # once deactivated → pin that script by absolute path instead of a bare command.
    bindir = tmp_path / ".venv" / "bin"
    bindir.mkdir(parents=True)
    script = bindir / "coderag"
    script.write_text("#!stub\n")
    _patch_venv(monkeypatch, prefix=str(tmp_path / ".venv"), base_prefix="/usr")
    monkeypatch.setattr(inst.sys, "executable", str(bindir / "python"))
    monkeypatch.setattr(inst.shutil, "which", lambda *_a, **_k: str(script))
    assert inst._server_invocation(None) == (str(script), ["mcp"])


def test_invocation_absolute_when_venv_not_activated(monkeypatch, tmp_path):
    # Ran via `.venv/bin/coderag install` without activating: `coderag` is not on PATH,
    # but the env's console script is found in sys.executable's bin dir.
    bindir = tmp_path / ".venv" / "bin"
    bindir.mkdir(parents=True)
    script = bindir / "coderag"
    script.write_text("#!stub\n")
    _patch_venv(monkeypatch, prefix=str(tmp_path / ".venv"), base_prefix="/usr")
    monkeypatch.setattr(inst.sys, "executable", str(bindir / "python"))
    monkeypatch.setattr(
        inst.shutil, "which", lambda _n, path=None: str(script) if path else None
    )
    assert inst._server_invocation(None) == (str(script), ["mcp"])


def test_invocation_bare_for_pipx_shim_outside_venv(monkeypatch, tmp_path):
    # pipx: in a venv, but the PATH launcher lives outside the venv prefix (~/.local/bin),
    # so a bare `coderag` resolves without activation.
    (tmp_path / "venvs" / "coderag" / "bin").mkdir(parents=True)
    shim = tmp_path / ".local" / "bin" / "coderag"
    shim.parent.mkdir(parents=True)
    shim.write_text("#!stub\n")
    _patch_venv(
        monkeypatch, prefix=str(tmp_path / "venvs" / "coderag"), base_prefix="/usr"
    )
    monkeypatch.setattr(inst.shutil, "which", lambda *_a, **_k: str(shim))
    assert inst._server_invocation(None) == ("coderag", ["mcp"])


def test_invocation_module_fallback_when_no_script(monkeypatch):
    # Source checkout, no installed console script anywhere → run the module by interpreter.
    _patch_venv(monkeypatch, prefix="/usr", base_prefix="/usr")
    monkeypatch.setattr(inst.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(inst.shutil, "which", lambda *_a, **_k: None)
    cmd, args = inst._server_invocation(None)
    assert cmd == "/usr/bin/python3"
    assert args == ["-m", "coderag.surfaces.cli", "mcp"]
