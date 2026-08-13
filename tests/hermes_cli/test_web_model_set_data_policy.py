"""POST /api/model/set must surface the data-training-tier warning.

Mirrors the expensive-model confirm flow: selecting a contributor-tier model in
the desktop/web picker returns confirm_required (reason=data_policy) instead of
silently applying; re-sending with confirm_data_policy=True applies it; a
non-contributor model applies with no confirm.
"""

import asyncio

import pytest


def _set(model, provider="meta-ai", confirm_data_policy=False):
    from hermes_cli.web_models import ModelAssignment
    from hermes_cli import web_server

    body = ModelAssignment(
        scope="main",
        provider=provider,
        model=model,
        confirm_data_policy=confirm_data_policy,
        confirm_expensive_model=True,  # isolate the data-policy guard
    )
    return asyncio.run(web_server.set_model_assignment(body))


@pytest.fixture
def _cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("model:\n  default: gpt-4o\n  provider: openai\n")
    return tmp_path


def test_contributor_requires_confirmation(_cfg):
    r = _set("muse-spark-1.2-contributor")
    assert r["ok"] is False
    assert r["confirm_required"] is True
    assert r.get("confirm_reason") == "data_policy"
    assert "train" in r["confirm_message"].lower()


def test_contributor_applies_when_confirmed(_cfg):
    r = _set("muse-spark-1.2-contributor", confirm_data_policy=True)
    assert r.get("ok") is True
    assert not r.get("confirm_required")


def test_non_contributor_applies_without_confirmation(_cfg):
    r = _set("muse-spark-1.2")
    assert r.get("ok") is True
    assert not r.get("confirm_required")


def test_model_assignment_has_confirm_data_policy_field():
    from hermes_cli.web_models import ModelAssignment

    m = ModelAssignment(scope="main", provider="meta-ai", model="x")
    assert m.confirm_data_policy is False  # opt-in, defaults off
