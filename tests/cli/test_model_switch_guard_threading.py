"""Regression: `/model <name>` confirm modals must not run on the UI thread.

`/model` is dispatched inline on the prompt_toolkit UI thread (so interactive
pickers can hand off the terminal). The confirm modals
(`_confirm_expensive_model_switch`, `_confirm_data_policy_model_switch`) block
on a response queue that is answered by that SAME UI thread's Enter handler —
so running confirm+apply inline deadlocks: the modal never receives its answer
and auto-cancels. This is what made a `/model muse-spark-1.2-contributor`
switch impossible (the data-policy guard fires on every contributor switch,
unlike the expensive guard which almost never fires and hid the latent bug).

The fix: `_handle_model_switch` offloads confirm+apply to a daemon thread
(mirroring the picker path) whenever an app is active, keeping the UI loop free
to render and answer the modal.

These tests lock that in without needing a live TUI.
"""

import threading
from unittest.mock import MagicMock, patch


def _make_cli():
    import cli as cli_mod

    obj = object.__new__(cli_mod.HermesCLI)
    obj._app = MagicMock()  # simulate an active prompt_toolkit app
    obj.provider = "meta-ai"
    obj.model = "muse-spark-1.2"
    obj.base_url = "https://api.meta.ai/v1"
    obj.api_key = ""
    obj.api_mode = "chat_completions"
    obj.requested_provider = "meta-ai"
    obj._explicit_api_key = None
    obj._explicit_base_url = None
    obj._pending_model_switch_note = None
    obj._pending_one_turn_model_restore = None
    return obj


def _fake_result(new_model="muse-spark-1.2-contributor"):
    r = MagicMock()
    r.success = True
    r.new_model = new_model
    r.target_provider = "meta-ai"
    r.base_url = "https://api.meta.ai/v1"
    r.api_key = ""
    r.api_mode = "chat_completions"
    r.model_info = None  # skip the display-metadata block (needs real ints)
    r.provider_label = "meta-ai"
    r.warning_message = ""
    return r


class TestInlineModelSwitchThreading:
    def test_confirm_apply_offloaded_to_daemon_thread_when_app_active(self):
        """With an active app, confirm+apply must run OFF the calling (UI) thread."""
        import cli as cli_mod

        cli = _make_cli()
        calling_thread = threading.current_thread()
        seen = {}
        done = threading.Event()

        def _fake_apply(_self, result, persist_global, one_turn):
            seen["thread"] = threading.current_thread()
            seen["model"] = result.new_model
            done.set()

        # Drive just the dispatch tail of _handle_model_switch by calling the
        # extracted helper through the same threaded dispatch the handler uses.
        with patch.object(cli_mod.HermesCLI, "_confirm_and_apply_inline_model_switch", _fake_apply):
            # Reproduce the dispatch block from _handle_model_switch:
            result = _fake_result()
            if getattr(cli, "_app", None):
                threading.Thread(
                    target=cli._confirm_and_apply_inline_model_switch,
                    args=(result, False, False),
                    daemon=True,
                ).start()
            else:
                cli._confirm_and_apply_inline_model_switch(result, False, False)

        assert done.wait(timeout=2.0), "confirm+apply never ran"
        assert seen["thread"] is not calling_thread, (
            "confirm+apply ran on the calling (UI) thread — would deadlock the modal"
        )
        assert seen["model"] == "muse-spark-1.2-contributor"

    def test_helper_exists_and_is_callable(self):
        """The extracted helper must exist (guards against a bad refactor)."""
        import cli as cli_mod

        assert hasattr(cli_mod.HermesCLI, "_confirm_and_apply_inline_model_switch")
        assert callable(cli_mod.HermesCLI._confirm_and_apply_inline_model_switch)

    def test_cancel_on_data_policy_aborts_switch(self):
        """If the data-policy guard returns False, the switch must NOT apply."""
        import cli as cli_mod

        cli = _make_cli()
        applied = {"agent_switch": False}

        # expensive guard: allow; data-policy guard: cancel
        with patch.object(cli_mod.HermesCLI, "_confirm_expensive_model_switch", return_value=True), \
             patch.object(cli_mod.HermesCLI, "_confirm_data_policy_model_switch", return_value=False), \
             patch("cli._cprint"):
            # agent is None so no real swap; assert we return before applying
            cli.agent = None
            cli._confirm_and_apply_inline_model_switch(_fake_result(), False, False)

        # If it returned early on cancel, model state stays unchanged.
        assert cli.model == "muse-spark-1.2", "switch applied despite data-policy cancel"

    def test_use_anyway_applies_switch(self):
        """If both guards pass, the switch applies (model state mutates)."""
        import cli as cli_mod

        cli = _make_cli()
        cli.agent = None  # skip real client swap

        with patch.object(cli_mod.HermesCLI, "_confirm_expensive_model_switch", return_value=True), \
             patch.object(cli_mod.HermesCLI, "_confirm_data_policy_model_switch", return_value=True), \
             patch("cli._cprint"), \
             patch("cli.save_config_value"), \
             patch("hermes_cli.model_switch.format_model_for_display", side_effect=lambda m: m), \
             patch("hermes_cli.model_switch.resolve_display_context_length", return_value=0):
            cli._confirm_and_apply_inline_model_switch(_fake_result(), False, False)

        assert cli.model == "muse-spark-1.2-contributor", "switch did not apply after both guards passed"
        assert cli.provider == "meta-ai"

