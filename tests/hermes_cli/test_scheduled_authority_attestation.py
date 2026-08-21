"""Core contract for plugin-attested scheduled turn authority."""

from hermes_cli import plugins


def test_exact_plugin_attestation_accepts_defensive_job_copy(monkeypatch):
    job = {"id": "e28031e0acf8", "prompt": "tracked", "schedule": {"kind": "cron"}}
    monkeypatch.setattr(plugins, "discover_plugins", lambda: None)

    def attest(hook_name, **kwargs):
        assert hook_name == "attest_scheduled_turn"
        kwargs["job"]["schedule"]["kind"] = "mutated"
        return [{
            "schema_version": 1,
            "authority": "scheduled",
            "job_id": "e28031e0acf8",
        }]

    monkeypatch.setattr(plugins, "invoke_hook", attest)
    assert plugins.scheduled_turn_authority_attested(
        job, fire_provenance="scheduled_due", has_extra_prompt=False,
    ) is True
    assert job["schedule"]["kind"] == "cron"


def test_inexact_or_boolean_plugin_attestation_fails_closed(monkeypatch):
    monkeypatch.setattr(plugins, "discover_plugins", lambda: None)
    monkeypatch.setattr(
        plugins,
        "invoke_hook",
        lambda *_args, **_kwargs: [True, {"authority": "scheduled"}],
    )
    assert plugins.scheduled_turn_authority_attested(
        {"id": "other"}, fire_provenance="scheduled_due", has_extra_prompt=False,
    ) is False


def test_manual_or_prompt_augmented_fire_cannot_be_attested(monkeypatch):
    monkeypatch.setattr(plugins, "discover_plugins", lambda: None)
    invoked = []
    monkeypatch.setattr(plugins, "invoke_hook", lambda *args, **kwargs: invoked.append(kwargs))
    job = {"id": "e28031e0acf8"}

    assert plugins.scheduled_turn_authority_attested(
        job, fire_provenance="manual", has_extra_prompt=False,
    ) is False
    assert plugins.scheduled_turn_authority_attested(
        job, fire_provenance="scheduled_due", has_extra_prompt=True,
    ) is False
    assert invoked == []


def test_direct_user_manual_attestation_binds_exact_turn_and_job(monkeypatch):
    job = {"id": "acf87062f340", "prompt": "tracked"}
    monkeypatch.setattr(plugins, "discover_plugins", lambda: None)

    def attest(hook_name, **kwargs):
        assert hook_name == "attest_manual_scheduled_turn"
        return [{
            "schema_version": 1,
            "authority": "manual_direct_user",
            "job_id": kwargs["job"]["id"],
            "session_id": kwargs["session_id"],
            "task_id": kwargs["task_id"],
            "turn_id": kwargs["turn_id"],
            "revision": kwargs["direct_user_authority_revision"],
        }]

    monkeypatch.setattr(plugins, "invoke_hook", attest)
    assert plugins.manual_scheduled_turn_authority_attested(
        job,
        session_id="session-1",
        task_id="task-1",
        turn_id="turn-1",
        direct_user_authority_revision=0,
        direct_user_authority_kind="direct_user",
        has_extra_prompt=False,
    ) is True


def test_manual_attestation_rejects_untrusted_or_augmented_fire(monkeypatch):
    monkeypatch.setattr(plugins, "discover_plugins", lambda: None)
    invoked = []
    monkeypatch.setattr(plugins, "invoke_hook", lambda *args, **kwargs: invoked.append(kwargs))
    base = dict(
        job={"id": "acf87062f340"},
        session_id="session-1",
        task_id="task-1",
        turn_id="turn-1",
        direct_user_authority_revision=0,
    )

    assert plugins.manual_scheduled_turn_authority_attested(
        **base,
        direct_user_authority_kind="untrusted",
        has_extra_prompt=False,
    ) is False
    assert plugins.manual_scheduled_turn_authority_attested(
        **base,
        direct_user_authority_kind="direct_user",
        has_extra_prompt=True,
    ) is False
    assert invoked == []
