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
