"""Exercise the canonical shell runner's interpreter selection in isolated venvs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import venv

import pytest


ROOT = Path(__file__).resolve().parents[1]
CASES = ("default", "fallback-runtime", "fallback-env", "explicit", "relative",
         "missing", "no-pytest", "missing-value")


def _check_python_selection(tmp_path: Path, case: str) -> None:
    root = tmp_path / "checkout with spaces"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/run_tests.sh", scripts / "run_tests.sh")
    # Keep the driver observable without launching another full pytest suite.
    (scripts / "run_tests_parallel.py").write_text(
        "import json, os, sys\n"
        "print('PROBE=' + json.dumps({'prefix': sys.prefix, 'args': sys.argv[1:], "
        "'credential_present': 'OPENAI_API_KEY' in os.environ}))\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "scripts/run_tests_parallel.py"], check=True)

    def make_python(path: Path, *, pytest_available: bool = True) -> Path:
        venv.EnvBuilder(with_pip=False).create(path)
        python = path / "bin/python"
        if pytest_available:
            site = subprocess.check_output(
                [str(python), "-I", "-c", "import site; print(site.getsitepackages()[0])"],
                text=True,
            ).strip()
            (Path(site) / "pytest.py").write_text("", encoding="utf-8")
        return python

    dev = make_python(root / ".venv", pytest_available=not case.startswith("fallback-"))
    runtime = make_python(root / "venv", pytest_available=case != "fallback-env")
    fallback = dev
    args = ["-j", "1", "tests/fixture.py", "-q"]
    expected = dev.parent.parent
    if case == "fallback-runtime":
        expected = runtime.parent.parent
    elif case == "fallback-env":
        fallback = make_python(tmp_path / "external development venv")
        expected = fallback.parent.parent
    elif case == "explicit":
        args = ["--python", str(runtime), *args]
        expected = runtime.parent.parent
    elif case == "relative":
        args = ["--python", "checkout with spaces/venv/bin/python", *args]
        expected = runtime.parent.parent
    elif case == "missing":
        args = ["--python", str(tmp_path / "missing-python"), *args]
    elif case == "no-pytest":
        args = ["--python", str(make_python(tmp_path / "without-pytest", pytest_available=False)), *args]
    elif case == "missing-value":
        args = ["--python"]
    home = tmp_path / "home"
    home.mkdir()
    result = subprocess.run(
        ["bash", str(scripts / "run_tests.sh"), *args], cwd=tmp_path,
        env={**os.environ, "HOME": str(home), "HERMES_PYTHON": str(fallback),
             "OPENAI_API_KEY": "fixture-must-be-cleared"},
        capture_output=True, text=True, timeout=30,
    )
    probe_lines = [line[6:] for line in result.stdout.splitlines() if line.startswith("PROBE=")]
    if case in ("missing", "no-pytest", "missing-value"):
        assert result.returncode != 0, result.stdout
        assert not probe_lines, "an invalid explicit Python must never fall back"
        assert "--python" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    assert len(probe_lines) == 1, result.stdout
    probe = json.loads(probe_lines[0])
    assert Path(probe["prefix"]).resolve() == expected.resolve()
    assert probe["args"] == ["-j", "1", "tests/fixture.py", "-q"]
    assert probe["credential_present"] is False


@pytest.mark.linux_only
@pytest.mark.parametrize("case", CASES)
def test_linux_python_selection(tmp_path: Path, case: str) -> None:
    _check_python_selection(tmp_path, case)


@pytest.mark.macos_only
@pytest.mark.parametrize("case", CASES)
def test_macos_python_selection(tmp_path: Path, case: str) -> None:
    _check_python_selection(tmp_path, case)
