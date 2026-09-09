"""
CLI registration, mirroring oadr-cpep's tests/test_cli.py.

Cheap, but it catches the failure mode where a command is renamed or its import
breaks and nothing notices until a Nextflow process fails mid-run.
"""
import re

from typer.testing import CliRunner

from pvclust_py.cli import app

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(result) -> str:
    """The output with colour removed.

    Rich highlights option-like tokens, and it splits them: ``--matrix`` is emitted as
    a styled ``-`` followed by a styled ``-matrix``, so the literal substring is not in
    the raw text. Locally that never shows, because colour is off when stdout is not a
    terminal; GitHub Actions forces colour on, so an assertion against the raw output
    passes on a laptop and fails only in CI. Strip the escapes and the assertion means
    what it says on both.
    """
    return _ANSI.sub("", result.output)

EXPECTED_COMMANDS = [
    "cluster",
    "kmeans",
    "project-features",
    "project-stats",
    "count-edges",
    "apply-edges",
    "heatmap",
    "diagnose",
    "shared-features",
    "aggregate-trees",
]


def test_help_exits_zero():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_all_commands_registered():
    result = runner.invoke(app, ["--help"])
    for command in EXPECTED_COMMANDS:
        assert command in plain(result)


def test_subcommand_help():
    for command in EXPECTED_COMMANDS:
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, f"{command} --help failed"


def test_input_form_is_required():
    """Giving neither --matrix nor the SomaScan trio must fail with a useful message,
    not a traceback."""
    result = runner.invoke(app, ["cluster", "--project", "P"])
    assert result.exit_code != 0
    assert "--matrix" in plain(result)


def test_the_two_input_forms_are_mutually_exclusive():
    result = runner.invoke(app, ["cluster", "--project", "P", "--matrix", "a.csv",
                                 "--rfu", "b.txt"])
    assert result.exit_code != 0
