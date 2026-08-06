from __future__ import annotations

from benchmarks.cli import main


def test_list_command_describes_native_cases(capsys) -> None:
    assert main(["list"]) == 0

    output = capsys.readouterr().out
    assert "accounting.fft\tdevices=cpu\tpresets=smoke,reference" in output
    assert "training.dpsgd\tdevices=cuda\tpresets=smoke,reference" in output


def test_run_command_rejects_unknown_preset(capsys) -> None:
    assert main(["run", "optimizers.state", "--preset", "missing"]) == 2

    assert "Unknown preset" in capsys.readouterr().err
