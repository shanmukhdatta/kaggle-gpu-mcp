from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from kaggle_gpu_mcp.provider import KaggleCLI, KaggleError


def test_kaggle_provider_invokes_expected_commands_without_shell(settings):
    command_history = []

    def recorder(args, **kwargs):
        command_history.append((args, kwargs))
        return CompletedProcess(args, 0, stdout="ok", stderr="")

    adapter = KaggleCLI(settings)
    with patch("kaggle_gpu_mcp.provider.subprocess.run", side_effect=recorder):
        assert adapter.push(settings.workspace, "NvidiaTeslaT4") == "ok"
        assert adapter.status("owner/kernel-slug") == "ok"
        assert adapter.logs("owner/kernel-slug") == "ok"
        assert adapter.output("owner/kernel-slug", settings.workspace / "outputs") == "ok"
        assert adapter.dataset_create(settings.workspace) == "ok"
    assert command_history[0][0][-2:] == ["--accelerator", "NvidiaTeslaT4"]
    assert command_history[0][0][:3] == [settings.kaggle_bin, "kernels", "push"]
    assert all(kwargs.get("check") is False for _, kwargs in command_history)
    assert all("shell" not in kwargs for _, kwargs in command_history)
    assert command_history[-1][0][:3] == [settings.kaggle_bin, "datasets", "create"]
    with pytest.raises(ValueError):
        adapter.status("--dangerous")


def test_kaggle_errors_redact_stderr(settings):
    secret = "MY_PASSWORD_DO_NOT_LEAK"
    adapter = KaggleCLI(settings)
    with patch("kaggle_gpu_mcp.provider.subprocess.run",
               return_value=CompletedProcess(["kaggle"], 1, stdout="", stderr=secret)):
        with pytest.raises(KaggleError) as caught:
            adapter.status("owner/kernel-slug")
    assert secret not in str(caught.value)
