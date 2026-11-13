"""Smoke-test local HTTP launch flags and an in-process MCP tool call.

No Kaggle credentials, network calls, or GPU are needed.
"""
import asyncio
import sys
from unittest.mock import Mock

import pytest

from kaggle_gpu_mcp import cli
from kaggle_gpu_mcp.server import create_server


def test_serve_http_local_binds_only_loopback(monkeypatch):
    fake_server = Mock()
    monkeypatch.setattr("kaggle_gpu_mcp.server.create_server", lambda settings: fake_server)
    monkeypatch.setattr(sys, "argv", ["kaggle-gpu-mcp", "serve-http-local", "--port", "8123"])

    cli.main()

    fake_server.run.assert_called_once_with(
        transport="streamable-http",
        host="127.0.0.1",
        port=8123,
        json_response=True,
        stateless_http=True,
    )


@pytest.mark.parametrize("port", ["0", "80", "65536"])
def test_serve_http_local_rejects_invalid_ports(monkeypatch, port):
    fake_server = Mock()
    monkeypatch.setattr("kaggle_gpu_mcp.server.create_server", lambda settings: fake_server)
    monkeypatch.setattr(sys, "argv", ["kaggle-gpu-mcp", "serve-http-local", "--port", port])
    with pytest.raises(SystemExit) as ex:
        cli.main()
    assert ex.value.code == 2
    fake_server.run.assert_not_called()


def test_mcp_sdk_inprocess_client_roundtrip(settings):
    """Real MCP SDK client calls a safe read-only registered tool."""
    from mcp import Client

    async def exercise():
        async with Client(create_server(settings)) as client:
            result = await client.call_tool(
                "resource_select", {"mode": "cpu", "workload": "preprocess"}
            )
            assert not result.is_error
            assert result.content
            assert 'cpu' in str(result.content).lower()

    asyncio.run(exercise())
