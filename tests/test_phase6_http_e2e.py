"""HTTP-over-loopback MCP smoke test with a real subprocess and real client.

This test deliberately does not use Kaggle credentials or submit any remote jobs.
"""
import asyncio
import os
import socket
import subprocess
import sys
import time

from mcp import Client


def test_local_streamable_http_roundtrip(tmp_path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    env = os.environ.copy()
    env["KGM_WORKSPACE"] = str(tmp_path / "workspace")
    env["KGM_KAGGLE_USERNAME"] = "testowner"
    env["KGM_ALLOW_SUBMISSIONS"] = "0"

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "kaggle_gpu_mcp.cli",
            "serve-http-local",
            "--port",
            str(port),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        ready = False
        for _ in range(150):
            if process.poll() is not None:
                error = process.stderr.read()[-3000:]
                raise AssertionError(f"HTTP MCP server quit during startup: {error}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    ready = True
                    break
            except OSError:
                time.sleep(0.1)
        assert ready, "HTTP MCP server did not bind the loopback port"

        async def exercise():
            async with Client(f"http://127.0.0.1:{port}/mcp") as client:
                available = await client.list_tools()
                assert any(tool.name == "resource_select" for tool in available.tools)
                outcome = await client.call_tool(
                    "resource_select", {"mode": "cpu", "workload": "preprocess"}
                )
                assert not outcome.is_error
                assert outcome.content

        asyncio.run(exercise())
    finally:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)
