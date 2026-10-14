"""Actual high-level MCP tool registration contract in supported SDK 2.x."""
import asyncio

import pytest

from kaggle_gpu_mcp.server import create_server


@pytest.mark.skipif(pytest.importorskip("mcp", reason="MCP SDK not installed locally") is None,
                    reason="MCP SDK unavailable")
def test_tools_registered_through_real_mcp_sdk(settings):
    server = create_server(settings)
    tools = asyncio.run(server.list_tools())
    names = {tool.name for tool in tools}
    required = {
        "kaggle_check_connection", "resource_select", "kaggle_submit_job",
        "kaggle_workflow_start", "kaggle_get_job", "kaggle_list_jobs",
        "kaggle_get_artifacts", "kaggle_followup_experiment", "kaggle_merge_local_csv",
        "kaggle_register_dataset_source", "kaggle_list_dataset_sources",
        "kaggle_profile_local_csv", "kaggle_workflow_status",
        "kaggle_notification_inbox", "kaggle_retry_notification",
    }
    assert required <= names
    assert len(names) >= 23
    submission = next(t for t in tools if t.name == "kaggle_submit_job")
    assert "code_path" in submission.input_schema["properties"]
    assert "approved" in submission.input_schema["properties"]
