"""MCP tool surface for the persistent Kaggle orchestration service.

Use `kaggle-gpu-mcp serve` for stdio. Start the worker separately.
"""
from .config import Settings
from .demo import install_demo
from .service import Service


def create_server(settings: Settings | None = None):
    # Import lazily so the core and its offline tests work without an MCP installation.
    from mcp.server import MCPServer

    settings = settings or Settings.from_env()
    service = Service(settings)
    mcp = MCPServer("Kaggle GPU MCP")

    @mcp.tool()
    def kaggle_check_connection() -> dict:
        """Verify Kaggle CLI version and Kaggle account access."""
        return service.provider.check()

    @mcp.tool()
    def resource_select(mode: str = "auto", workload: str = "pytorch") -> dict:
        """Recommend CPU or GPU based on job preference and workload."""
        return service.select_resource(mode, workload)

    @mcp.tool()
    def demo_install() -> dict:
        """Create editable synthetic CSV preparation and GPU training demo scripts."""
        return install_demo(settings)

    @mcp.tool()
    def kaggle_submit_job(code_path: str, approved: bool = False,
                          mode: str = "auto", workload: str = "pytorch",
                          dataset_sources: list[str] | None = None,
                          kernel_sources: list[str] | None = None,
                          kind: str = "train", require_contract: bool = True) -> dict:
        """Queue an approved Python or Jupyter notebook in the workspace for Kaggle."""
        return service.submit(code_path, mode, workload, dataset_sources, kernel_sources,
                              kind, approved, require_contract=require_contract)

    @mcp.tool()
    def kaggle_workflow_start(prep_code: str, train_code: str,
                              dataset_sources: list[str] | None = None,
                              approved: bool = False) -> dict:
        """Queue CPU data preparation and dependent GPU training as one workflow."""
        return service.workflow(prep_code, train_code, dataset_sources, approved)

    @mcp.tool()
    def kaggle_get_job(job_id: str) -> dict:
        """Retrieve persistent job state, audit events, metrics and artifacts."""
        return service.details(job_id)

    @mcp.tool()
    def kaggle_list_jobs(limit: int = 25) -> list[dict]:
        """List saved jobs, newest first."""
        return service.db.list(limit)

    @mcp.tool()
    def kaggle_get_logs(job_id: str, max_chars: int = 12000) -> str:
        """Get last downloaded Kaggle execution logs for an approved job."""
        return service.logs(job_id, max_chars)

    @mcp.tool()
    def kaggle_get_artifacts(job_id: str) -> dict:
        """List downloaded artifacts and metrics; local paths only, not file contents."""
        return service.details(job_id).get("artifacts", {"status": "not_downloaded"})

    @mcp.tool()
    def kaggle_publish_dataset(directory: str, slug: str, approved: bool = False,
                                license_name: str = "", version_note: str = "") -> dict:
        """Publish private dataset from approved workspace files; version_note updates existing."""
        return service.publish_dataset(directory, slug, license_name, version_note, approved)

    @mcp.tool()
    def kaggle_job_events(job_id: str) -> list[dict]:
        """View timestamped state changes for a job."""
        return service.db.events(job_id)

    @mcp.tool()
    def kaggle_reconcile_job(job_id: str) -> dict:
        """Check Kaggle after an uncertain submission without duplicating GPU jobs."""
        return service.reconcile(job_id)

    @mcp.tool()
    def kaggle_refresh_logs(job_id: str) -> str:
        """Fetch current logs for a submitted job (availability depends on Kaggle)."""
        return service.refresh_logs(job_id)

    @mcp.tool()
    def kaggle_compare_jobs(job_ids: list[str]) -> dict:
        """Compare model metrics across completed experiments."""
        return service.compare(job_ids)

    @mcp.tool()
    def kaggle_followup_experiment(previous_job_id: str, code_path: str,
                                    approved: bool = False, mode: str = "auto",
                                    workload: str = "pytorch") -> dict:
        """Submit an approved improved training program with prior outputs attached."""
        return service.followup(previous_job_id, code_path, approved, mode, workload)

    @mcp.tool()
    def kaggle_merge_local_csv(paths: list[str], output_name: str = "combined.csv") -> dict:
        """Deduplicate and merge same-schema CSV files within the workspace."""
        return service.merge_csv(paths, output_name)

    @mcp.tool()
    def kaggle_test_notification(approved: bool = False) -> dict:
        """Send a manual SMTP test to the configured notification address."""
        if not approved:
            raise ValueError("Sending a test email requires approved=true")
        from .notify import Mailer
        mailer = Mailer(settings)
        if not mailer.enabled():
            return {"configured": False, "sent": False}
        sent = mailer.send("Kaggle GPU MCP SMTP test", "The email notification channel is connected.")
        return {"configured": True, "sent": sent}

    return mcp
