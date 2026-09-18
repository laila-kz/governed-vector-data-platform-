from typer.testing import CliRunner

from cli.gvpctl import app
from qdrant_client import QdrantClient

runner = CliRunner()


def test_status_command_renders_active_alias_and_model() -> None:
    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "vectors_live" in result.output
    assert "bge-small-en-v1.5" in result.output


def test_staleness_command_renders_risk_summary() -> None:
    result = runner.invoke(app, ["staleness"])

    assert result.exit_code == 0
    assert "active" in result.output.lower() or "stale" in result.output.lower()


def test_lineage_trace_command_outputs_provenance() -> None:
    result = runner.invoke(app, ["lineage", "trace", "vec_0001"])

    assert result.exit_code == 0
    assert "vec_0001" in result.output or "document" in result.output.lower()


def test_qdrant_client_matches_configured_server_api() -> None:
    client = QdrantClient(url="http://localhost:6333")
    try:
        assert client.get_collections().collections is not None
        assert hasattr(client, "search")
        assert hasattr(client, "close")
    finally:
        client.close()
