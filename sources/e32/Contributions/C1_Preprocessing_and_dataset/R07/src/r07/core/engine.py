"""Bounded local API health gate. Never read datasets or submit fitting tasks."""
from importlib import import_module
from pathlib import Path
import uuid
from r07.core.artifacts import utc_now, write_json
from r07.core.runtime import configure_local_engine, REQUIRED

def check_local_engine(workspace, root):
    configure_local_engine(workspace, root)
    report = dict(started_utc=utc_now(), status="LOCAL_ENGINE_CHECK_STARTED",
                  required_versions=REQUIRED, model_fits=0, simulations=0,
                  endpoint_scope="127.0.0.1 loopback only")
    target = Path(workspace) / "environment" / ("engine_" + uuid.uuid4().hex + ".json")
    server = None
    try:
        # Importing all server routers catches the observed Body(min_items=...)
        # incompatibility, unlike pip check or numerical fixture tests alone.
        api = import_module("prefect.server.api.server")
        server = api.SubprocessASGIServer()
        server.start(timeout=60)
        httpx = import_module("httpx")
        with httpx.Client(trust_env=False, timeout=5) as client:
            response = client.get(server.api_url + "/health")
            response.raise_for_status()
            if response.json() is not True:
                raise RuntimeError("Unexpected local Prefect health response")
        report["status"] = "LOCAL_ENGINE_HEALTH_PASS"
    except BaseException as exc:
        report.update(status="LOCAL_ENGINE_CHECK_FAILED",
                      error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        try:
            if server is not None:
                server.stop()
        except BaseException as exc:
            report.update(status="LOCAL_ENGINE_CLEANUP_FAILED",
                          cleanup_error_type=type(exc).__name__, cleanup_error=str(exc))
            raise
        finally:
            report["finished_utc"] = utc_now()
            write_json(target, report)
    print("[ENGINE HEALTH PASS] Local API started, answered and stopped; fits=0; simulations=0.",
          flush=True)
    print("engine_report=" + str(target), flush=True)
    return report
