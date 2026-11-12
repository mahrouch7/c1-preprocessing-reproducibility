"""Prefect local-first adapter. Explicit zero retries and no implicit pickle cache."""
from prefect import flow, task
from r07.reporting.clean import build_report
from r07.workflows.preparation import prepare
from r07.workflows.units import execute_unit

@task(name="validate-dataset-and-freeze-folds", cache_policy=None, persist_result=False, retries=0)
def prepare_task(*args):
    return prepare(*args)

@task(name="fit-one-grouped-fold", cache_policy=None, persist_result=False, retries=0)
def fit_task(prepared, modality, model, fold):
    result = execute_unit(prepared, modality, model, fold)
    return {"unit": result["name"], "receipt_sha256": result["receipt_sha256"]}

@task(name="assemble-verified-oof-evidence", cache_policy=None, persist_result=False, retries=0)
def report_task(prepared):
    return build_report(prepared)

@flow(name="C1-E20-R07-local-clean-pilot", retries=0, persist_result=False, log_prints=True)
def clean_pilot(bundle, registry, config, root, repo, workspace, environment_sha):
    prepared = prepare_task(bundle, registry, config, root, repo, workspace, environment_sha)
    for modality in prepared["config"]["modalities"]:
        for model in prepared["config"]["models"]:
            for fold in range(1, prepared["config"]["n_splits"]+1):
                fit_task(prepared, modality, model, fold)
    return report_task(prepared)
