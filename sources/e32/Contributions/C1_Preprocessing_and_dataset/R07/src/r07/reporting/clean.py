"""Pooled OOF evidence and publication-ready source table, no statistical overclaims."""
from pathlib import Path
import uuid
import numpy as np
import pandas as pd
from r07.core.artifacts import digest, sha256, utc_now, write_csv, write_json, publish_bytes
from r07.evaluation.metrics import score
from r07.workflows.units import verify_unit

def build_report(prepared):
    config = prepared["config"]
    root = Path(prepared["output"])
    output = root / "reports" / uuid.uuid4().hex
    output.mkdir(parents=True)
    summary, fold_rows, receipts = [], [], []
    for modality in config["modalities"]:
        for model in config["models"]:
            parts = []
            for fold in range(1, config["n_splits"] + 1):
                result = verify_unit(prepared, modality, model, fold)
                if result is None:
                    raise ValueError("Incomplete fold collection; no partial headline score")
                fold_rows.append(dict(modality=modality, model=model, fold=fold, **result["metrics"]))
                receipts.append(dict(file=result["receipt_path"], sha256=result["receipt_sha256"]))
                prediction = result["predictions"].copy()
                prediction["fold"] = fold
                parts.append(prediction)
            oof = pd.concat(parts).sort_values("Id").reset_index(drop=True)
            if oof["window_id"].tolist() != prepared["frame"]["window_id"].tolist():
                raise ValueError("OOF windows omitted, duplicated or mixed")
            if not (oof.groupby("acquisition_id")["fold"].nunique() == 1).all():
                raise ValueError("OOF event leakage")
            metrics, cm = score(oof["y_true"], oof["y_pred"])
            write_csv(output / f"oof_{modality}_{model}.csv", oof)
            write_csv(output / f"confusion_{modality}_{model}.csv", cm)
            summary.append(dict(modality=modality, model=model, **metrics))
    frame = pd.DataFrame(summary)
    write_csv(output / "clean_summary.csv", frame)
    write_csv(output / "fold_metrics.csv", pd.DataFrame(fold_rows))
    # Caption says exactly what was evaluated; inference remains a separate stage.
    lines = [r"\begin{table}[ht]", r"\centering",
             r"\caption{Exploratory clean grouped out-of-fold pilot on the regenerated reference dataset. No confidence intervals or physical qualification are implied.}",
             r"\begin{tabular}{llrrr}", r"\hline",
             r"Modality & Model & Macro F1 & Bal. accuracy & MCC \\", r"\hline"]
    for row in summary:
        label = row["model"].replace("_", r"\_")
        lines.append(f'{row["modality"]} & {label} & {row["macro_f1"]:.4f} & {row["balanced_accuracy"]:.4f} & {row["mcc"]:.4f} ' + r"\\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    publish_bytes(output / "clean_table.tex", ("\n".join(lines) + "\n").encode())
    limits = ["Engineering/clean baseline only, not corrected full-paper results.",
              "No confidence intervals, paired tests, noise, denoising, fusion or normalization sweep yet.",
              "Acquisition-grouped interpolation protocol, not held-out spatial/operating-condition generalization.",
              "Fourteen E16 voltage failures and electrical-model/segmentation/bandwidth concerns remain open.",
              "858 windows derive from 429 paired acquisitions; not 858 independent simulations.",
              "No new simulation; numeric bulk MAT files are not rehashed by the feature-table pilot."]
    report = dict(experiment_id="C1-E20", status="COMPLETE",
                  scientific_status="LOCAL_CLEAN_PILOT_COMPLETE_REVIEW_REQUIRED",
                  run_id=prepared["run_id"], created_utc=utc_now(),
                  acquisitions=prepared["identity"]["dataset"]["acquisitions"],
                  windows=len(prepared["frame"]), fitted_units=len(fold_rows),
                  publication_qualified=False, new_simulations=0,
                  inference_complete=False, limits=limits, task_receipts=receipts)
    report["lineage_snapshot"] = prepared["lineage_snapshot"]
    report["lineage_sha256"] = prepared["lineage_sha256"]
    write_json(output / "pilot_report.json", report)
    artifacts = {path.name: sha256(path) for path in output.iterdir() if path.is_file()}
    write_json(root / "latest_report.json",
               dict(relative_directory=output.relative_to(root).as_posix(), artifacts=artifacts,
                    report_source_sha256=sha256(Path(__file__))))
    print("[COMPLETE] Local clean pilot; scientific review remains required.", flush=True)
    print("report_directory=" + str(output), flush=True)
    return report
