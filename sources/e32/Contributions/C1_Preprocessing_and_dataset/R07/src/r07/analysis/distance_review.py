"""C1-E22 saved-only signal/distance review. Never calls fit, sim or a server."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import traceback
import uuid
import numpy as np
import pandas as pd
from r07.core.artifacts import child, digest, exclusive_lock, publish_bytes, read_json, sha256, utc_now, write_csv, write_json
from r07.data.views import load_view, read_csv, LINES as VIEW_LINES
from r07.features.profiles import LABELS, PROFILES, RMS
from r07.analysis.core import LINES, BINS, require, unique, fault_distance, distance_bin, relative_change, differences
from r07.analysis.figures import response_figure, error_figure, html_report

class Sources:
    def __init__(self, roots):
        self.roots = {k:Path(v).resolve() for k,v in roots.items()}
        self.files = {}
    def record(self,path,expected=None):
        path=Path(path).resolve()
        value=sha256(path)
        require(expected is None or value==expected,"Saved source hash changed: "+str(path))
        choices=[(len(str(root)),alias,path.relative_to(root).as_posix())
                 for alias,root in self.roots.items() if path.is_relative_to(root)]
        require(bool(choices),"Unregistered source root")
        _,alias,relative=max(choices)
        key=alias+"/"+relative
        require(key not in self.files or self.files[key]==value,"Source changed during review")
        self.files[key]=value
        return path
    def json(self,path,expected=None):
        return read_json(self.record(path,expected))
    def unchanged(self):
        for key,value in self.files.items():
            alias,relative=key.split("/",1)
            require(sha256(child(self.roots[alias],relative))==value,"Source changed before publication")

def check_lineage(package):
    spec=read_json(package/"provenance/distance_analysis_lineage.json")
    for item in spec["files"]:
        require(sha256(child(package,item["path"]))==item["sha256"],"Analysis/ancestor lineage changed")
    return spec

def load_cases(dataset, sources):
    require([(line,length) for line,_,_,length in LINES]==VIEW_LINES,
            'Distance graph and frozen view matrix differ')
    rows=[];cache={}
    line_map={r[0]:r for r in LINES}
    for link in dataset.manifest["case_links"]:
        receipt_path=child(dataset.bundle,link["case_receipt_relative_path"])
        receipt=sources.json(receipt_path,link["case_receipt_sha256"])
        cfg=sources.json(receipt_path.parent/"input_config.json",receipt["config_sha256"])
        c=receipt["case"];s=cfg["spatial_contract"];scenario=cfg["scenario"]
        line,u,v,length=line_map[link["line"]]
        require(c["id"]==scenario["id"]==link["acquisition_id"],"Configuration case identity mismatch")
        require((s["line"],s["ref_bus"],s["other_bus"],s["line_length_km"],s["position_km"])==
                (line,u,v,length,link["position_km"]),"Spatial coordinate mismatch")
        require((c["ref_bus"],c["other_bus"],c["line_length_km"],c["position_km"])==
                (u,v,length,link["position_km"]),"Receipt coordinate mismatch")
        internal_from=cfg["fault"]["branch_from"];internal_to=cfg["fault"]["branch_to"]
        mapping=cfg["internal_to_paper_bus"]
        a,b=mapping[internal_from-1],mapping[internal_to-1]
        require({a,b}=={u,v},"Internal/paper endpoint mapping mismatch")
        position=float(link["position_km"])
        expected_fraction=position/length if a==u else 1-position/length
        require(math.isclose(cfg["fault"]["electrical_fraction_from"],expected_fraction,rel_tol=0,abs_tol=1e-14),
                "Electrical fraction is not the oriented local coordinate")
        require(math.isclose(s["internal_fraction_from"],expected_fraction,rel_tol=0,abs_tol=1e-14),
                "Spatial/config fraction mismatch")
        distance=fault_distance(line,position)
        row=dict(acquisition_id=link["acquisition_id"],line=line,position_km=position,
                 local_reference_bus=u,other_bus=v,line_length_km=length,fault=link["fault"],
                 measurement_route="B7B5",distance_reference_bus=7,distance_B7_km=distance,
                 distance_bin_km=distance_bin(distance))
        rows.append(row);cache[row["acquisition_id"]]=(receipt_path.parent,receipt,cfg)
    require(len(rows)==429 and len(unique(rows,"acquisition_id"))==429,"Expected429unique fault acquisitions")
    return rows,cache

def signal_effects(frame, cases):
    windows=unique(frame.to_dict("records"),"window_id")
    result=[]
    for c in cases:
        before=windows[c["acquisition_id"]+"__prefault"]
        during=windows[c["acquisition_id"]+"__fault"]
        require(before["fault"]=="noFault" and during["fault"]==c["fault"],"Paired label mismatch")
        for name in RMS:
            a,b=float(before[name]),float(during[name])
            result.append(dict(**c,channel=name,prefault_rms=a,fault_rms=b,
                               absolute_rms_change=b-a,relative_rms_change=relative_change(a,b)))
    require(len(result)==2574,"Missing phase effects")
    return result

def validate_predictions(rows, frame, folds):
    actual=unique(rows,"window_id");expected=unique(frame.to_dict("records"),"window_id")
    fold_map=unique(folds.to_dict("records"),"window_id")
    require(set(actual)==set(expected)==set(fold_map),"Prediction window coverage differs")
    for key,e in expected.items():
        r=actual[key];f=fold_map[key]
        require((r["Id"],r["acquisition_id"],r["y_true"])==(e["Id"],e["acquisition_id"],e["fault"]),
                "Prediction joined to wrong acquisition or label")
        require((f["Id"],f["acquisition_id"],f["fault"])==(e["Id"],e["acquisition_id"],e["fault"]),
                "Fold joined to wrong window")
        require(int(r["fold"])==int(f["fold"]) and int(r["fold"]) in range(1,6),"Wrong saved fold")
        require(r["y_pred"] in LABELS,"Unknown predicted class")
        probs=[float(r["p_"+label]) for label in LABELS]
        require(all(math.isfinite(x) and 0<=x<=1 for x in probs) and
                math.isclose(sum(probs),1,rel_tol=0,abs_tol=1e-10),"Invalid saved probability")
        require(probs[LABELS.index(r["y_pred"])]>=max(probs)-1e-12,"Prediction inconsistent with probabilities")
    groups={}
    for r in rows:
        groups.setdefault(r["acquisition_id"],set()).add(int(r["fold"]))
    require(all(len(v)==1 for v in groups.values()),"Source acquisition crosses folds")
    return actual

def load_predictions(dataset, run, config, sources):
    manifest=sources.json(run/"run_manifest.json",config["run_manifest_sha256"])
    require(manifest["run_id"]==config["run_id"] and
            manifest["identity"]["dataset"]["manifest_sha256"]==dataset.identity["manifest_sha256"],
            "Run and dataset differ")
    folds=read_csv(sources.record(run/"fold_manifest.csv",manifest["fold_file_sha256"]))
    pointer=sources.json(run/"latest_report.json",config["report_pointer_sha256"])
    require(pointer["relative_directory"]==config["report_directory"],"Wrong saved report directory")
    report_root=child(run,pointer["relative_directory"])
    for name,h in pointer["artifacts"].items():
        sources.record(child(report_root,name),h)
    report=sources.json(report_root/"pilot_report.json",config["report_sha256"])
    require(report["status"]=="COMPLETE" and report["run_id"]==config["run_id"],"Incomplete/wrong pilot")
    receipts={r["file"]:r["sha256"] for r in report["task_receipts"]}
    all_predictions={}
    for modality in PROFILES:
        rows=read_csv(report_root/f"oof_{modality}_logistic_regression.csv").to_dict("records")
        pooled=validate_predictions(rows,dataset.frame,folds)
        task_rows=[]
        for fold in range(1,6):
            relative=f"tasks/{modality}__logistic_regression__fold{fold:02d}/complete.json"
            receipt=sources.json(child(run,relative),receipts[relative])
            task_name=f"{modality}__logistic_regression__fold{fold:02d}"
            require(receipt["status"]=="VALIDATED" and
                    receipt["identity"]==digest(dict(run_id=config["run_id"],task=task_name)),
                    "Invalid saved task identity")
            attempt=child(child(run,f"tasks/{task_name}"),receipt["attempt"])
            for name,h in receipt["artifacts"].items():
                sources.record(child(attempt,name),h)
            raw=read_csv(attempt/"predictions.csv").to_dict("records")
            for row in raw:
                row["fold"]=fold;task_rows.append(row)
        tasks=validate_predictions(task_rows,dataset.frame,folds)
        for key,r in pooled.items():
            t=tasks[key]
            require(all(r[k]==t[k] for k in ["Id","acquisition_id","y_true","y_pred","fold"]),
                    "Pooled decisions differ from original folds")
            require(all(math.isclose(float(r["p_"+k]),float(t["p_"+k]),rel_tol=1e-12,abs_tol=1e-14)
                        for k in LABELS),"Pooled probabilities differ from saved folds")
        all_predictions[modality]=rows
    return all_predictions,manifest

def classified_faults(predictions, cases):
    metadata=unique(cases,"acquisition_id");result=[]
    for modality,rows in predictions.items():
        faults=[r for r in rows if r["y_true"]!="noFault"]
        require(len(faults)==429,"Wrong fault-only prediction denominator")
        for r in faults:
            c=metadata[r["acquisition_id"]]
            require(c["fault"]==r["y_true"],"Spatial join changed true fault label")
            result.append(dict(**c,modality=modality,y_true=r["y_true"],y_pred=r["y_pred"],
                               fold=int(r["fold"]),type_error=int(r["y_true"]!=r["y_pred"]),
                               missed_fault=int(r["y_pred"]=="noFault")))
    return result

def grouped_errors(rows):
    by_bin=[];by_type_line=[];by_type_distance=[]
    for modality in PROFILES:
        for lo,hi in zip(BINS,BINS[1:]):
            label=f"[{lo},{hi})"
            selected=[r for r in rows if r["modality"]==modality and r["distance_bin_km"]==label]
            n=len(selected);errors=sum(r["type_error"] for r in selected);missed=sum(r["missed_fault"] for r in selected)
            by_bin.append(dict(modality=modality,distance_bin_km=label,fault_windows=n,
                               type_errors=errors,missed_faults=missed,
                               type_error_fraction=errors/n if n else None,
                               missed_fault_fraction=missed/n if n else None))
        for fault in LABELS[1:]:
            for lo,hi in zip(BINS,BINS[1:]):
                label=f'[{lo},{hi})'
                selected=[r for r in rows if (r['modality'],r['fault'],r['distance_bin_km'])==(modality,fault,label)]
                n=len(selected);errors=sum(r['type_error'] for r in selected)
                by_type_distance.append(dict(modality=modality,fault=fault,distance_bin_km=label,
                    fault_windows=n,type_errors=errors,type_error_fraction=errors/n if n else None,
                    missed_faults=sum(r['missed_fault'] for r in selected)))
            for line,_,_,_ in LINES:
                selected=[r for r in rows if (r["modality"],r["fault"],r["line"])==(modality,fault,line)]
                n=len(selected)
                by_type_line.append(dict(modality=modality,fault=fault,line=line,fault_windows=n,
                                         type_errors=sum(r["type_error"] for r in selected),
                                         missed_faults=sum(r["missed_fault"] for r in selected)))
    require(all(sum(r["fault_windows"] for r in by_bin if r["modality"]==m)==429 for m in PROFILES),
            "Distance-bin counts do not reconcile")
    return by_bin,by_type_line,by_type_distance

def normalized(value, model):
    if isinstance(value,str):
        return value.replace(model+"/","MODEL/") if value!=model else "MODEL"
    if isinstance(value,list):
        return [normalized(x,model) for x in value]
    if isinstance(value,dict):
        return {k:normalized(v,model) for k,v in value.items()}
    return value

def native_evidence_root(receipt, bundle, repo):
    roots={'bundle':Path(bundle), 'repo':Path(repo)}
    require(receipt.get('native_source_root') in roots, 'Unknown native evidence root')
    return child(roots[receipt['native_source_root']],receipt['native_relative_path'])

def audit_pair(left, right):
    """Saved mask consistency is not certification of topology or fault physics."""
    lc,la,lb=left;rc,ra,rb=right;issues=[]
    allowed={"fault.type","scenario.id","scenario.fault_type","scenario.ground",
             "scenario.effective_ground_resistance_ohm"}
    config_diff=differences(lc,rc)
    extra=[p for p in config_diff if p not in allowed]
    if extra:issues.append("UNEXPECTED_CONFIG_DIFFERENCES: "+", ".join(extra))
    for cfg,audit,ground in [(lc,la,0),(rc,ra,1)]:
        require(cfg["scenario"]["ground"]==ground and cfg["scenario"]["phases"]==[1,1,1],
                "ABC/ABCG configuration labels disagree")
        expected_type="lll" if ground==0 else "lllg"
        require(cfg["fault"]["type"]==cfg["scenario"]["fault_type"]==expected_type,"ABC fault type mismatch")
        model=audit["model_name"]
        blocks={item["path"].split("/",1)[1]:item["mask_values"] for item in audit["runtime_block_audit"]}
        candidates=[b for b in blocks.values() if all(k in b for k in ("FaultA","FaultB","FaultC","GroundFault"))]
        if len(candidates)!=1:
            issues.append("EXPECTED_ONE_FAULT_MASK");continue
        mask=candidates[0]
        for key in ("FaultA","FaultB","FaultC"):
            if mask[key]!="on":issues.append("PHASE_MASK_MISMATCH:"+key)
        if mask["GroundFault"]!=("on" if ground else "off"):issues.append("GROUND_FLAG_MISMATCH")
        times=[float(x) for x in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?",mask["SwitchTimes"])]
        if len(times)!=2 or not all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(times,[.2,.4])):
            issues.append("SWITCH_TIMING_MISMATCH")
        for key,expected in [("FaultResistance",cfg["fault"]["phase_breaker_resistance_ohm"]),
                             ("GroundResistance",cfg["fault"]["ground_resistance_ohm"]),
                             ("SnubberResistance",cfg["fault"]["snubber_resistance_ohm"])]:
            if not math.isclose(float(mask[key]),expected,rel_tol=1e-12):issues.append("MASK_PARAMETER_MISMATCH:"+key)
        if mask["External"]!="off" or mask["InitialStates"]!="0":issues.append("FAULT_CONTROL_MISMATCH")
        bindings=[normalized(b,model) for b in (lb if ground==0 else rb)["candidate_log_bindings"]
                  if b["group"] in ("IB7B5","VB7B5")]
        if {b["group"] for b in bindings}!={"IB7B5","VB7B5"} or len(bindings)!=2:
            issues.append("B7B5_BINDINGS_MISSING")
    def masks(a):
        return {b["path"].split("/",1)[1]:b["mask_values"] for b in a["runtime_block_audit"]}
    mask_diff=differences(masks(la),masks(ra))
    unexpected=[p for p in mask_diff if not p.endswith(".GroundFault")]
    if unexpected:issues.append("UNEXPECTED_RUNTIME_MASK_DIFFERENCES: "+", ".join(unexpected))
    routes=[]
    for build,audit in [(lb,la),(rb,ra)]:
        routes.append(sorted([normalized(b,audit["model_name"]) for b in build["candidate_log_bindings"]
                              if b["group"] in ("IB7B5","VB7B5")],key=lambda b:b["group"]))
    if routes[0]!=routes[1]:issues.append("B7B5_ROUTE_DIFFERENCE")
    if la["library_inventory"]!=ra["library_inventory"]:issues.append("LIBRARY_VERSION_DIFFERENCE")
    return dict(issues=issues,config_differences=config_diff,runtime_mask_differences=mask_diff,
                grounding_control="GroundFault off/on checked as configuration only",
                clearing_state="scheduled reference; independently observed contacts unavailable",
                physical_validation_established=False)

def configuration_audit(cases,cache,sources,frame):
    by_position={}
    feature=unique(frame.to_dict("records"),"window_id")
    for c in cases:
        if c["fault"] in ("abc","abcg"):
            by_position.setdefault((c["line"],c["position_km"]),{})[c["fault"]]=c
    require(len(by_position)==39,"Expected39ABC/ABCGlocations")
    result=[]
    for key,pair in by_position.items():
        require(set(pair)=={"abc","abcg"},"Missing paired ABC/ABCG acquisition")
        inputs=[];native_issues=[]
        for label in ("abc","abcg"):
            c=pair[label];root,receipt,cfg=cache[c["acquisition_id"]]
            native=native_evidence_root(receipt,sources.roots['DATASET_ROOT'],sources.roots['REPOSITORY_ROOT'])
            manifest=sources.json(native/"acquisition_manifest.json",receipt["native_manifest_sha256"])
            require(manifest['status']=='COMPLETE','Incomplete native evidence')
            native_cfg=sources.json(native/'input_config.json',manifest['config_sha256'])
            if receipt['native_source_root']=='bundle':
                require(manifest['config_sha256']==receipt['config_sha256'],'Wrong native configuration')
            else:
                reuse=sources.json(root/'reuse_receipt.json')
                require(reuse['status']=='REUSED_COMPLETE' and reuse['new_simulations']==0 and
                        reuse['source_manifest_sha256']==receipt['native_manifest_sha256'] and
                        reuse['plan_sha256']==receipt['plan_sha256'],'Reused evidence belongs to another source')
                keys=['base_mva','base_voltage_ll_v','frequency_hz','generators_bus_p_mw_v_pu',
                      'bus_load_bus_p_mw_q_mvar','auxiliary_load','branches_from_to_r_x_b_pu',
                      'line_representation','zero_sequence_assumption','fault','solver']
                diff=differences({k:cfg[k] for k in keys},{k:native_cfg[k] for k in keys})
                expected={k:v for k,v in cfg['scenario'].items() if k!='id'}
                actual={k:v for k,v in native_cfg['scenario'].items() if k!='id'}
                diff+=differences(expected,actual,'scenario')
                if diff:native_issues.append('REUSED_NATIVE_CONFIG_DIFFERENCES: '+', '.join(diff))
            files=unique(manifest["artifacts"],"file")
            electrical=sources.json(native/"electrical_build_audit.json",files["electrical_build_audit.json"]["sha256"])
            build=sources.json(native/"build_audit.json",files["build_audit.json"]["sha256"])
            inputs.append((cfg,electrical,build))
        audit=audit_pair(*inputs)
        audit['issues'].extend(native_issues)
        a=feature[pair["abc"]["acquisition_id"]+"__fault"]
        b=feature[pair["abcg"]["acquisition_id"]+"__fault"]
        contrast=max(abs(float(a[k])-float(b[k]))/max(abs(float(a[k])),abs(float(b[k])),1e-15) for k in RMS)
        result.append(dict(line=key[0],position_km=key[1],distance_B7_km=pair["abc"]["distance_B7_km"],
                           max_rms_pair_relative_difference=contrast,**audit))
    return result

def execute(bundle, run, workspace, package, config, mode):
    output_root=workspace/"experiments"/"C1-E22"
    require(workspace!=Path(workspace.anchor),"Use a named output workspace")
    for protected in (bundle,run,package,package.parents[2]):
        require(output_root!=protected and not output_root.is_relative_to(protected) and
                not protected.is_relative_to(output_root),"Output overlaps protected inputs")
    output=output_root/("review_"+utc_now().replace("-","").replace(":","").replace(".","").replace("+0000","Z")+"_"+uuid.uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    sources=Sources({"DATASET_ROOT":bundle,"BASELINE_RUN_ROOT":run,"R07_PACKAGE_ROOT":package,
                     'REPOSITORY_ROOT':package.parents[2]})
    write_json(output/"status.json",dict(status="RUNNING",new_fits=0,new_simulations=0))
    try:
        lineage=check_lineage(package)
        sources.record(package/'provenance/distance_analysis_lineage.json')
        for item in lineage['files']:
            sources.record(child(package,item['path']),item['sha256'])
        dataset_config=package/"configs/datasets/e19_reference.json"
        sources.record(dataset_config)
        dataset=load_view(bundle,dataset_config)
        pointer=sources.json(bundle/"latest_reference_10km.json")
        published=child(bundle,pointer["relative_directory"])
        sources.record(published/"view_manifest.json",dataset.identity["manifest_sha256"])
        for a in dataset.manifest["artifacts"]:
            sources.record(child(published,a["file"]),a["sha256"])
        cases,cache=load_cases(dataset,sources)
        predictions,manifest=load_predictions(dataset,run,config,sources)
        if mode=="preflight":
            sources.unchanged()
            write_json(output/"preflight.json",dict(status="SAVED_INPUT_PREFLIGHT_PASS",sources=sources.files,
                                                    acquisitions=429,new_fits=0,new_simulations=0))
            write_json(output/"status.json",dict(status="PREFLIGHT_COMPLETE",new_fits=0,new_simulations=0))
            print("preflight_directory="+str(output),flush=True)
            return
        effects=signal_effects(dataset.frame,cases)
        classified=classified_faults(predictions,cases)
        bins,types,type_distance=grouped_errors(classified)
        audit=configuration_audit(cases,cache,sources,dataset.frame)
        issues=sum(len(r["issues"]) for r in audit)
        summary=dict(experiment_id="C1-E22",status="COMPLETE",scientific_status="SAVED_SPATIAL_DESCRIPTION_REVIEW_REQUIRED",
                     created_utc=utc_now(),baseline_run_id=config["run_id"],classifier="standardized multinomial logistic regression",
                     acquisitions=429,paired_windows=858,signal_effect_rows=len(effects),fault_prediction_rows=len(classified),
                     abc_abcg_pairs=len(audit),configuration_issue_records=issues,source_sha256=sources.files,
                     model_fitting_executed=False,new_fits=0,new_simulations=0,new_bulk_MAT_reads=0,
                     fixed_reference_bus=7,measurement_route="B7B5",distance_bin_edges_km=list(BINS),
                     publication_qualified=False,physical_validation_established=False,
                     persistent_fault_validation=False,spatial_holdout=False,
                     limits=["Descriptive saved-only analysis; no causal distance/attenuation conclusion.",
                             "Noisy data, statistical inference, actual contacts and fault physics remain unqualified.",
                             "14 E16 voltage failures preserved; do not erase ambiguous classes.",
                             "No recovery information used; fixed timing does not validate persistent operation."])
        (output/"figures").mkdir()
        for name,rows in [("case_distances",cases),("signal_effects",effects),("fault_predictions",classified),
                          ("classification_by_distance",bins),("classification_by_type_line",types),
                          ('classification_by_type_distance',type_distance)]:
            write_csv(output/(name+".csv"),pd.DataFrame(rows))
        write_json(output/"abc_abcg_configuration_audit.json",audit)
        for label in LABELS[1:]:
            publish_bytes(output/"figures"/f"response_{label}.svg",response_figure(effects,label).encode())
        publish_bytes(output/"figures/classification_errors.svg",error_figure(bins).encode())
        publish_bytes(output/"index.html",html_report(summary,LABELS[1:],audit).encode())
        sources.unchanged()
        summary["artifacts"]={p.relative_to(output).as_posix():sha256(p)
                              for p in output.rglob("*") if p.is_file() and p.name!="status.json"}
        write_json(output/"analysis_report.json",summary)
        write_json(output/"status.json",dict(status="COMPLETE",new_fits=0,new_simulations=0,
                                             report_sha256=sha256(output/"analysis_report.json")))
        print(json.dumps({k:summary[k] for k in ("status","scientific_status","acquisitions","configuration_issue_records","new_fits","new_simulations")}),flush=True)
        print("report_directory="+str(output),flush=True)
        print("open_report="+str(output/"index.html"),flush=True)
    except BaseException as exc:
        write_json(output/"status.json",dict(status="INTERRUPTED" if isinstance(exc,KeyboardInterrupt) else "FAILED",
                                             error=str(exc),traceback=traceback.format_exc(),
                                             new_fits=0,new_simulations=0))
        print("preserved_attempt="+str(output),flush=True)
        raise

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--workspace",type=Path,required=True)
    parser.add_argument("--stage",choices=("preflight","run"),default="preflight")
    args=parser.parse_args()
    package=Path(__file__).resolve().parents[3]
    cfg=read_json(package/"configs/experiments/distance_review.json")
    require(cfg["measurement_route"]=="B7B5" and cfg["distance_reference_bus"]==7 and
            cfg["distance_bin_edges_km"]==list(BINS),"Unreviewed spatial protocol")
    require(args.run.name==cfg["run_directory"],"Supply the frozen baseline run")
    workspace=args.workspace.resolve()
    require(workspace!=Path(workspace.anchor), 'Use a named workspace, not a drive root')
    for protected in (args.bundle.resolve(),args.run.resolve(),package,package.parents[2]):
        require(workspace!=protected and not workspace.is_relative_to(protected),
                'Workspace must be outside acquisition, saved run and code inputs')
    with exclusive_lock(workspace/"distance_review.lock"):
        execute(args.bundle.resolve(),args.run.resolve(),args.workspace.resolve(),package,cfg,args.stage)

if __name__=="__main__":
    main()
