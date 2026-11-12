"""Read-only E31 ingestion and family-fold design; never simulates or fits."""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import c1e25 as base
import c1e32_spec as spec
import c1e32_common as c

def family_folds(families):
    """Balance numbers of held-out regimes, NOT windows from the dense nominal."""
    families=sorted(set(families))
    rng=np.random.Generator(np.random.PCG64(20261026))
    order=rng.permutation(len(families))
    return {families[int(i)]:1+j%5 for j,i in enumerate(order)}

def leakage_audit(rows):
    """Reject direct identity/regime reuse; do not equate similarity with independence."""
    spec.validate_rows(rows)
    load_families=defaultdict(set)
    for row in rows:
        key=tuple(round(float(v),12) for v in row['factors'])
        load_families[key].add(row['family_id'])
    overlaps=[key for key,value in load_families.items() if len(value)>1]
    c.require(not overlaps,'Exact load regime belongs to multiple families; regroup before training')
    for name in ('raw_sha256','clean_window_numeric_sha256'):
        c.require(all(name in r for r in rows),'Missing direct-duplicate audit identity: '+name)
        identities=defaultdict(list)
        for row in rows: identities[row[name]].append(row)
        repeated=[value for value in identities.values() if len(value)>1]
        c.require(not repeated,'Repeated saved waveform identity: '+name+'; stop before training')
    return {'status':'DIRECT_IDENTITY_AND_REGIME_LEAKAGE_GUARDS_PASS','windows':len(rows),
        'unique_acquisitions':len({r['acquisition_id'] for r in rows}),
        'families':len({r['family_id'] for r in rows}),'exact_load_points':len(load_families),
        'load_regime_rounding_decimals':12,
        'cross_family_exact_load_collisions':0,'repeated_raw_hashes':0,'repeated_numeric_window_hashes':0,
        'outer_train_test_family_overlap':0,'learned_scalers':'TRAIN_PARTITION_ONLY_SKLEARN_PIPELINE',
        'test_score_tuning':False,'predictors':'FIXED_SIX_MEASURED_CHANNELS_ONLY',
        'limits':'No direct reuse found; near-template dependence and external generalization are NOT certified. Feature-value equality can be genuine projection/identifiability, not proof of row reuse.'}

def metadata(bundle):
    bundle=Path(bundle); manifest=bundle/'views/paper_reduced/view_manifest.json'
    c.require(c.sha(manifest)==spec.DATASET_SHA,'Wrong candidate dataset view')
    view,plan,fault,gate=(c.read(bundle/p) for p in ('views/paper_reduced/view_manifest.json','plan.json','fault_plan.json','healthy_gate.json'))
    for key,name in [('parent_plan_sha256','plan.json'),('fault_plan_sha256','fault_plan.json'),('healthy_gate_sha256','healthy_gate.json')]:
        c.require(view[key]==c.sha(bundle/name),'Broken acquisition view binding')
    c.require(view['common_prefix_s']==.05 and view['prefix_in_compaction'] is False and view['prefaults_are_noFault_rows'] is False,'Timing/healthy construction changed')
    jobs={j['case_id']:j for j in plan['healthy_jobs']+fault['jobs']}
    c.require(len(jobs)==len(view['records'])==1839,'Missing/duplicate acquired case')
    folds=family_folds(r['family_id'] for r in view['records'])
    rows=[]
    for i,record in enumerate(view['records']):
        job=jobs[record['case_id']]; cfg=job['config']; directory=c.child(bundle,job['relative_directory'])
        window=c.read(directory/'window.json'); receipt=c.read(directory/'case_receipt.json'); native=c.read(directory/'native_receipt.json')
        c.require(c.sha(directory/'window.json')==record['window_sha256'],'Changed acquired window metadata')
        c.require(c.sha(directory/'input_config.json')==job['config_sha256']==native['config_sha256']==receipt['config_sha256'],'Acquisition config changed')
        c.require(c.sha(directory/'native_receipt.json')==receipt['native_receipt_sha256'],'Native receipt changed')
        c.require(receipt['status']=='QUALITY_READY_NOT_PUBLICATION_QUALIFIED' and native['status']=='COMPLETE' and native['simulations_executed']==1,'Acquisition not quality ready')
        c.require(window['samples']==8000 and window['window_start_s']==.05 and window['window_stop_exclusive_s']==.25,'Wrong useful interval')
        c.require(window['predictor_channels']==base.CHANNELS and window['true_location_is_predictor'] is False,'Predictor channel/scope mismatch')
        c.require(window['case_id']==record['case_id']==receipt['case_id'] and window['label']==record['label']==receipt['label'] and window['family_id']==record['family_id']==receipt['family_id'],'Case/class/family mismatch')
        chosen=next(r for r in receipt['candidate_prefixes'] if r['prefix_s']==.05)
        c.require(chosen['pass'] and chosen['clean_window_numeric_sha256']==window['clean_window_numeric_sha256'],'Selected prefix no longer passes')
        raw=next(a for a in native['artifacts'] if a['file']=='raw.mat')
        c.require(raw['sha256']==window['raw_file_sha256']==receipt['raw_file_sha256'],'Raw identity mismatch')
        rows.append({'Id':i,'window_id':record['case_id'],'acquisition_id':record['case_id'],
            'family_id':record['family_id'],'source_group_id':record['family_id'],'fold':folds[record['family_id']],
            'fault':record['label'],'healthy_anchor':window['healthy_anchor'],
            'factors':cfg['c1e31']['factors_B5_B6_B8'],'nominal':bool(np.allclose(cfg['c1e31']['factors_B5_B6_B8'],1,atol=1e-14,rtol=0)),
            'line_id':None if record['label']=='noFault' else cfg['scenario']['line_id'],
            'fraction':None if record['label']=='noFault' else cfg['fault']['electrical_fraction_from'],
            'raw_sha256':raw['sha256'],'raw_relative_path':job['relative_directory']+'/raw.mat',
            'clean_window_numeric_sha256':window['clean_window_numeric_sha256'],
            'window_sha256':record['window_sha256'],'case_receipt_sha256':c.sha(directory/'case_receipt.json')})
    # Fixed density sensitivity: three distinct lines, each closest to midpoint.
    nominal=[r for r in rows if r['nominal'] and r['fault']=='ab']
    lines=sorted({r['line_id'] for r in nominal})
    c.require(len(lines)>=3,'Missing nominal spatial coverage')
    chosen_lines=[lines[0],lines[len(lines)//2],lines[-1]]
    chosen={(line,min((r['fraction'] for r in nominal if r['line_id']==line),key=lambda v:(abs(v-.5),v))) for line in chosen_lines}
    for r in rows:
        r['density_control_included']=r['fault']=='noFault' or not r['nominal'] or (r['line_id'],r['fraction']) in chosen
    spec.validate_rows(rows)
    leakage_audit(rows)
    return rows,view,{'family_fold_counts':dict(Counter(folds.values())),
        'fold_rule':'fixed PCG64 seed20261026; equal number of families; not optimized on outcomes',
        'density_control_nominal_cells':[{'line_id':a,'fraction':b} for a,b in sorted(chosen)]}

def extract(bundle,output,rows):
    """Hash-check raw files and re-extract fixed channels. Writes only analysis copy."""
    path=Path(output)/'clean.npy'
    c.require(not path.exists(),'Never overwrite saved waveforms')
    values=np.lib.format.open_memmap(path,mode='w+',dtype=np.float64,shape=(len(rows),8000,6))
    features=np.empty((len(rows),18)); summaries=[]; quality=[]
    for row in rows:
        raw_path=c.child(bundle,row['raw_relative_path'])
        c.require(c.sha(raw_path)==row['raw_sha256'],'Changed raw input')
        data=loadmat(raw_path,squeeze_me=True); time=np.asarray(data['time']).reshape(-1); raw=np.asarray(data['raw'])
        mapping=c.read(raw_path.parent/'channel_map.json')['channels']
        c.require(raw.shape==(len(time),108) and len(mapping)==108 and len(set(mapping))==108 and np.isfinite(raw).all(),'Raw schema/finiteness failure')
        c.require(np.max(abs(time-np.arange(len(time))*25e-6))<1e-11,'Raw sample clock changed')
        indices=[mapping.index(ch) for ch in base.CHANNELS]
        window=np.asarray(raw[2000:10000][:,indices],dtype='<f8',order='C')
        w=c.read(raw_path.parent/'window.json')
        import hashlib
        c.require(window.shape==(8000,6) and hashlib.sha256(window.tobytes()).hexdigest()==w['clean_window_numeric_sha256'],'Useful window bytes changed')
        vector=base.feature_vector(window)
        reference=np.asarray(w['feature_vectors']['RMS6']+w['feature_vectors']['RAW12'])
        c.require(np.allclose(vector,reference,atol=1e-10,rtol=1e-12),'Re-extracted descriptors differ')
        values[row['Id']]=window; features[row['Id']]=vector
        for ch,name in enumerate(base.CHANNELS):
            x=window[:,ch]; t=time[2000:10000]
            design=np.column_stack([np.ones(8000),np.cos(2*np.pi*60*t),np.sin(2*np.pi*60*t)])
            beta=np.linalg.lstsq(design,x,rcond=None)[0]
            summaries.append({'Id':row['Id'],'family_id':row['family_id'],'label':row['fault'],'channel':name,
                'mean':float(x.mean()),'std':float(x.std(ddof=1)),'rms':float(np.sqrt(np.mean(x*x))),
                'minimum':float(x.min()),'maximum':float(x.max()),'peak':float(np.max(abs(x))),
                'fundamental_rms':float(np.hypot(beta[1],beta[2])/np.sqrt(2)),
                'phase_rad':float(np.arctan2(-beta[2],beta[1]))})
        q=c.read(raw_path.parent/'case_receipt.json'); chosen=next(v for v in q['candidate_prefixes'] if v['prefix_s']==.05)
        quality.append({'Id':row['Id'],'label':row['fault'],'family_id':row['family_id'],
            'voltage_min_pu':chosen['voltage_min_pu'],'voltage_max_pu':chosen['voltage_max_pu'],
            'kcl_relative':q['kcl_relative'],'aux_ohm_relative':q['aux_ohm_relative'],
            'off_snubber_error_A':q['off_snubber_error_A'],'prefault_construction_relative':q['prefault_construction_relative'],
            **{k:v for k,v in chosen['metrics'].items() if isinstance(v,(float,int))}})
        if row['Id']%50==0: print(f"[DATASET READBACK] {row['Id']}/{len(rows)}; fits=0",flush=True)
    values.flush(); del values
    np.save(Path(output)/'clean_base_features.npy',features,allow_pickle=False)
    duplicates=[]
    for name,cols in base.REPRESENTATIONS.items():
        groups=defaultdict(list)
        for row,vector in zip(rows,features[:,cols]): groups[vector.astype('<f8').tobytes()].append(row)
        exact=[group for group in groups.values() if len(group)>1]
        duplicates.append({'representation':name,'exact_duplicate_feature_groups':len(exact),
            'groups_crossing_outer_folds':sum(len({r['fold'] for r in group})>1 for group in exact),
            'groups_with_conflicting_labels':sum(len({r['fault'] for r in group})>1 for group in exact),
            'interpretation':'DESCRIPTOR_COLLISION_DIAGNOSTIC_NOT_AUTOMATIC_EVIDENCE_OF_RAW_REUSE'})
    c.table(Path(output)/'exact_descriptor_collisions.csv',duplicates)
    c.table(Path(output)/'channel_statistics.csv',summaries)
    # Healthy/full-window metric fields differ; normalize explicitly, never infer zeros.
    keys=sorted(set().union(*(r.keys() for r in quality)))
    c.table(Path(output)/'electrical_quality.csv',[{k:r.get(k) for k in keys} for r in quality])
    return path

def characterize(rows,clean,output):
    counts=Counter(r['fault'] for r in rows); cells=Counter((r['family_id'],r['fault'],r['fold']) for r in rows)
    c.table(output/'class_family_fold_counts.csv',[{'family_id':f,'label':label,'fold':fold,'count':n} for (f,label,fold),n in sorted(cells.items())])
    c.table(output/'load_and_spatial_coverage.csv',[{k:r[k] for k in ('Id','family_id','fault','fold','nominal','healthy_anchor','line_id','fraction','density_control_included')} | {'B5':r['factors'][0],'B6':r['factors'][1],'B8':r['factors'][2]} for r in rows])
    healthy=np.array([r['fault']=='noFault' for r in rows]); h=np.asarray(clean[healthy])
    # Six-channel relative waveform distances and phase-insensitive RMS distances.
    scale=np.maximum(np.sqrt(np.mean(h*h,axis=(0,1))),1e-12)
    z=(h/scale).reshape(len(h),-1); norm=np.sum(z*z,axis=1)
    distances=np.sqrt(np.maximum(0,(norm[:,None]+norm[None,:]-2*z@z.T)/z.shape[1]))
    np.fill_diagonal(distances,np.inf)
    rms=np.sqrt(np.mean(h*h,axis=1)); d=rms[:,None,:]-rms[None,:,:]
    amp=np.sqrt(np.mean((d/scale)**2,axis=-1)); np.fill_diagonal(amp,np.inf)
    hr=[r for r in rows if r['fault']=='noFault']
    nearest=[]
    for i,row in enumerate(hr):
        j=int(np.argmin(distances[i])); k=int(np.argmin(amp[i]))
        nearest.append({'case_id':row['window_id'],'family_id':row['family_id'],
            'nearest_waveform_case':hr[j]['window_id'],'nearest_waveform_relative_l2':float(distances[i,j]),
            'nearest_amplitude_case':hr[k]['window_id'],'nearest_rms_relative_l2':float(amp[i,k]),
            'waveform_neighbor_same_family':row['family_id']==hr[j]['family_id']})
    c.table(output/'healthy_nearest_neighbors.csv',nearest)
    phase_summary={'healthy_rms_min':rms.min(axis=0).tolist(),'healthy_rms_max':rms.max(axis=0).tolist(),
        'healthy_rms_cv':(rms.std(axis=0,ddof=1)/rms.mean(axis=0)).tolist(),
        'nearest_distance_metric':'pooled channel-RMS normalized raw waveform L2; amplitude RMS distance is phase-insensitive, not sample alignment',
        'healthy_count':len(hr),'class_counts':dict(counts),'physical_validity_established':False,
        'support_warning':'117 healthy-only points; matched-anchor and training-enrichment controls required',
        'exact_numeric_identity_is_not_independence':True}
    c.write(output/'dataset_characterization.json',phase_summary)
    np.savez_compressed(output/'healthy_diagnostic_examples.npz',row_ids=np.array([r['Id'] for r in hr]),
                        waveforms=h[:8],all_healthy_rms=rms,normalized_pair_distances=np.where(np.isfinite(distances),distances,np.nan))
