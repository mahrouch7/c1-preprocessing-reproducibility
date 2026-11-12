"""C1-E32 scientific units. Real work is reachable only through execute()."""
from __future__ import annotations
import gzip
import importlib.metadata
from pathlib import Path
import pickle
import platform
import time
import warnings
import numpy as np
import c1e25 as base
from c1e27_worker import activate_wavelet, make_estimator
import c1e32_spec as spec
import c1e32_common as c
import c1e32_probabilities as posterior

def observed(raw,row,condition):
    import c1e27_signal as signal
    entry=spec.BY_CONDITION[condition]; identity=dict(row)
    # The repeat is in the RNG identity, not another independent acquired case.
    identity['window_id']=row['window_id']+'__C1E32_repeat'+str(entry['repeat'])
    return signal.corrupt(raw,identity,entry['base'])

def preprocess(config,inputs,output):
    activate_wavelet(inputs['pywavelets_wheel'],Path(inputs['metadata']).parent.parent)
    import c1e27_signal as signal
    rows=spec.validate_rows(c.read(inputs['metadata']))
    ids=np.asarray(config['row_ids'],dtype=int); clean=np.load(inputs['clean'],mmap_mode='r',allow_pickle=False)
    c.require(clean.shape==(len(ids),8000,6) and clean.dtype==np.float64,'Chunk waveform shape/precision mismatch')
    c.require(len(set(ids))==len(ids) and np.all((ids>=0)&(ids<len(rows))),'Invalid chunk row identities')
    features=np.empty((len(spec.CONDITIONS),len(spec.PREPS),len(ids),36))
    tpi=np.empty((len(spec.CONDITIONS),len(spec.PREPS),len(ids),6,3,9))
    error=np.empty((len(spec.CONDITIONS),len(spec.PREPS),len(ids),6))
    noise=np.full((len(spec.CONDITIONS),len(ids),6,5),np.nan)
    examples=[]; started=time.perf_counter()
    for local,row_id in enumerate(ids):
        original=np.asarray(clean[local]); row=rows[int(row_id)]; power=np.mean(original**2,axis=0)
        c.require(np.isfinite(original).all() and np.all(power>0),'Undefined signal power')
        ec=signal.energies(original)
        for ci,entry in enumerate(spec.CONDITIONS):
            corrupted,report=observed(original,row,entry['name']); en=signal.energies(corrupted)
            delta=corrupted-original
            if entry['base']!='clean':
                noise[ci,local,:,0]=report['additive_snr_db']; noise[ci,local,:,1]=report['total_error_snr_db']
                noise[ci,local,:,2]=np.mean(delta,axis=0)/np.sqrt(power)
                noise[ci,local,:,3]=np.sqrt(np.mean(delta**2,axis=0)/power)
                noise[ci,local,:,4]=np.max(abs(delta),axis=0)/np.sqrt(power)
            for pi,prep in enumerate(spec.PREPS):
                processed=signal.preprocess(corrupted,prep)
                features[ci,pi,local]=signal.features(processed)
                tpi[ci,pi,local]=signal.tpi(original,corrupted,processed,ec,en)
                error[ci,pi,local]=np.sqrt(np.mean((processed-original)**2,axis=0)/power)
                if local==0 and entry['name'] in spec.SENSITIVITY:
                    examples.append((ci,pi,processed[::8,:].copy()))
        if local%10==0:
            print(f"[PREPROCESSING] chunk={config['chunk']} {local}/{len(ids)}; conditions=30; fits=0",flush=True)
    np.savez_compressed(output/'features.npz',row_ids=ids,features=features)
    np.savez_compressed(output/'signal_diagnostics.npz',row_ids=ids,tpi=tpi,relative_l2=error,noise=noise)
    np.savez_compressed(output/'waveform_examples.npz',row_id=ids[:1],clean=clean[0,::8,:],
        cases=np.array([[v[0],v[1]] for v in examples]),processed=np.array([v[2] for v in examples]))
    c.write(output/'preprocessing_receipt.json',{'scope_sha256':config['scope_sha256'],'row_ids':ids.tolist(),
        'conditions':spec.CONDITIONS,'preprocessors':spec.PREPS,'features':spec.FEATURES,
        'noise_fields':['additive_snr_db','total_error_snr_db','relative_mean_error','relative_error_rms','relative_peak_error'],
        'tpi_fields':signal.TPI_FIELDS,'tpi_operators':signal.TPI_OPERATORS,
        'elapsed_seconds':time.perf_counter()-started,'fits_completed':0})
    return 0

def training_mask(rows,job):
    mask=np.array([r['fold']!=job['fold'] for r in rows])
    if job['anchor_only']: mask &= np.array([r['fault']!='noFault' or r['healthy_anchor'] for r in rows],dtype=bool)
    if job['density_balanced']: mask &= np.array([r['density_control_included'] for r in rows],dtype=bool)
    return mask

def predict(model,x):
    native,probability=posterior.checked_prediction(model,x)
    # Same decision rule for constituents and probability fusion (including SVC).
    predicted=np.argmax(probability,axis=1)
    native=np.array([spec.LABELS.index(v) for v in native],dtype=np.int64)
    return predicted,probability,native

def save_prediction(output,case,row_ids,truth,predicted,probabilities,native,eligible):
    p=np.asarray(probabilities,dtype=np.float64); ids=np.asarray(row_ids,dtype=np.int64)
    truth=np.asarray(truth,dtype=np.int64); predicted=np.asarray(predicted,dtype=np.int64)
    c.require(len(set(ids))==len(ids) and truth.shape==predicted.shape==ids.shape and p.shape==(len(ids),12),'Prediction dimensions/identity invalid')
    c.require(np.isfinite(p).all() and np.all((p>=0)&(p<=1)) and np.allclose(p.sum(axis=1),1,atol=1e-10,rtol=0),'Invalid probability values')
    c.require(np.all((truth>=0)&(truth<12)) and np.all((predicted>=0)&(predicted<12)),'Invalid label codes')
    path=output/('prediction_'+spec.digest(case)+'.npz')
    with path.open('xb') as stream:
        np.savez_compressed(stream,row_ids=ids,truth=truth,predicted=predicted,
                            probabilities=p,native_predicted=np.asarray(native,dtype=np.int64))
    return {'case':case,'path':path.name,'sha256':c.sha(path),'eligible_for_inference':bool(eligible)}

def fit(config,inputs,output,estimator_factory=make_estimator):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.dummy import DummyClassifier
    from threadpoolctl import threadpool_limits
    rows=spec.validate_rows(c.read(inputs['metadata']))
    arrays={entry['name']:np.load(inputs[entry['name']],allow_pickle=False) for entry in spec.CONDITIONS}
    c.require(all(a.shape==(len(rows),36) and a.dtype==np.float64 and np.isfinite(a).all() for a in arrays.values()),'Incomplete feature inputs')
    test=np.array([r['fold']==config['fold'] for r in rows]); ids=np.arange(len(rows))[test]
    truth=np.array([spec.LABELS.index(r['fault']) for r in rows]); records=[]; catalogue=[]; fusions={}
    c.require(not ({r['family_id'] for r in rows if r['fold']==config['fold']} & {r['family_id'] for r in rows if r['fold']!=config['fold']}),'Cross-fold family leakage')
    definitions=spec.jobs(config['preprocessor'],config['fold'],config['classifier'])
    with threadpool_limits(limits=1):
        for ordinal,job in enumerate(definitions,1):
            train=training_mask(rows,job); cols=spec.REPS[job['representation']]
            y=np.array([r['fault'] for r in rows])[train]
            c.require(set(y)==set(spec.LABELS),'A controlled training view lacks a class')
            if job['classifier'].startswith('dummy_'):
                model=DummyClassifier(strategy='prior' if job['classifier']=='dummy_prior' else 'most_frequent')
            else: model=estimator_factory(job['classifier'],job['normalizer'])
            started=time.perf_counter()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                model.fit(arrays[job['training_condition']][train][:,cols],y)
            elapsed=time.perf_counter()-started
            eligible=not any(issubclass(w.category,ConvergenceWarning) for w in captured)
            model_path=output/('model_'+job['job_id']+'.pickle.gz')
            with model_path.open('xb') as stream:
                with gzip.GzipFile(fileobj=stream,mode='wb',compresslevel=3,mtime=0) as target:
                    pickle.dump(model,target,protocol=pickle.HIGHEST_PROTOCOL)
            record={'job':job,'fit_seconds':elapsed,'eligible_for_inference':eligible,
                'train_windows':int(train.sum()),'test_windows':int(test.sum()),
                'training_case_ids_sha256':spec.digest([r['window_id'] for r,m in zip(rows,train) if m]),
                'training_families':sorted({r['family_id'] for r,m in zip(rows,train) if m}),
                'feature_columns':[spec.FEATURES[v] for v in cols],
                'warnings':[{'category':w.category.__name__,'message':str(w.message)} for w in captured],
                'saved_model':{'path':model_path.name,'sha256':c.sha(model_path),'bytes':model_path.stat().st_size},
                'probability_computation':posterior.policy(model),'prediction_seconds':0.0}
            for protocol,condition in spec.evaluations(job):
                started=time.perf_counter(); predicted,p,native=predict(model,arrays[condition][test][:,cols])
                record['prediction_seconds']+=time.perf_counter()-started
                case=spec.case(job,protocol,condition)
                catalogue.append(save_prediction(output,case,ids,truth[test],predicted,p,native,eligible))
                if job['study']=='robustness' and job['representation'] in ('RMS6','RAW12'):
                    fusions.setdefault((protocol,condition),{})[job['representation']]=(p,eligible)
            records.append(record)
            print(f"[FIT COMPLETE] {config['preprocessor']} fold={config['fold']} {job['classifier']} {ordinal}/{len(definitions)} eligible={eligible}",flush=True)
        for (protocol,condition),parts in fusions.items():
            c.require(set(parts)=={'RMS6','RAW12'},'Late-fusion constituents incomplete')
            base_condition=spec.BY_CONDITION[condition]['base']; _,snr=spec.old.condition(base_condition)
            for name,w in [('LATE_SIMPLE',.5),('LATE_ORACLE_SNR',.7 if snr is not None and snr<=20 else .3)]:
                p=w*parts['RMS6'][0]+(1-w)*parts['RAW12'][0]
                job={'study':'robustness','representation':name,'normalizer':spec.default_scale(config['classifier']),
                     'anchor_only':False,'density_balanced':False,'classifier':config['classifier'],'preprocessor':config['preprocessor']}
                case=spec.case(job,protocol,condition); pred=np.argmax(p,axis=1)
                catalogue.append(save_prediction(output,case,ids,truth[test],pred,p,pred,parts['RMS6'][1] and parts['RAW12'][1]))
    c.require(len(records)==config['expected_fits'],'Fit budget changed')
    c.write(output/'prediction_catalogue.json',catalogue)
    c.write(output/'fit_records.json',{'fits_completed':len(records),'reused_fits':0,
        'nonconverged':sum(not r['eligible_for_inference'] for r in records),'records':records})
    return len(records)

def execute(args):
    configuration=Path(args.configuration); output=Path(args.output)
    c.require(c.sha(configuration)==args.configuration_sha256,'Configuration bytes changed')
    config=c.read(configuration); spec.validate_configuration(config,args.expected_fits)
    c.capacity(configuration.parent,5)
    wanted=base.VERSIONS
    c.require({k:importlib.metadata.version(k) for k in wanted}==wanted,'Scientific dependency versions changed')
    paths=c.inputs(config,configuration.parent)
    fits=(preprocess if config['stage']=='A' else fit)(config,paths,output)
    versions={k:importlib.metadata.version(k) for k in (spec.VERSIONS if config['stage']=='A' else base.VERSIONS)}
    c.write(output/'runtime.json',{'host':platform.node(),'python':platform.python_version(),'versions':versions,
        'fits_completed':fits,'simulations':0,'stage':config['stage'],'scope_sha256':config['scope_sha256']})
    artifacts=[{'path':p.name,'bytes':p.stat().st_size,'sha256':c.sha(p)} for p in sorted(output.iterdir()) if p.is_file()]
    c.write(output/'scientific_receipt.json',{'status':'COMPLETED','unit_id':args.unit_id,'role':args.role,
        'assignment_sha256':args.assignment_sha256,'configuration_sha256':args.configuration_sha256,
        'expected_fits':args.expected_fits,'fits_completed':fits,'simulations':0,'artifacts':artifacts})
