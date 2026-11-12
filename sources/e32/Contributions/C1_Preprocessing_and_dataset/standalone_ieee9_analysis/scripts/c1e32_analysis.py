"""Complete-return aggregation and prospective family-aware statistical study."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import io
from pathlib import Path
import zipfile
import numpy as np
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
import c1e32_spec as spec
import c1e32_common as c
import c1e32_statistics as stats
import c1e32_probabilities as posterior


def collection_units(sources):
    """Retain each return's genuine signed-by-hash assignment; never rebind markers."""
    from r07.distributed.contracts import validate_assignment
    sources = [(sources, None)] if isinstance(sources, dict) else sources
    result = []; seen = set()
    for assignment, selected_ids in sources:
        validate_assignment(assignment, require_authorized=True)
        selected_ids = set(selected_ids) if selected_ids is not None else {u['unit_id'] for u in assignment['units']}
        available = {u['unit_id']: u for u in assignment['units']}
        c.require(selected_ids <= set(available), 'Selected return absent from original assignment')
        for uid in sorted(selected_ids):
            unit = available[uid]; ident = unit['identity']
            logical = (ident['preprocessor'], ident['fold'], ident['classifier'])
            c.require(logical not in seen, 'Duplicate logical scientific unit across assignments')
            seen.add(logical); result.append((unit, assignment))
    expected = {(u['preprocessor'], u['fold'], u['classifier']) for u in spec.learning_plan()}
    c.require(seen == expected, 'Combined scientific unit coverage differs from fixed scope')
    return result

def pooled(path):
    with np.load(path,allow_pickle=False) as data: return {k:data[k] for k in data.files}

def collect(assignment,transport,output,rows):
    from run_c1e25_pipeline import returned_archive
    target=output/'shards'; target.mkdir(); groups=defaultdict(list); fits=[]
    expected=stats.expected_cases()
    units = collection_units(assignment)
    for ordinal,(unit,origin) in enumerate(units,1):
        ids=np.array([r['Id'] for r in rows if r['fold']==unit['identity']['fold']])
        with zipfile.ZipFile(returned_archive(unit,origin,transport)) as archive:
            record=__import__('json').loads(archive.read('fit_records.json'))
            c.require(record['fits_completed']==unit['expected_fits'] and record['reused_fits']==0,'Returned fit ceiling differs')
            definitions=spec.jobs(unit['identity']['preprocessor'],unit['identity']['fold'],unit['identity']['classifier'])
            c.require({r['job']['job_id'] for r in record['records']}=={j['job_id'] for j in definitions} and len(record['records'])==len(definitions),'Returned fit definitions differ')
            by_job = {j['job_id']: j for j in definitions}
            c.require(all(r['job'] == by_job[r['job']['job_id']] for r in record['records']),
                      'Returned fit configuration differs from declared job')
            for fit in record['records']:
                if fit['job']['classifier'] == 'naive_bayes':
                    c.require(fit.get('probability_computation') == posterior.POLICY,
                              'GaussianNB posterior implementation provenance missing/different')
                fit['role']=unit['role']; fit['expected_host']=unit['expected_host']; fit['unit_id']=unit['unit_id']; fits.append(fit)
            entries=__import__('json').loads(archive.read('prediction_catalogue.json')); seen=set()
            for entry in entries:
                identity=spec.digest(entry['case']); c.require(identity in expected and expected[identity]==entry['case'] and identity not in seen,'Foreign/duplicate prediction case')
                seen.add(identity); payload=archive.read(entry['path'])
                c.require(hashlib.sha256(payload).hexdigest()==entry['sha256'],'Prediction catalogue hash mismatch')
                with np.load(io.BytesIO(payload),allow_pickle=False) as data:
                    c.require(np.array_equal(data['row_ids'],ids),'OOF fold row identity mismatch')
                    c.require(np.array_equal(data['truth'],[spec.LABELS.index(rows[i]['fault']) for i in ids]),'OOF labels differ')
                    p=data['probabilities']
                    c.require(p.shape==(len(ids),12) and np.isfinite(p).all() and np.all((p>=0)&(p<=1)) and np.allclose(p.sum(axis=1),1,atol=1e-10,rtol=0),'Invalid returned probabilities')
                    c.require(np.array_equal(data['predicted'],p.argmax(axis=1)),'Main probability-argmax rule changed')
                path=target/(identity+'_fold'+str(unit['identity']['fold'])+'.npz')
                with path.open('xb') as stream: stream.write(payload)
                groups[identity].append({'case':entry['case'],'fold':unit['identity']['fold'],'path':path,
                                        'eligible':entry['eligible_for_inference']})
        if ordinal%10==0: print(f'[AGGREGATE] verified {ordinal}/{len(units)} units',flush=True)
    c.require(len(fits)==sum(u['expected_fits'] for u,_ in units)==spec.scope()['fit_budget'],'Global completed fit count differs')
    c.require(set(groups)==set(expected),'Scientific prediction grid is incomplete')
    c.write(output/'all_fit_records.json',fits)
    index={}; directory=output/'oof'; directory.mkdir()
    for identity,parts in sorted(groups.items()):
        c.require(len(parts)==5 and {p['fold'] for p in parts}==set(range(1,6)),'Incomplete/duplicate outer folds')
        shards=[pooled(p['path']) for p in parts]; ids=np.concatenate([d['row_ids'] for d in shards]); order=np.argsort(ids)
        c.require(np.array_equal(ids[order],np.arange(len(rows))),'Pooled OOF coverage failed')
        arrays={k:np.concatenate([d[k] for d in shards])[order] for k in shards[0]}
        path=directory/(identity+'.npz')
        with path.open('xb') as stream: np.savez_compressed(stream,**arrays)
        index[identity]={'case':parts[0]['case'],'path':path.relative_to(output).as_posix(),
            'sha256':c.sha(path),'eligible_for_inference':all(p['eligible'] for p in parts)}
    c.write(output/'pooled_prediction_index.json',index)
    return index,fits

def metrics(index,rows,output):
    result=[]; per_class=[]; calibration=[]; family_errors=[]; matrices=[]; matrix_ids=[]
    anchor=np.array([r['fault']!='noFault' or r['healthy_anchor'] for r in rows],dtype=bool)
    extra=np.array([r['fault']=='noFault' and not r['healthy_anchor'] for r in rows],dtype=bool)
    for ordinal,(identity,entry) in enumerate(sorted(index.items()),1):
        data=pooled(output/entry['path']); native_diff=int(np.count_nonzero(data['native_predicted']!=data['predicted']))
        for population in ('family_class_standardized','pooled'):
            for view,mask in [('full',np.ones(len(rows),bool)),('common_anchor',anchor)]:
                scores,matrix=stats.score(rows,data['truth'],data['predicted'],population,mask)
                w=stats.weights(rows,population,mask); probability=data['probabilities']; encoded=np.eye(12)[data['truth']]
                scores['brier_multiclass']=float(np.sum(w*np.sum((probability-encoded)**2,axis=1))/w.sum())
                scores['log_loss']=float(np.sum(-w*np.log(np.maximum(probability[np.arange(len(rows)),data['truth']],1e-15)))/w.sum())
                extra_false=float(np.mean(data['predicted'][extra]!=0))
                result.append({'case_id':identity,**entry['case'],'population':population,'view':view,
                    'eligible_for_inference':entry['eligible_for_inference'],'native_argmax_disagreements':native_diff,
                    'extra_healthy_false_alarm':extra_false,'log_loss_epsilon':1e-15,**scores})
                matrices.append(matrix); matrix_ids.append(identity+'__'+population+'__'+view)
                for label,code in zip(spec.LABELS,range(12)):
                    actual=float(matrix[code].sum()); predicted=float(matrix[:,code].sum()); tp=float(matrix[code,code])
                    per_class.append({'case_id':identity,'population':population,'view':view,'label':label,
                        'eligible_for_inference':entry['eligible_for_inference'],'support_weight':actual,
                        'precision':tp/predicted if predicted else 0.,'recall':tp/actual,
                        'f1':2*tp/(actual+predicted) if actual+predicted else 0.})
        if selected(entry['case']):
            for family in sorted({r['family_id'] for r in rows}):
                mask=np.array([r['family_id']==family for r in rows]); score,_=stats.score(rows,data['truth'],data['predicted'],'family_class_standardized',mask)
                family_errors.append({'case_id':identity,'family_id':family,'eligible_for_inference':entry['eligible_for_inference'],**score})
            confidence=data['probabilities'].max(axis=1); correctness=data['truth']==data['predicted']; w=stats.weights(rows)
            for low in np.arange(0,1,.1):
                mask=(confidence>=low)&(confidence<(low+.1) if low<.9 else confidence<=1)
                calibration.append({'case_id':identity,'bin_left':float(low),'count':int(mask.sum()),
                    'mean_confidence':float(np.average(confidence[mask],weights=w[mask])) if mask.any() else None,
                    'accuracy':float(np.average(correctness[mask],weights=w[mask])) if mask.any() else None})
        if ordinal%100==0: print(f'[METRICS] {ordinal}/{len(index)} pooled cases',flush=True)
    c.table(output/'metrics.csv',result); c.table(output/'per_class_metrics.csv',per_class)
    c.table(output/'family_metrics_selected.csv',family_errors); c.table(output/'calibration_selected.csv',calibration)
    np.savez_compressed(output/'confusion_matrices.npz',case_ids=np.array(matrix_ids),matrices=np.array(matrices))
    return result

def selected(case):
    conditions=('clean','awgn__20')+tuple(n+'__10' for n in spec.old.NOISES)
    return case['study']=='robustness' and case['protocol']=='deployment' and case['preprocessor']=='none' and case['representation'] in ('RMS6','RAW12','MULTI18','LATE_SIMPLE') and case['condition'] in conditions

def inference(index,rows,output,comparisons,bootstrap=None,swaps=None):
    bootstrap=bootstrap or spec.STATS['bootstrap_resamples']; swaps=swaps or spec.STATS['paired_swaps']; intervals=[]; tests=[]
    for identity,entry in sorted(index.items()):
        if selected(entry['case']):
            if entry['eligible_for_inference']:
                data=pooled(output/entry['path']); values=stats.matrices(rows,data['truth'],data['predicted'])
                seed=int(identity[:16],16); ci=stats.interval(values,bootstrap,seed)
            else: ci={'lower_95':None,'upper_95':None,'families':39}
            intervals.append({'case_id':identity,'eligible_for_inference':entry['eligible_for_inference'],**ci})
    for ordinal,comparison in enumerate(comparisons,1):
        left_id,right_id=spec.digest(comparison['left']),spec.digest(comparison['right'])
        c.require(left_id in index and right_id in index,'A prospective comparison is missing')
        left,right=index[left_id],index[right_id]; eligible=left['eligible_for_inference'] and right['eligible_for_inference']
        if eligible:
            values=stats.paired(pooled(output/left['path']),pooled(output/right['path']),rows,
                bootstrap=bootstrap,swaps=swaps,seed=int(comparison['comparison_id'][:16],16))
        else:
            values={'difference':None,'lower_95':None,'upper_95':None,'permutation_p':1.,
                    'p_mc_lower_99':1.,'p_mc_upper_99':1.,'families':39}
        tests.append({'comparison_id':comparison['comparison_id'],'family':comparison['family'],
            'left_case_id':left_id,'right_case_id':right_id,'eligible_for_inference':eligible,**values})
        if ordinal%20==0: print(f'[PAIRED INFERENCE] {ordinal}/{len(comparisons)}; cluster=load_family',flush=True)
    for family in sorted({t['family'] for t in tests}):
        members=[t for t in tests if t['family']==family]
        for field,target in [('permutation_p','holm_p'),('p_mc_lower_99','holm_mc_lower'),('p_mc_upper_99','holm_mc_upper')]:
            adjusted=stats.holm([t[field] for t in members])
            for t,value in zip(members,adjusted): t[target]=float(value)
        for t in members:
            t['family_size_including_exclusions']=len(members)
            t['mc_precision_review']=bool(t['eligible_for_inference'] and t['holm_mc_lower']<=.05<t['holm_mc_upper'])
            if not t['eligible_for_inference']: t['interpretation']='EXCLUDED_NUMERICAL_WARNING'
            elif t['mc_precision_review']: t['interpretation']='MONTE_CARLO_PRECISION_REVIEW'
            elif t['holm_p']<.05 and t['lower_95']>0: t['interpretation']='SUPPORTED_POSITIVE_DIFFERENCE'
            elif t['holm_p']<.05 and t['upper_95']<0: t['interpretation']='SUPPORTED_NEGATIVE_DIFFERENCE'
            else: t['interpretation']='INCONCLUSIVE_NOT_EQUIVALENT'
    c.table(output/'selected_score_intervals.csv',intervals); c.table(output/'paired_family_inference.csv',tests)
    return tests

def diagnostic_summaries(root,rows,output):
    tpi=np.load(root/'verified_diagnostics/tpi.npy',mmap_mode='r',allow_pickle=False)
    errors=np.load(root/'verified_diagnostics/relative_l2.npy',allow_pickle=False)
    noise=np.load(root/'verified_diagnostics/noise.npy',allow_pickle=False)
    c.require(tpi.shape==(30,5,len(rows),6,3,9) and np.isfinite(tpi).all() and np.isfinite(errors).all(),'Invalid signal diagnostic tensors')
    summary=[]; diagnostic_noise=[]; shifts=[]
    clean=np.load(root/'verified_features/clean_none.npy',allow_pickle=False); std=clean.std(axis=0,ddof=1)
    correlation=np.corrcoef(clean,rowvar=False)
    np.savez_compressed(output/'feature_correlation.npz',names=np.array(spec.FEATURES),correlation=correlation)
    safe_std=np.where(std>1e-12,std,1); centered=(clean-clean.mean(axis=0))/safe_std
    _,singular,right=np.linalg.svd(centered,full_matrices=False)
    np.savez_compressed(output/'exploratory_pca.npz',scores=centered@right[:3].T,
        explained_variance_ratio=singular**2/np.sum(singular**2),components=right[:3])
    for ci,entry in enumerate(spec.CONDITIONS):
        for pi,prep in enumerate(spec.PREPS):
            for channel in range(6):
                for op,name in enumerate(('DWT_D3_2500_5000Hz','DWT_D1_10000_20000Hz','FFT_2500_5000Hz')):
                    values=tpi[ci,pi,:,channel,op]
                    summary.append({'condition':entry['name'],'preprocessor':prep,'channel':spec.base.CHANNELS[channel],
                        'operator':name,'median_tpi_noisy':float(np.median(values[:,3])),
                        'median_tpi_clean':float(np.median(values[:,4])),'median_noise_inflation':float(np.median(values[:,5])),
                        'median_hf_error':float(np.median(values[:,6])),'tpi_gt_one_count':int(np.sum(values[:,3]>1)),
                        'clean_near_zero_count':int(np.sum(values[:,7])),'noisy_near_zero_count':int(np.sum(values[:,8])),
                        'median_relative_l2':float(np.median(errors[ci,pi,:,channel]))})
        if entry['base']!='clean':
            expected=spec.old.condition(entry['base'])[1]
            c.require(np.isfinite(noise[ci]).all() and np.max(abs(noise[ci,:, :,0]-expected))<=spec.base.SNR_ATOL_DB,'Measured additive SNR failed final tolerance')
            for channel,name in enumerate(spec.base.CHANNELS):
                value=noise[ci,:,channel]
                diagnostic_noise.append({'condition':entry['name'],'channel':name,'additive_snr_min':float(value[:,0].min()),
                    'additive_snr_max':float(value[:,0].max()),'total_error_snr_median':float(np.median(value[:,1])),
                    'relative_mean_error_median':float(np.median(value[:,2])),
                    'relative_error_rms_median':float(np.median(value[:,3])),
                    'maximum_relative_peak_error':float(value[:,4].max())})
        values=np.load(root/'verified_features'/(entry['name']+'_none.npy'),allow_pickle=False)
        for column,name in enumerate(spec.FEATURES):
            shifts.append({'condition':entry['name'],'feature':name,'median_change':float(np.median(values[:,column]-clean[:,column])),
                'median_abs_change_over_clean_std':float(np.median(abs(values[:,column]-clean[:,column]))/std[column]) if std[column]>1e-12 else None})
    c.table(output/'tpi_signal_diagnostics.csv',summary); c.table(output/'measured_noise_diagnostics.csv',diagnostic_noise)
    c.table(output/'descriptor_shifts.csv',shifts)
    return summary

def secondary_outputs(metrics,index,rows,fits,output,signal_summary):
    primary=[r for r in metrics if r['population']=='family_class_standardized' and r['view']=='full']
    lookup={r['case_id']:r for r in primary}; drops=[]; repeats=[]; associations=[]; timing=[]
    for row in primary:
        case=index[row['case_id']]['case']
        if case['study']=='robustness' and case['condition']!='clean':
            clean={**case,'condition':'clean'}; clean_id=spec.digest(clean)
            if clean_id in lookup:
                reference=lookup[clean_id]
                drops.append({'case_id':row['case_id'],'clean_case_id':clean_id,
                    **{k:case[k] for k in ('protocol','condition','classifier','preprocessor','representation')},
                    'eligible_for_inference':row['eligible_for_inference'] and reference['eligible_for_inference'],
                    'signed_macro_f1_drop':reference['macro_f1']-row['macro_f1'],
                    'relative_macro_f1_drop':(reference['macro_f1']-row['macro_f1'])/reference['macro_f1'] if reference['macro_f1'] else None})
        if '__r' in case['condition'] and case['representation'] in stats.base_reps():
            base_condition=spec.BY_CONDITION[case['condition']]['base']
            reference={**case,'study':'robustness','condition':base_condition}
            ref_id=spec.digest(reference)
            if ref_id in lookup:
                repeats.append({'case_id':row['case_id'],'reference_case_id':ref_id,
                    'repeat':spec.BY_CONDITION[case['condition']]['repeat'],'protocol':case['protocol'],
                    'macro_f1':row['macro_f1'],'reference_macro_f1':lookup[ref_id]['macro_f1'],
                    'eligible_for_inference':row['eligible_for_inference'] and lookup[ref_id]['eligible_for_inference']})
    for model in spec.MODELS:
        for prep in spec.PREPS:
            selected_fits=[r for r in fits if r['job']['classifier']==model and r['job']['preprocessor']==prep]
            timing.append({'classifier':model,'preprocessor':prep,'fits':len(selected_fits),
                'sum_fit_seconds':float(sum(r['fit_seconds'] for r in selected_fits)),
                'median_fit_seconds':float(np.median([r['fit_seconds'] for r in selected_fits])),
                'median_compressed_model_bytes':float(np.median([r['saved_model']['bytes'] for r in selected_fits])),
                'sum_prediction_seconds':float(sum(r['prediction_seconds'] for r in selected_fits)),
                'nonconverged_fits':sum(not r['eligible_for_inference'] for r in selected_fits)})
    for noise in spec.old.NOISES:
        for operator in ('DWT_D3_2500_5000Hz','DWT_D1_10000_20000Hz','FFT_2500_5000Hz'):
            xs=[]; ys=[]
            for snr in spec.base.SNR_LEVELS:
                condition=noise+'__'+str(snr)
                for prep in spec.PREPS:
                    relevant=[r['median_tpi_noisy'] for r in signal_summary if r['condition']==condition and r['preprocessor']==prep and r['operator']==operator]
                    scores=[r['macro_f1'] for r in primary if r['study']=='robustness' and r['protocol']=='deployment' and r['representation']=='MULTI18' and r['condition']==condition and r['preprocessor']==prep and r['eligible_for_inference']]
                    if relevant and scores: xs.append(float(np.median(relevant))); ys.append(float(np.median(scores)))
            valid=len(xs)>=3 and np.ptp(xs)>0 and np.ptp(ys)>0
            associations.append({'noise':noise,'operator':operator,'points':len(xs),
                'spearman_rho':float(spearmanr(xs,ys).statistic) if valid else None,
                'status':'EXPLORATORY_CONDITION_DEPENDENT_NOT_CAUSAL','p_value_reported':False})
    c.table(output/'signed_robustness_drops.csv',drops); c.table(output/'noise_repeat_sensitivity.csv',repeats)
    c.table(output/'fit_prediction_timing.csv',timing); c.table(output/'tpi_exploratory_association.csv',associations)
    # Exact paired constituent disagreement for the predeclared severe-noise slices.
    complement=[]
    for model in spec.MODELS:
        for noise in spec.old.NOISES:
            a,b=(spec.digest(stats.robust(model,noise+'__10',rep)) for rep in ('RMS6','RAW12'))
            left,right=pooled(output/index[a]['path']),pooled(output/index[b]['path']); y=left['truth']; w=stats.weights(rows)
            for name,mask in [('disagreement',left['predicted']!=right['predicted']),
                ('RMS_correct_RAW_wrong',(left['predicted']==y)&(right['predicted']!=y)),
                ('RAW_correct_RMS_wrong',(left['predicted']!=y)&(right['predicted']==y)),
                ('joint_wrong',(left['predicted']!=y)&(right['predicted']!=y))]:
                complement.append({'classifier':model,'condition':noise+'__10','error_component':name,
                    'standardized_fraction':float(np.average(mask,weights=w)),
                    'eligible_for_inference':index[a]['eligible_for_inference'] and index[b]['eligible_for_inference']})
    c.table(output/'constituent_error_complementarity.csv',complement)

def aggregate(assignment,transport,root,rows,graphics_python,graphics_font):
    output=root/'final_analysis'; output.mkdir()
    comparisons=c.read(root/'comparison_plan.json')
    c.require(comparisons==stats.comparison_plan(),'Prospective inference plan changed')
    with threadpool_limits(limits=1):
        index,fits=collect(assignment,transport,output,rows)
        scores=metrics(index,rows,output); summaries=diagnostic_summaries(root,rows,output)
        tests=inference(index,rows,output,comparisons)
        secondary_outputs(scores,index,rows,fits,output,summaries)
        from c1e32_figures import render
        render(root,output,scores,index,rows,tests,summaries)
    import subprocess
    print('[PHASE] Export every scientific graph to vector PDF with embedded fonts',flush=True)
    subprocess.run([str(graphics_python),'-B',str(Path(__file__).with_name('c1e32_pdf.py')),
        '--directory',str(output/'figures'),'--font',str(graphics_font)],check=True)
    c.write(output/'analysis_receipt.json',{'status':'COMPLETE_STATISTICAL_ANALYSIS_NOT_PHYSICAL_CERTIFICATION',
        'dataset_view_sha256':spec.DATASET_SHA,'scope_sha256':spec.digest(spec.scope()),
        'fits_completed':len(fits),'pooled_cases':len(index),'paired_contrasts':len(tests),
        'excluded_pooled_cases':sum(not r['eligible_for_inference'] for r in index.values()),
        'statistics':spec.STATS,'scientific_limits':spec.LIMITS,
        'artifacts':{p.relative_to(output).as_posix():c.sha(p) for p in sorted(output.rglob('*')) if p.is_file() and 'shards' not in p.parts and 'oof' not in p.parts}})
