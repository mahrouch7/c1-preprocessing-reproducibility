"""Family/class standardization and paired cluster inference, no fitting."""
from __future__ import annotations
from collections import Counter
import numpy as np
from scipy.stats import beta
from c1e27_analysis import score_matrix, holm
import c1e32_spec as spec
import c1e32_common as c

def weights(rows,population='family_class_standardized',mask=None):
    mask=np.ones(len(rows),bool) if mask is None else np.asarray(mask,dtype=bool)
    if population=='pooled': return mask.astype(float)
    c.require(population=='family_class_standardized','Unknown estimand')
    cells=Counter((r['family_id'],r['fault']) for r,m in zip(rows,mask) if m)
    return np.array([1/cells[(r['family_id'],r['fault'])] if m else 0 for r,m in zip(rows,mask)])

def matrices(rows,truth,predicted,population='family_class_standardized',mask=None):
    truth,predicted=np.asarray(truth),np.asarray(predicted)
    c.require(truth.shape==predicted.shape==(len(rows),) and np.all((truth>=0)&(truth<12)) and np.all((predicted>=0)&(predicted<12)),'Invalid labels')
    families,index=np.unique([r['family_id'] for r in rows],return_inverse=True)
    result=np.zeros((len(families),144)); w=weights(rows,population,mask)
    np.add.at(result,(index,12*truth+predicted),w)
    c.require(np.all(result.sum(axis=0).reshape(12,12).sum(axis=1)>0),'A score population lacks a target class')
    return result

def score(rows,truth,predicted,population='family_class_standardized',mask=None):
    matrix=matrices(rows,truth,predicted,population,mask).sum(axis=0).reshape(12,12)
    result={k:float(v) for k,v in score_matrix(matrix).items()}
    actual=matrix.sum(axis=1); guesses=matrix.sum(axis=0); diagonal=matrix.diagonal()
    result.update(healthy_false_alarm=float((actual[0]-diagonal[0])/actual[0]),
        missed_fault=float(matrix[1:,0].sum()/actual[1:].sum()))
    return result,matrix

def interval(cluster_matrices,repetitions=10000,seed=42):
    rng=np.random.Generator(np.random.PCG64(seed)); g=len(cluster_matrices); values=[]
    for start in range(0,repetitions,250):
        w=rng.multinomial(g,np.full(g,1/g),size=min(250,repetitions-start))
        values.extend(score_matrix((w@cluster_matrices).reshape(-1,12,12))['macro_f1'].tolist())
    low,high=np.quantile(values,[.025,.975])
    return {'lower_95':float(low),'upper_95':float(high),'families':g}

def paired(left,right,rows,population='family_class_standardized',bootstrap=10000,swaps=50000,seed=42):
    c.require(np.array_equal(left['row_ids'],right['row_ids']) and np.array_equal(left['truth'],right['truth']),'Paired identities differ')
    a=matrices(rows,left['truth'],left['predicted'],population)
    b=matrices(rows,right['truth'],right['predicted'],population)
    delta=float(score_matrix(a.sum(axis=0).reshape(12,12))['macro_f1']-score_matrix(b.sum(axis=0).reshape(12,12))['macro_f1'])
    if np.array_equal(a,b):
        return {'difference':delta,'lower_95':0.,'upper_95':0.,'permutation_p':1.,'p_mc_lower_99':1.,'p_mc_upper_99':1.,'families':len(a)}
    rng=np.random.Generator(np.random.PCG64(seed)); g=len(a); diffs=[]; exceed=0
    for start in range(0,bootstrap,250):
        w=rng.multinomial(g,np.full(g,1/g),size=min(250,bootstrap-start))
        diffs.extend((score_matrix((w@a).reshape(-1,12,12))['macro_f1']-score_matrix((w@b).reshape(-1,12,12))['macro_f1']).tolist())
    total=a.sum(axis=0)+b.sum(axis=0)
    for start in range(0,swaps,500):
        signs=rng.integers(0,2,size=(min(500,swaps-start),g)); pa=b.sum(axis=0)+signs@(a-b)
        null=score_matrix(pa.reshape(-1,12,12))['macro_f1']-score_matrix((total-pa).reshape(-1,12,12))['macro_f1']
        exceed+=int(np.count_nonzero(abs(null)>=abs(delta)-1e-14))
    lo,hi=np.quantile(diffs,[.025,.975]); pl=0. if exceed==0 else float(beta.ppf(.005,exceed,swaps-exceed+1)); pu=1. if exceed==swaps else float(beta.ppf(.995,exceed+1,swaps-exceed))
    return {'difference':delta,'lower_95':float(lo),'upper_95':float(hi),'permutation_p':(exceed+1)/(swaps+1),
        'p_mc_lower_99':(1+swaps*pl)/(swaps+1),'p_mc_upper_99':(1+swaps*pu)/(swaps+1),'families':g}

def robust(model,condition,rep='MULTI18',prep='none',protocol='deployment'):
    return {'study':'robustness','representation':rep,'normalizer':spec.default_scale(model),'anchor_only':False,
        'density_balanced':False,'classifier':model,'preprocessor':prep,'protocol':protocol,'condition':condition}

def comparison_plan():
    result=[]
    def add(family,left,right):
        value={'family':family,'left':left,'right':right}; value['comparison_id']=spec.digest(value); result.append(value)
    for model in spec.MODELS:
        for noise in spec.old.NOISES:
            condition=noise+'__10'
            for left in ('MULTI18','LATE_SIMPLE'):
                for right in ('RAW12','RMS6'):
                    add(('early' if left=='MULTI18' else 'late')+'_vs_'+right,
                        robust(model,condition,left),robust(model,condition,right))
            for prep in spec.PREPS[1:]:
                add('denoiser_vs_none',robust(model,condition,prep=prep),robust(model,condition))
        for rep in base_reps():
            for scale in spec.NORMALIZERS:
                if scale=='none': continue
                def scaler_case(value):
                    case=robust(model,'clean',rep); case['normalizer']=value
                    if value!=spec.default_scale(model): case['study']='scaling'
                    return case
                add('scaler_vs_none',scaler_case(scale),scaler_case('none'))
        if model in spec.old.EXTRA_MODELS:
            for condition in spec.EXTRA[1:]:
                for root in ('RAW','MULTI'):
                    original='RAW12' if root=='RAW' else 'MULTI18'
                    skew={**robust(model,condition,root+'_SKEW'),'study':'descriptors'}
                    bundle={**robust(model,condition,root+'_PLUS'),'study':'descriptors'}
                    add('skew_vs_base',skew,robust(model,condition,original))
                    add('bundle_vs_base',bundle,robust(model,condition,original))
                    add('bundle_vs_skew',bundle,skew)
    for model in spec.CONTROL_MODELS:
        for prep in ('none','median'):
            for condition in spec.SENSITIVITY:
                for rep in base_reps():
                    main=robust(model,condition,rep,prep)
                    add('healthy_enrichment',main,{**main,'study':'support','anchor_only':True})
                    add('nominal_density',main,{**main,'study':'density','density_balanced':True})
                for suffix,combined in [('RMS','RMS6'),('RAW','RAW12'),('MULTI','MULTI18')]:
                    for channel in ('I','V'):
                        add('I_V_ablation',robust(model,condition,combined,prep),
                            {**robust(model,condition,channel+'_'+suffix,prep),'study':'channels'})
    c.require(len({r['comparison_id'] for r in result})==len(result),'Duplicate inference contrast')
    return sorted(result,key=lambda r:r['comparison_id'])

def base_reps(): return ('RMS6','RAW12','MULTI18')

def expected_cases():
    cases={}
    for unit in spec.learning_plan():
        if unit['fold']!=1: continue
        for job in spec.jobs(unit['preprocessor'],1,unit['classifier']):
            for protocol,condition in spec.evaluations(job):
                case=spec.case(job,protocol,condition); cases[spec.digest(case)]=case
        if unit['classifier']=='dummy': continue
        conditions=list(spec.BASE_CONDITIONS)
        if unit['classifier'] in spec.CONTROL_MODELS and unit['preprocessor'] in ('none','median'):
            conditions += [v['name'] for v in spec.CONDITIONS if v['repeat']]
        for protocol in ('deployment','matched'):
            for condition in conditions:
                # Matched repeat cases are evaluated from repeat_noise jobs, not base fusions.
                if '__r' in condition and protocol=='matched': continue
                for rep in ('LATE_SIMPLE','LATE_ORACLE_SNR'):
                    case=robust(unit['classifier'],condition,rep,unit['preprocessor'],protocol)
                    cases[spec.digest(case)]=case
    return cases
