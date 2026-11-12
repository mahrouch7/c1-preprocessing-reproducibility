"""Prospective load-family evaluation. Imports never fit, read data or submit."""
from __future__ import annotations
import hashlib
import json
import c1e25 as base
import c1e26 as ext
import c1e27_spec as old

EXPERIMENT = 'C1-E32'
PROTOCOL = 'C1-load-family-preprocessing-fusion-v1'
DATASET_SHA = '61c43740c1c589634b0922070a6cfab2e559c09915e5fb2e4a766c39ac247aee'
LABELS, MODELS, PREPS = base.LABELS, old.MODELS, old.PREPROCESSORS
FEATURES, VERSIONS, LIMITS = old.FEATURES, old.VERSIONS, base.LIMITS
NORMALIZERS = old.NORMALIZERS
REPS = dict(base.REPRESENTATIONS)
REPS.update(RAW_SKEW=list(range(6,18))+[18+3*i for i in range(6)],
            MULTI_SKEW=list(range(18))+[18+3*i for i in range(6)],
            RAW_PLUS=list(range(6,36)), MULTI_PLUS=list(range(36)))
for name, channels in [('I',range(3)),('V',range(3,6))]:
    rms=list(channels); raw=[6+2*i+j for i in channels for j in range(2)]
    REPS.update({name+'_RMS':rms,name+'_RAW':raw,name+'_MULTI':rms+raw})
BASE_CONDITIONS = old.CONDITIONS
CONDITIONS = [{'name':v,'base':v,'repeat':0} for v in BASE_CONDITIONS]
CONDITIONS += [{'name':f'{v}__r{r}','base':v,'repeat':r}
               for r in (1,2) for v in ('awgn__20','mixed_gain_additive__10')]
BY_CONDITION = {v['name']:v for v in CONDITIONS}
SENSITIVITY = ('clean','awgn__20','mixed_gain_additive__10')
EXTRA = old.EXTRA_CONDITIONS
CONTROL_MODELS = ('logistic_regression','random_forest')
STATS = {'primary_population':'family_class_standardized','secondary_population':'pooled',
    'cluster':'family_id','folds':5,'bootstrap_resamples':10000,'paired_swaps':50000,
    'alpha':.05,'multiplicity':'Holm_per_prospective_contrast_family',
    'primary_metric':'macro_f1','secondary_metrics':['balanced_accuracy','mcc'],
    'primary_scope':'new-cohort prospective comparisons; no formal preregistration claimed',
    'uncertainty_scope':'conditional on fixed trained outer models and simulated cohort',
    'selection_policy':'fixed conditions; no test-score tuning or best-per-condition deployment',
    'probability_decision':'argmax_aligned_probabilities_for_all_main_comparisons',
    'model_warnings':'retain all; convergence warning excludes affected pooled case; no refit'}

def canonical(v):
    return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode('ascii')

def digest(v): return hashlib.sha256(canonical(v)).hexdigest()

def default_scale(model): return old.default_normalizer(model)

def jobs(prep,fold,model):
    if prep not in PREPS or fold not in range(1,6) or model not in (*MODELS,'dummy'):
        raise ValueError('Unknown unit identity')
    result=[]
    def add(study,rep,condition,scale=None,anchor_only=False,estimator=None,density=False):
        v={'study':study,'representation':rep,'training_condition':condition,
           'normalizer':scale or default_scale(model),'anchor_only':anchor_only,'density_balanced':density,
           'classifier':estimator or model,'fold':fold,'preprocessor':prep}
        v['job_id']=digest(v); result.append(v)
    if model=='dummy':
        if prep!='none': raise ValueError('Dummy only in unfiltered units')
        for name in ('dummy_prior','dummy_majority'):
            add('baseline','RMS6','clean','none',estimator=name)
        return result
    for rep in base.REPRESENTATIONS:
        for condition in BASE_CONDITIONS: add('robustness',rep,condition)
    if prep=='none':
        for rep in base.REPRESENTATIONS:
            for condition in SENSITIVITY:
                for scale in NORMALIZERS:
                    if scale!=default_scale(model): add('scaling',rep,condition,scale)
        if model in old.EXTRA_MODELS:
            for rep in ('RAW_SKEW','MULTI_SKEW','RAW_PLUS','MULTI_PLUS'):
                for condition in EXTRA: add('descriptors',rep,condition)
    if prep in ('none','median') and model in CONTROL_MODELS:
        for rep in ('I_RMS','V_RMS','I_RAW','V_RAW','I_MULTI','V_MULTI'):
            for condition in SENSITIVITY: add('channels',rep,condition)
        for rep in base.REPRESENTATIONS:
            for condition in SENSITIVITY:
                add('support',rep,condition,anchor_only=True)
                add('density',rep,condition,density=True)
        for rep in base.REPRESENTATIONS:
            for condition in CONDITIONS:
                if condition['repeat']: add('repeat_noise',rep,condition['name'])
    return result

def evaluations(job):
    if job['training_condition']!='clean': return [('matched',job['training_condition'])]
    if job['study'] in ('robustness','baseline'):
        levels=list(BASE_CONDITIONS)
        if job['classifier'] in CONTROL_MODELS and job['preprocessor'] in ('none','median'):
            levels += [v['name'] for v in CONDITIONS if v['repeat']]
    elif job['study']=='descriptors': levels=list(EXTRA)
    else: levels=list(SENSITIVITY)
    return [('deployment',v) for v in levels]+[('matched','clean')]

def case(job,protocol,condition):
    return {k:job[k] for k in ('study','representation','normalizer','anchor_only','density_balanced','classifier','preprocessor')} | {
        'protocol':protocol,'condition':condition}

def learning_plan():
    units=[]
    for p,prep in enumerate(PREPS):
        for fold in range(1,6):
            for m,model in enumerate(MODELS):
                units.append({'preprocessor':prep,'fold':fold,'classifier':model,
                    'role':'PC1' if (p+fold+m)%2==0 else 'PC2',
                    'expected_fits':len(jobs(prep,fold,model))})
    for fold in range(1,6):
        units.append({'preprocessor':'none','fold':fold,'classifier':'dummy',
            'role':'PC1' if fold%2 else 'PC2','expected_fits':2})
    return units

def scope():
    plan=learning_plan()
    return {'experiment_id':EXPERIMENT,'protocol':PROTOCOL,'dataset_view_sha256':DATASET_SHA,
        'rows':1839,'families':39,'labels':LABELS,'samples':8000,'sampling_rate_hz':40000,
        'preprocessors':PREPS,'conditions':CONDITIONS,'representations':REPS,
        'features':FEATURES,'normalizers':NORMALIZERS,'models':old.MODEL_CONTRACTS,
        'statistics':STATS,'signal_contract':old.SIGNAL_CONTRACT,'versions':VERSIONS,
        'fit_budget':sum(u['expected_fits'] for u in plan),'fit_budget_per_role':{
            r:sum(u['expected_fits'] for u in plan if u['role']==r) for r in ('PC1','PC2')},
        'preprocessing_chunks':10,'learning_units':len(plan),'historical_fits_reused':0,
        'new_native_simulations':0,'conditional_not_in_core':['temporal_DL','learned_fusion','granularity','external_data'],
        'limits':LIMITS}

def validate_rows(rows,production=True):
    if not isinstance(rows,list) or not rows: raise ValueError('Metadata list required')
    if [r['Id'] for r in rows]!=list(range(len(rows))): raise ValueError('Row identities/order changed')
    if len({r['window_id'] for r in rows})!=len(rows): raise ValueError('Duplicate windows')
    if len({r['acquisition_id'] for r in rows})!=len(rows): raise ValueError('Repeated acquisition across analysis rows')
    families={}
    for r in rows:
        if r['fault'] not in LABELS or r['fold'] not in range(1,6): raise ValueError('Label/fold invalid')
        if r['family_id'] in families and families[r['family_id']]!=r['fold']: raise ValueError('Family crosses folds')
        families[r['family_id']]=r['fold']
        if r['source_group_id']!=r['family_id']: raise ValueError('Source group drift')
    for fold in range(1,6):
        for flag in (True,False):
            if {r['fault'] for r in rows if (r['fold']==fold)==flag}!=set(LABELS):
                raise ValueError('Missing train/test class')
    if production:
        from collections import Counter
        if len(rows)!=1839 or len(families)!=39 or Counter(r['fault'] for r in rows)!={'noFault':156,**{v:153 for v in LABELS[1:]}}:
            raise ValueError('Wrong production cohort')
    return rows

def validate_configuration(config,expected_fits):
    if config['experiment_id']!=EXPERIMENT or config['scope_sha256']!=digest(scope()):
        raise ValueError('Frozen scope changed')
    if config['stage']=='A': wanted=0
    elif config['stage']=='B': wanted=len(jobs(config['preprocessor'],config['fold'],config['classifier']))
    else: raise ValueError('Unknown scientific stage')
    if wanted!=expected_fits or config['expected_fits']!=wanted: raise ValueError('Fit ceiling changed')
