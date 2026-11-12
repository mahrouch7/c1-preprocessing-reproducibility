"""Descriptive comparison of frozen OOF probabilities; no training or inference tests."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone
import csv
import importlib.metadata
import json
import platform
import sys
import numpy as np
from .data import LABELS, validate_rows, make_split
from .io import digest, read_json, require, sha256, write_json, write_csv
from .metrics import score, family_class_weights

ROOT=Path(__file__).resolve().parents[2]
CASES=[('shared_clean','clean'),('clean_trained','awgn__20'),('noise_matched','awgn__20'),
       ('clean_trained','mixed_gain_additive__10'),('noise_matched','mixed_gain_additive__10')]
METRICS=('macro_f1','balanced_accuracy','mcc','healthy_false_alarm_rate','missed_fault_rate','brier','log_loss','accuracy')
PERCENT_METRICS={'macro_f1','balanced_accuracy','healthy_false_alarm_rate','missed_fault_rate','accuracy'}
SCALERS={'logistic_regression':'standard','random_forest':'none','naive_bayes':'none','knn':'standard',
         'decision_tree':'none','svm':'standard','gradient_boosting':'none','mlp':'standard'}
NAMES={'logistic_regression':'LR','random_forest':'RF','naive_bayes':'NB','knn':'kNN',
       'decision_tree':'DT','svm':'SVM','gradient_boosting':'GB','mlp':'MLP'}
PARTS=('healthy_to_fault_mass_pp','fault_to_healthy_mass_pp','abc_pair_mass_pp','other_fault_error_mass_pp')


def resolved(relative):return (ROOT/relative).resolve()


def safe_child(root,relative):
    path=(Path(root)/relative).resolve()
    require(Path(root).resolve() in path.parents and path.is_file(),'Unsafe/missing input: '+str(relative))
    return path


def aligned(ids,p,truth,predicted,rows,expected_ids):
    ids=np.asarray(ids)
    require(ids.ndim==1 and np.issubdtype(ids.dtype,np.integer),'Nonintegral row identities')
    ids=ids.astype(np.int64)
    require(len(ids)==len(set(ids.tolist())) and set(ids.tolist())==set(expected_ids),'Missing/duplicate/unexpected test rows')
    p=np.asarray(p,dtype=np.float64)
    require(p.shape==(len(ids),12) and np.isfinite(p).all(),'Invalid probability shape/values')
    require(np.all((p>=0)&(p<=1)) and np.allclose(p.sum(axis=1),1,atol=1e-6,rtol=0),'Invalid probabilities')
    expected=np.array([LABELS.index(rows[int(i)]['fault']) for i in ids])
    require(np.array_equal(truth,expected),'Saved truth differs from metadata')
    require(np.array_equal(predicted,p.argmax(axis=1)),'Saved decision is not probability argmax')
    order=np.argsort(ids)
    return ids[order],p[order]


def pool(chunks,rows):
    ids=np.concatenate([c[0] for c in chunks]);p=np.concatenate([c[1] for c in chunks])
    require(len(ids)==len(rows) and len(set(ids.tolist()))==len(rows),'Incomplete/duplicated outer-fold coverage')
    order=np.argsort(ids);ids,p=ids[order],p[order]
    require(np.array_equal(ids,np.arange(len(rows))),'OOF identity drift')
    return p


def error_parts(result):
    cm=np.asarray(result['weighted_confusion'],dtype=float);total=cm.sum()
    a,b=LABELS.index('abc'),LABELS.index('abcg')
    h=cm[0,1:].sum();f=cm[1:,0].sum();pair=cm[a,b]+cm[b,a]
    mask=np.ones_like(cm,dtype=bool);np.fill_diagonal(mask,False)
    mask[0,:]=False;mask[:,0]=False;mask[a,b]=False;mask[b,a]=False
    other=cm[mask].sum()
    parts=dict(zip(PARTS,(100*h/total,100*f/total,100*pair/total,100*other/total)))
    require(abs(sum(parts.values())-100*(1-result['balanced_accuracy']))<1e-9,'BA error masses do not partition total error')
    return {**parts,'total_ba_error_pp':100*(1-result['balanced_accuracy']),
        'abc_to_abcg_rate':float(cm[a,b]/cm[a].sum()),'abcg_to_abc_rate':float(cm[b,a]/cm[b].sum())}


def paired_outcomes(rows,pa,pb):
    y=np.array([LABELS.index(r['fault']) for r in rows]);a=pa.argmax(axis=1);b=pb.argmax(axis=1)
    w=family_class_weights(rows,np.arange(len(rows)));w=w/w.sum();ac=a==y;bc=b==y
    values={'both_correct':w[ac&bc].sum(),'a_only_correct':w[ac&~bc].sum(),
            'b_only_correct':w[~ac&bc].sum(),'both_wrong':w[~ac&~bc].sum()}
    require(abs(sum(values.values())-1)<1e-12,'Overlap masses do not sum to one')
    return {k:float(v) for k,v in values.items()}|{'decision_disagreement':float(w[a!=b].sum()),
        'ba_difference_pp_from_error_overlap':float(100*(values['a_only_correct']-values['b_only_correct']))}


def deltas(a,b):
    return {m+('_difference_pp' if m in PERCENT_METRICS else '_difference'):float((a[m]-b[m])*(100 if m in PERCENT_METRICS else 1)) for m in METRICS}


def describe(values):
    a=np.asarray(values,dtype=float)
    return {'mean':float(a.mean()),'sd':float(a.std(ddof=1)) if len(a)>1 else None}


def select_ml(index,exposure,condition):
    protocol='matched' if exposure=='noise_matched' else 'deployment'
    selected=[]
    for case_id,entry in index.items():
        c=entry['case']
        if (c['study']=='robustness' and c['preprocessor']=='none' and c['condition']==condition
            and c['protocol']==protocol and not c['anchor_only'] and not c['density_balanced']
            and c['representation'] in ('RMS6','RAW12','MULTI18') and c['classifier'] in SCALERS
            and c['normalizer']==SCALERS[c['classifier']]):selected.append((case_id,entry))
    identities=[(e['case']['classifier'],e['case']['representation']) for _,e in selected]
    wanted={(c,r) for c in SCALERS for r in ('RMS6','RAW12','MULTI18')}
    require(len(selected)==24 and set(identities)==wanted,'Expected exactly 24 fixed reference ML pipelines')
    return sorted(selected,key=lambda item:(list(SCALERS).index(item[1]['case']['classifier']),item[1]['case']['representation']))


def load_neural(name,entry,rows,inputs):
    root=resolved(entry['run_relative']);audit_path=resolved(entry['audit_relative'])
    require(sha256(audit_path)==entry['audit_sha256'],'Retrieval audit changed: '+name)
    audit=read_json(audit_path);manifest=read_json(root/'run_manifest.json')
    require(sha256(root/'run_manifest.json')==audit['run_manifest_sha256']==entry['manifest_sha256'],'Run identity drift: '+name)
    require(manifest['experiment_id']==entry['experiment_id'] and manifest['profile']=='pilot','Wrong temporal experiment/profile')
    require(manifest['git']=={'commit':entry['source_commit'],'dirty':False},'Wrong execution source')
    require(manifest['model']['parameters']==entry['parameters'],'Architecture mismatch')
    require(manifest['config_sha256']==digest(manifest['config']),'Scientific configuration identity mismatch')
    if name=='InceptionTime':
        receipts={(r['training_condition'],r['fold'],r['seed']):r['receipt_sha256'] for r in audit['completed_job_receipts']}
        require(audit['completed_jobs']==45 and audit['completed_prediction_sets']==75,'Incomplete E43 import')
    else:
        receipts={(r['condition'],r['fold'],r['seed']):r['completed_receipt_sha256'] for r in audit['jobs']}
        require(audit['jobs_verified']==45 and audit['prediction_pairs_verified']==75,'Incomplete E33 import')
    cfg=manifest['config'];groups=defaultdict(list);costs=[];timings=[];prediction_count=0
    for condition in cfg['profiles']['pilot']:
        for fold in cfg['training']['outer_folds']:
            for seed in cfg['training']['seeds']:
                directory=root/'jobs'/condition/f'fold{fold:02d}'/f'seed{seed}'
                completed_path=directory/'completed.json'
                require(sha256(completed_path)==receipts[(condition,fold,seed)],'Changed completion receipt')
                saved=read_json(completed_path);identity=read_json(directory/'identity.json');split=read_json(directory/'split.json')
                require(saved['status']=='COMPLETE' and saved['fit_procedures']==2,'Unexpected fit receipt')
                require(saved['identity_hash']==digest(identity) and identity['run_manifest_sha256']==sha256(root/'run_manifest.json'),'Changed job identity')
                require(identity['job']=={'training_condition':condition,'fold':fold,'seed':seed},'Wrong job')
                require(identity['training_cache_sha256']==entry['condition_cache_sha256'][condition],'Different frozen training inputs')
                require(identity['split_sha256']==digest(split),'Changed split')
                require(split==make_split(rows,fold,cfg['training']['validation_family_fraction'],cfg['training']['inner_split_seed']),
                        'Nested family partition differs from the declared common protocol')
                expected=[r['Id'] for r in rows if r['fold']==fold]
                require(set(split['test'])==set(expected),'Outer test set changed')
                for rel,h in saved['artifacts'].items():
                    if rel.endswith('.pt'):continue  # unused checkpoints were verified by the importer
                    f=safe_child(directory,rel);require(sha256(f)==h,'Changed used artifact: '+str(f))
                selection=read_json(directory/'selection/summary.json');refit=read_json(directory/'refit/summary.json')
                require(selection['best_epoch']==refit['epochs_completed'],'Full refit did not use selected epoch count')
                for phase,ids in [('selection',split['inner_train']),('refit',split['outer_train'])]:
                    scaler=read_json(directory/phase/'scaler.json')
                    require(scaler['fit_row_ids']==ids,'Scaler fit IDs changed')
                    train_families={rows[i]['family_id'] for i in ids};test_families={rows[i]['family_id'] for i in expected}
                    require(not train_families&test_families,'Family leakage in scaler/model input')
                costs.append({'model':name,'training_condition':condition,'fold':fold,'seed':seed,
                    'selected_epochs':selection['best_epoch'],'selection_epochs':selection['epochs_completed'],
                    'refit_epochs':refit['epochs_completed'],'selection_seconds':selection['elapsed_seconds'],
                    'refit_seconds':refit['elapsed_seconds'],'parameters':entry['parameters']})
                catalogue=read_json(directory/'prediction_catalogue.json')
                wanted=cfg['profiles']['pilot'] if condition=='clean' else [condition]
                require([r['condition'] for r in catalogue]==wanted,'Prediction domain mismatch')
                for item in catalogue:
                    f=safe_child(directory/'predictions',item['prediction_file'])
                    require(sha256(f)==item['prediction_sha256'],'Prediction checksum mismatch')
                    with np.load(f,allow_pickle=False) as z:
                        ids,p=aligned(z['row_ids'],z['probabilities'],z['truth'],z['predicted'],rows,expected)
                        require(np.array_equal(z['family_id'],[rows[int(i)]['family_id'] for i in z['row_ids']]),'Family identity mismatch')
                    check=score(rows,ids,p)
                    require(all(abs(check[m]-item['metrics'][m])<=1e-10 for m in METRICS),'Saved fold metrics do not reproduce')
                    exposure='shared_clean' if item['condition']=='clean' else ('clean_trained' if condition=='clean' else 'noise_matched')
                    groups[(exposure,item['condition'],seed)].append((ids,p));prediction_count+=1
                    timings.append({'model':name,'exposure':exposure,'condition':item['condition'],'seed':seed,'fold':fold,
                        'elapsed_prediction_seconds':item['elapsed_prediction_seconds'],'test_windows':len(ids)})
                    inputs.append({'role':name+'_predictions','path':f.relative_to(root).as_posix(),'sha256':item['prediction_sha256']})
    require(len(receipts)==45 and prediction_count==75 and len(groups)==15,'Incomplete neural prediction grid')
    results={}
    for (exposure,condition,seed),chunks in groups.items():
        require(len(chunks)==5,'Missing outer fold')
        results[(name,exposure,condition,seed)]=pool(chunks,rows)
    return manifest,results,costs,timings


def analyze(policy_path,destination):
    policy_path=Path(policy_path);policy=read_json(policy_path);destination=Path(destination).resolve()
    require(not destination.exists(),'Use a new output folder; earlier analyses are immutable')
    versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','scikit-learn')}
    require(versions==policy['scientific_versions'] and sys.version_info[:2]==(3,12),'Analysis dependency drift')
    for rel in [policy['metadata_relative'],policy['ml_root_relative'],*[r['run_relative'] for r in policy['neural'].values()]]:
        protected=resolved(rel)
        require(destination!=protected and protected not in destination.parents,'Output is inside protected inputs')
    metadata=resolved(policy['metadata_relative']);require(sha256(metadata)==policy['metadata_sha256'],'Metadata bytes changed')
    rows=validate_rows(read_json(metadata));require(len(rows)==1839,'Wrong cohort')
    inputs=[{'role':'metadata','path':policy['metadata_relative'],'sha256':policy['metadata_sha256']}]
    predictions={};manifests={};costs=[];timings=[]
    for name,entry in policy['neural'].items():
        manifests[name],values,c,t=load_neural(name,entry,rows,inputs);predictions.update(values);costs+=c;timings+=t
    a,b=manifests['InceptionTime'],manifests['ResNet']
    for key in ('source_pins','dataset_view_sha256','expected_rows','expected_families','samples','sampling_rate_hz','channels','labels','training','profiles'):
        require(a['config'][key]==b['config'][key],'Neural protocol mismatch: '+key)
    require(a['runtime_key']==b['runtime_key'],'Neural execution stacks/GPU differ; declare before comparison')
    mlroot=resolved(policy['ml_root_relative']);index_path=mlroot/'pooled_prediction_index.json'
    require(sha256(index_path)==policy['ml_index_sha256'],'ML index changed')
    index=read_json(index_path);mlinfo={}
    for exposure,condition in CASES:
        for case_id,entry in select_ml(index,exposure,condition):
            c=entry['case'];name=NAMES[c['classifier']]+'--'+c['representation']
            f=safe_child(mlroot,entry['path']);require(sha256(f)==entry['sha256'],'ML prediction bytes changed')
            with np.load(f,allow_pickle=False) as z:_,p=aligned(z['row_ids'],z['probabilities'],z['truth'],z['predicted'],rows,range(len(rows)))
            predictions[(name,exposure,condition,None)]=p
            mlinfo[(name,exposure,condition)]={'classifier':c['classifier'],'representation':c['representation'],
                'normalizer':c['normalizer'],'E32_inferential_eligibility':entry['eligible_for_inference'],'case_id':case_id}
            inputs.append({'role':'ML_predictions','path':entry['path'],'sha256':entry['sha256']})
    require(len(predictions)==150,'Expected 30 neural and 120 ML OOF evaluations')
    scores={k:score(rows,np.arange(len(rows)),p) for k,p in predictions.items()}
    reference=resolved(policy['resnet_reference_relative'])
    require(sha256(reference)==policy['resnet_reference_sha256'],'Existing ResNet reference CSV changed')
    with reference.open(encoding='utf-8-sig',newline='') as f:reference_rows=list(csv.DictReader(f))
    require(len(reference_rows)==15,'Expected 15 existing ResNet seed/case references')
    for r in reference_rows:
        result=scores[('ResNet',r['exposure'],r['condition'],int(r['seed']))]
        require(all(abs(result[m]-float(r[m]))<=1e-10 for m in METRICS if m in r),'Existing ResNet manuscript scores do not reproduce')
    destination.mkdir(parents=True);(destination/'oof').mkdir()
    records=[];classwise=[];confusions=[];summary=[];groups=defaultdict(list)
    for key,result in sorted(scores.items(),key=lambda kv:tuple(str(x) for x in kv[0])):
        name,exposure,condition,seed=key;neural=seed is not None
        base={'model':name,'model_family':'neural' if neural else 'ML','exposure':exposure,'condition':condition,'seed':seed if neural else ''}
        record={**base,**{m:result[m] for m in METRICS},**error_parts(result)}
        if not neural:record.update(mlinfo[(name,exposure,condition)])
        records.append(record);groups[(name,exposure,condition)].append(record)
        cm=np.asarray(result['weighted_confusion']);precision=np.divide(np.diag(cm),cm.sum(axis=0),out=np.zeros(12),where=cm.sum(axis=0)>0)
        recall=np.diag(cm)/cm.sum(axis=1);f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(12),where=precision+recall>0)
        require(abs(float(f1.mean())-result['macro_f1'])<1e-10,'Class-wise F1 does not reproduce Macro-F1')
        for i,label in enumerate(LABELS):
            classwise.append({**base,'class':label,'precision':precision[i],'recall':recall[i],'f1':f1[i],'weighted_support':cm[i].sum()})
            for j,predicted in enumerate(LABELS):confusions.append({**base,'true_class':label,'predicted_class':predicted,'weighted_mass':cm[i,j]})
        if neural:
            tag=f'{name}_seed{seed}_{exposure}_{condition}'
            np.savez_compressed(destination/'oof'/f'{tag}.npz',row_ids=np.arange(len(rows)),probabilities=predictions[key],truth=np.array([LABELS.index(r['fault']) for r in rows]))
            write_json(destination/'oof'/f'{tag}.json',result)
    metrics_with_errors=METRICS+PARTS+('total_ba_error_pp','abc_to_abcg_rate','abcg_to_abc_rate')
    means={}
    for key,items in groups.items():
        name,exposure,condition=key;entry={'model':name,'model_family':items[0]['model_family'],'exposure':exposure,'condition':condition,
            'initializations':len(items),'summary_scope':'mean_and_sample_sd_of_three_initializations' if len(items)==3 else 'single_fixed_ML_pipeline'}
        means[key]={}
        for m in metrics_with_errors:
            d=describe([r[m] for r in items]);entry[m+'_mean']=d['mean'];entry[m+'_sd']=d['sd'];means[key][m]=d['mean']
        summary.append(entry)
    overlaps=[];all_seed_pairs=[];vs_ml=[];vs_ml_mean=[];comparison=[];group_errors=[];exposures=[];class_summary=[]
    for exposure,condition in CASES:
        ml_names=sorted(n for n,e,c in groups if e==exposure and c==condition and n not in policy['neural'])
        ml_values=[scores[(n,exposure,condition,None)]['macro_f1'] for n in ml_names]
        for model in policy['neural']:
            result=means[(model,exposure,condition)];observed=[r for r in summary if (r['model'],r['exposure'],r['condition'])==(model,exposure,condition)][0]
            delta=np.asarray([result['macro_f1']-x for x in ml_values])
            comparison.append({'model':model,'exposure':exposure,'condition':condition,
                'macro_f1_mean':result['macro_f1'],'macro_f1_sd':observed['macro_f1_sd'],
                'ML_macro_f1_median':float(np.median(ml_values)),'ML_macro_f1_min':min(ml_values),'ML_macro_f1_max':max(ml_values),
                'ML_pipelines_below_neural_mean':int((delta>1e-12).sum()),'ML_pipelines_above_neural_mean':int((delta<-1e-12).sum()),
                'ML_pipelines_tied_to_neural_mean':int((np.abs(delta)<=1e-12).sum()),
                'mean_difference_vs_ML_median_pp':100*(result['macro_f1']-np.median(ml_values))})
            for ml in ml_names:
                baseline=scores[(ml,exposure,condition,None)]
                vs_ml_mean.append({'neural_model':model,'ML_pipeline':ml,'exposure':exposure,'condition':condition,**deltas(result,baseline)})
                for seed in (42,43,44):
                    key=(model,exposure,condition,seed);mk=(ml,exposure,condition,None)
                    vs_ml.append({'neural_model':model,'seed':seed,'ML_pipeline':ml,'exposure':exposure,'condition':condition,**deltas(scores[key],baseline)})
                    overlaps.append({'a_model':model,'a_seed':seed,'b_model':ml,'b_seed':'','exposure':exposure,'condition':condition,**paired_outcomes(rows,predictions[key],predictions[mk])})
        for sa in (42,43,44):
            for sb in (42,43,44):
                ka=('InceptionTime',exposure,condition,sa);kb=('ResNet',exposure,condition,sb)
                all_seed_pairs.append({'inception_seed':sa,'resnet_seed':sb,'exposure':exposure,'condition':condition,**deltas(scores[ka],scores[kb])})
                overlaps.append({'a_model':'InceptionTime','a_seed':sa,'b_model':'ResNet','b_seed':sb,'exposure':exposure,'condition':condition,**paired_outcomes(rows,predictions[ka],predictions[kb])})
        for label,names in [('InceptionTime',['InceptionTime']),('ResNet',['ResNet']),('ML',ml_names)]:
            units=[r for r in records if r['model'] in names and r['exposure']==exposure and r['condition']==condition]
            group_errors.append({'model_group':label,'exposure':exposure,'condition':condition,'units':len(units),
                **{m:float(np.mean([r[m] for r in units])) for m in PARTS+('total_ba_error_pp',)}})
        for model in list(policy['neural'])+ml_names:
            for label in LABELS:
                units=[r['recall'] for r in classwise if r['model']==model and r['exposure']==exposure and r['condition']==condition and r['class']==label]
                class_summary.append({'model':model,'exposure':exposure,'condition':condition,'class':label,'units':len(units),**describe(units)})
    mean_differences=[{'exposure':e,'condition':c,**deltas(means[('InceptionTime',e,c)],means[('ResNet',e,c)])} for e,c in CASES]
    class_lookup={(r['model'],r['exposure'],r['condition'],r['class']):r for r in class_summary}
    class_differences=[]
    for e,c in CASES:
        for label in LABELS:
            a=class_lookup[('InceptionTime',e,c,label)]['mean'];b=class_lookup[('ResNet',e,c,label)]['mean']
            class_differences.append({'exposure':e,'condition':c,'class':label,
                'inception_recall_mean':a,'resnet_recall_mean':b,'recall_difference_pp':100*(a-b)})
    for (name,exposure,condition,seed),result in scores.items():
        if exposure!='noise_matched':continue
        baseline=scores[(name,'clean_trained',condition,seed)]
        exposures.append({'model':name,'seed':seed if seed is not None else '','condition':condition,
            'direction':'noise_matched_minus_clean_trained',**deltas(result,baseline)})
    cost_summary=[]
    for model in policy['neural']:
        units=[r for r in costs if r['model']==model]
        cost_summary.append({'model':model,'jobs':len(units),'fit_procedures':2*len(units),
            'parameters':manifests[model]['model']['parameters'],
            'selection_seconds_total':sum(r['selection_seconds'] for r in units),'refit_seconds_total':sum(r['refit_seconds'] for r in units),
            'selected_epochs_median':float(np.median([r['selected_epochs'] for r in units])),
            'selection_epochs_median':float(np.median([r['selection_epochs'] for r in units]))})
    files={'metrics_per_run.csv':records,'metrics_summary.csv':summary,'classwise_metrics.csv':classwise,
        'weighted_confusions.csv':confusions,'class_recall_summary.csv':class_summary,
        'class_recall_difference.csv':class_differences,
        'neural_vs_ML_per_seed.csv':vs_ml,'neural_mean_vs_ML.csv':vs_ml_mean,
        'InceptionTime_minus_ResNet_all_seed_pairs.csv':all_seed_pairs,
        'InceptionTime_minus_ResNet_mean.csv':mean_differences,'comparison_summary.csv':comparison,
        'paired_decision_outcomes.csv':overlaps,'error_partition_group_means.csv':group_errors,
        'exposure_effects.csv':exposures,'training_job_costs.csv':costs,'training_cost_summary.csv':cost_summary,
        'prediction_times_by_fold.csv':timings}
    require(len(records)==150 and len(summary)==130,'Unexpected model/seed/case summary size')
    require(len(vs_ml)==720 and len(vs_ml_mean)==240 and len(all_seed_pairs)==45 and len(overlaps)==765,
            'A declared model/initialization comparison is missing or duplicated')
    require(len(costs)==90 and len(timings)==150 and len(class_differences)==60,'Incomplete cost/class diagnostic grid')
    for filename,table in files.items():
        # ML records have additional provenance columns; preserve the union.
        keys=list(dict.fromkeys(k for r in table for k in r))
        write_csv(destination/filename,[{k:r.get(k,'') for k in keys} for r in table])
    write_json(destination/'inputs_manifest.json',{'policy_sha256':sha256(policy_path),'inputs':inputs,
        'neural_audits':policy['neural'],'ML_index_sha256':policy['ml_index_sha256'],
        'source_files':{f.relative_to(ROOT).as_posix():sha256(f) for f in [Path(__file__),ROOT/'src/c1inception/metrics.py',
            ROOT/'src/c1inception/data.py',ROOT/'src/c1inception/io.py',ROOT/'tools/analyze_results.py',
            ROOT/'tools/plot_comparison.py',ROOT/'Analyze-Results.ps1']}})
    report=['# Analyse descriptive — InceptionTime, ResNet et 24 pipelines ML\n',
        'Statut : calcul terminé, interprétation et validation auteur à faire. Aucun entraînement ni test statistique nouveau.\n',
        'Chaque score est calculé sur les prédictions hors pli réunies des 1839 fenêtres, avec poids famille/classe. '
        'Pour les réseaux, moyenne et écart-type échantillonnal portent sur les trois graines ; pas sur les plis, pas sur un ensemble, et pas sur un intervalle de confiance. '
        'La médiane ML décrit 24 pipelines différents, pas une incertitude.\n',
        '## Macro-F1\n\n| Condition / exposition | Réseau | Moyenne ± écart-type (%) | Médiane ML (%) | Étendue ML (%) | Pipelines ML en dessous / au-dessus |\n|---|---|---:|---:|---:|---:|']
    for r in comparison:report.append(f"| {r['condition']} / {r['exposure']} | {r['model']} | {100*r['macro_f1_mean']:.2f} ± {100*r['macro_f1_sd']:.2f} | {100*r['ML_macro_f1_median']:.2f} | {100*r['ML_macro_f1_min']:.2f}–{100*r['ML_macro_f1_max']:.2f} | {r['ML_pipelines_below_neural_mean']} / {r['ML_pipelines_above_neural_mean']} |")
    report+=['\n## Lecture des autres sorties\n',
        'metrics_summary.csv : vérifier aussi BA, MCC, fausses alarmes saines, défauts manqués, Brier et log loss. Les deux dernières mesurent la qualité probabiliste sans prouver la calibration.\n',
        'InceptionTime_minus_ResNet_mean.csv : différences des moyennes de graines, InceptionTime moins ResNet. Tous les neuf croisements de graines figurent dans le fichier all_seed_pairs ; les numéros identiques ne constituent pas une justification de couplage statistique.\n',
        'Les différences F1/BA/taux sont en points de pourcentage ; MCC, Brier et log loss restent dans leurs unités. Un delta positif de fausses alarmes ou de pertes indique une dégradation.\n',
        'error_partition_group_means.csv : masses qui partagent 100(1−BA), moyennes sur 3 graines ou 24 pipelines ; elles diffèrent des taux conditionnels de fausse alarme et ne décomposent pas le Macro-F1.\n',
        'paired_decision_outcomes.csv : les quatre masses totalisent 1. La différence « A seul correct − B seul correct » correspond à une différence de BA, pas de Macro-F1.\n',
        'exposure_effects.csv : comparaisons appariées sur les mêmes fenêtres, descriptives uniquement. Ne pas ajouter ces écarts aux 728 tests historiques.\n',
        'Les temps sont ceux des phases enregistrées ; ils excluent installation, qualification GPU et export. Les temps de prédiction incluent préparation et transfert, sans warm-up exclu ; ce ne sont pas des délais de détection en ligne.\n',
        'Limites : un réseau simulé, 39 familles, trois conditions, architectures adaptées et budget commun sans recherche propre à chaque modèle. Aucune supériorité générale, équivalence ABC/ABCG ou validation externe ne découle automatiquement de ces tableaux. Les figures doivent encore être inspectées sur leur PDF réel avant intégration au manuscrit.\n']
    (destination/'analyse_fr.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    status={'status':'COMPLETE_DESCRIPTIVE_ANALYSIS_FOR_AUTHOR_REVIEW','created_utc':datetime.now(timezone.utc).isoformat(),
        'rows':len(rows),'families':39,'neural_models':2,'initializations_each':3,'evaluation_cases':5,'ML_pipelines':24,
        'neural_pooled_evaluations':30,'ML_pooled_evaluations':120,'ResNet_existing_reference_reproduced':True,
        'OOF_pooling_not_mean_fold_scores':True,'new_training':False,'new_hypothesis_tests':False,
        'best_seed_selection_or_seed_ensemble':False,'author_scientific_validation':False,'figures_actual_PDF_QA':'PENDING',
        'versions':versions,'python':platform.python_version(),'table_rows':{name:len(table) for name,table in files.items()}}
    write_json(destination/'analysis_receipt.json',status)
    print(json.dumps({'status':status['status'],'destination':str(destination),'neural_evaluations':30,'ML_evaluations':120,'new_training':False,'new_hypothesis_tests':False}))
    return status
