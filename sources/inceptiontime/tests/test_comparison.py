"""Scientific-estimand and selection guards on synthetic predictions only."""
from copy import deepcopy
import numpy as np
import pytest
from c1inception.data import LABELS
from c1inception.metrics import score
from c1inception.comparison import aligned, pool, describe, deltas, paired_outcomes, error_parts, select_ml, SCALERS


def perfect(rows):return np.eye(12)[[LABELS.index(r['fault']) for r in rows]]


def test_pool_recovers_all_rows_without_averaging_fold_scores(synthetic_rows):
    p=perfect(synthetic_rows)
    chunks=[]
    for fold in reversed(range(1,6)):
        ids=np.array([r['Id'] for r in synthetic_rows if r['fold']==fold])[::-1]
        chunks.append((ids,p[ids]))
    np.testing.assert_array_equal(pool(chunks,synthetic_rows),p)
    with pytest.raises(ValueError,match='Incomplete/duplicated'):
        pool(chunks[:-1],synthetic_rows)
    with pytest.raises(ValueError,match='Incomplete/duplicated'):
        pool(chunks[:-1]+[chunks[0]],synthetic_rows)


def test_alignment_uses_window_identity_not_saved_order(synthetic_rows):
    p=perfect(synthetic_rows);ids=np.arange(len(synthetic_rows))[::-1]
    got_ids,got_p=aligned(ids,p[ids],p[ids].argmax(1),p[ids].argmax(1),synthetic_rows,range(len(synthetic_rows)))
    np.testing.assert_array_equal(got_ids,np.arange(len(ids)));np.testing.assert_array_equal(got_p,p)


@pytest.mark.parametrize('problem',['truth','decision','duplicate','probability'])
def test_prediction_contamination_is_rejected(synthetic_rows,problem):
    p=perfect(synthetic_rows);ids=np.arange(len(synthetic_rows));truth=p.argmax(1);predicted=truth.copy()
    if problem=='truth':truth[0]=1
    if problem=='decision':predicted[0]=1
    if problem=='duplicate':ids[-1]=ids[0]
    if problem=='probability':p[0,0]=.5
    with pytest.raises(ValueError):aligned(ids,p,truth,predicted,synthetic_rows,range(len(synthetic_rows)))


def test_seed_summary_uses_sample_sd_not_sem_or_interval():
    result=describe([.2,.4,.6])
    assert result['mean']==pytest.approx(.4)
    assert result['sd']==pytest.approx(.2)
    assert describe([.4])['sd'] is None


def test_difference_units_keep_losses_and_mcc_separate():
    a={k:.8 for k in ('macro_f1','balanced_accuracy','mcc','healthy_false_alarm_rate','missed_fault_rate','brier','log_loss','accuracy')}
    b={k:.7 for k in a};d=deltas(a,b)
    assert d['macro_f1_difference_pp']==pytest.approx(10)
    assert d['mcc_difference']==pytest.approx(.1)
    assert d['brier_difference']==pytest.approx(.1)
    assert 'brier_difference_pp' not in d


def test_error_overlap_partitions_weighted_support_and_matches_ba(synthetic_rows):
    a=perfect(synthetic_rows);b=a.copy()
    a[0]=np.eye(12)[2];b[1]=np.eye(12)[3];b[2]=np.eye(12)[4]
    result=paired_outcomes(synthetic_rows,a,b)
    assert sum(result[k] for k in ('both_correct','a_only_correct','b_only_correct','both_wrong'))==pytest.approx(1)
    sa=score(synthetic_rows,np.arange(len(a)),a);sb=score(synthetic_rows,np.arange(len(b)),b)
    assert result['ba_difference_pp_from_error_overlap']==pytest.approx(100*(sa['balanced_accuracy']-sb['balanced_accuracy']))


def test_conditional_false_alarm_is_not_total_error_mass(synthetic_rows):
    p=perfect(synthetic_rows);p[0]=np.eye(12)[1]
    result=score(synthetic_rows,np.arange(len(p)),p);parts=error_parts(result)
    assert parts['healthy_to_fault_mass_pp']==pytest.approx(100*result['healthy_false_alarm_rate']/12)
    assert sum(parts[k] for k in ('healthy_to_fault_mass_pp','fault_to_healthy_mass_pp','abc_pair_mass_pp','other_fault_error_mass_pp'))==pytest.approx(100*(1-result['balanced_accuracy']))


def ml_index():
    out={}
    for model,scaler in SCALERS.items():
        for rep in ('RMS6','RAW12','MULTI18'):
            for protocol in ('deployment','matched'):
                key=model+rep+protocol
                out[key]={'case':{'study':'robustness','preprocessor':'none','condition':'clean','protocol':protocol,
                    'anchor_only':False,'density_balanced':False,'classifier':model,'representation':rep,'normalizer':scaler}}
    return out


def test_clean_is_shared_and_does_not_double_the_24_pipelines():
    index=ml_index();selected=select_ml(index,'shared_clean','clean')
    assert len(selected)==24 and all(e['case']['protocol']=='deployment' for _,e in selected)


def test_duplicate_ml_pipeline_is_rejected_instead_of_selected():
    index=ml_index();index['duplicate']=deepcopy(next(v for v in index.values() if v['case']['protocol']=='deployment'))
    with pytest.raises(ValueError,match='exactly 24'):select_ml(index,'shared_clean','clean')
