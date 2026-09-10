"""Pure size reuse must never reuse split/media/ref/CAS decisions."""
import base64
from collections import Counter
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from ollmo_services import response_frames as frames


FRAME={'kind':'ollmo.response_frame','response_id':'size-response',
       'frame_id':'size-response:1','frame_sequence':1}


def payload():
    return {'nodes':[{'id':str(i),'decision_contract':{
        'evidence':['exact branch evidence '*160]*12,'created_at':'first'}} for i in range(12)]}


def counts():
    result=Counter()
    def note(**values):result.update({k:v for k,v in values.items() if isinstance(v,int)})
    return result,note


def test_exact_size_reuse_is_immutable_and_observes_mutated_input(tmp_path):
    frame=copy.deepcopy(FRAME);reuse=frames._SnapshotSizePreparation(frame,tmp_path)
    value={'empty':{'empty':''},'items':[1,False,'kept'],'created_at':'first'}
    original=copy.deepcopy(value);seen,note=counts()
    with patch.object(frames,'state_flow_note',note):
        first=reuse.size(value,frame=frame,frames_dir=tmp_path)
        assert reuse.size(copy.deepcopy(value),frame=frame,frames_dir=tmp_path)==first
        value['items'].append('changed')
        assert reuse.size(value,frame=frame,frames_dir=tmp_path)==frames._json_size_bytes(value)
        assert reuse.size(original,frame=frame,frames_dir=tmp_path)==first
    assert seen['prepared_size_hits']==2 and seen['prepared_size_misses']==2
    assert all(type(k) is bytes and type(v) is int for k,v in reuse._entries.items())


@pytest.mark.parametrize('change',['frame_object','response_id','frame_id','frame_sequence','frame_relation','ledger','index','same_byte_epoch','store'])
def test_changed_binding_uses_full_preparation(tmp_path,change):
    frame=copy.deepcopy(FRAME)
    (tmp_path/'responses.jsonl').write_bytes(b'ledger\n');(tmp_path/'current_index.json').write_bytes(b'index\n')
    reuse=frames._SnapshotSizePreparation(frame,tmp_path);value=payload()
    reuse.size(value,frame=frame,frames_dir=tmp_path);root=tmp_path
    if change=='frame_object':frame=copy.deepcopy(frame)
    elif change in ['response_id','frame_id','frame_sequence','frame_relation']:frame[change]=2 if change=='frame_sequence' else 'changed'
    elif change in ['ledger','index']:(tmp_path/('responses.jsonl' if change=='ledger' else 'current_index.json')).write_bytes(b'changed')
    elif change=='same_byte_epoch':
        path=tmp_path/'responses.jsonl';replacement=tmp_path/'replacement';replacement.write_bytes(path.read_bytes());replacement.replace(path)
    else:root=tmp_path/'other';root.mkdir()
    seen,note=counts()
    with patch.object(frames,'state_flow_note',note):assert reuse.size(value,frame=frame,frames_dir=root)==frames._json_size_bytes(value)
    assert seen['prepared_size_binding_fallbacks']==1 and not seen['prepared_size_hits']


def test_policy_unknown_binding_and_budget_fall_back(tmp_path):
    frame=copy.deepcopy(FRAME);value=payload();reuse=frames._SnapshotSizePreparation(frame,tmp_path)
    reuse.size(value,frame=frame,frames_dir=tmp_path);seen,note=counts();original=frames._json_safe
    with patch.object(frames,'state_flow_note',note),patch.object(frames,'_json_safe',lambda x:original(x)):
        assert reuse.size(value,frame=frame,frames_dir=tmp_path)==frames._json_size_bytes(value)
    assert seen['prepared_size_policy_fallbacks']==1
    with patch.object(frames,'_SNAPSHOT_SIZE_PREPARATION_MAX_BYTES',0):
        fresh=frames._SnapshotSizePreparation(frame,tmp_path);seen,note=counts()
        with patch.object(frames,'state_flow_note',note):
            for _ in range(2):assert fresh.size(value,frame=frame,frames_dir=tmp_path)==frames._json_size_bytes(value)
        assert seen['prepared_size_budget_fallbacks']==2 and fresh._retained_bytes==0 and not fresh._entries
    unknown=frames._SnapshotSizePreparation(frame,tmp_path/'missing')
    assert unknown.size(value,frame=frame,frames_dir=tmp_path/'missing')==frames._json_size_bytes(value)
    assert not unknown._entries


def test_typed_exact_input_cannot_alias_non_json_keys_or_custom_mappings(tmp_path):
    frame=copy.deepcopy(FRAME);reuse=frames._SnapshotSizePreparation(frame,tmp_path)
    class Custom(dict):
        def items(self):return [('changed','custom mapping meaning')]
    for value in [{'0':'kept'},{0:'dropped'},Custom(original='ignored'),{'path':Path('not-json')}, {'nested':(1,2)}]:
        assert reuse.size(value,frame=frame,frames_dir=tmp_path)==frames._json_size_bytes(value)
    assert len(reuse._entries)==1


def test_no_worker_or_compaction_scope_leak(tmp_path):
    frame=copy.deepcopy(FRAME);value=payload();reuse=frames._SnapshotSizePreparation(frame,tmp_path)
    reuse.size(value,frame=frame,frames_dir=tmp_path);seen,note=counts()
    with patch.object(frames,'state_flow_note',note),ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(reuse.size,value,frame=frame,frames_dir=tmp_path).result()==frames._json_size_bytes(value)
    assert seen['prepared_size_binding_fallbacks']==1
    fresh=frames._SnapshotSizePreparation(frame,tmp_path)
    assert not fresh._entries


def write(root,frame,value,*,path='runtime.request_phase_graph',depth=0,budget_used=0,enabled=True):
    root.mkdir(exist_ok=True)
    reuse=frames._SnapshotSizePreparation(frame,root) if enabled else None
    return frames._write_snapshot_ref(value,frame=frame,frames_dir=root,json_path=path,
                                     _sidecar_depth=depth,_sidecar_split_counter=[budget_used],_size_preparation=reuse)


@pytest.mark.parametrize('path,depth,budget_used',[
    ('runtime.request_phase_graph',0,0),('request.input',0,0),
    ('runtime.semantic_review_lens_contract',0,0),('runtime.request_phase_graph',3,0),
    ('runtime.request_phase_graph',0,frames._SIDECAR_SPLIT_MAX_REFS-1),
    ('runtime.request_phase_graph',0,frames._SIDECAR_SPLIT_MAX_REFS),
])
def test_path_depth_budget_and_timestamp_policy_keep_fresh_split_results(tmp_path,path,depth,budget_used):
    value=payload();frame=copy.deepcopy(FRAME)
    first=write(tmp_path/'baseline',frame,value,path=path,depth=depth,budget_used=budget_used,enabled=False)
    second=write(tmp_path/'candidate',frame,value,path=path,depth=depth,budget_used=budget_used)
    assert first==second
    files=lambda root:{str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*.json')}
    assert files(tmp_path/'baseline')==files(tmp_path/'candidate')


def test_size_hits_do_not_cache_predicates_or_sibling_reservation(tmp_path):
    frame=copy.deepcopy(FRAME);reuse=frames._SnapshotSizePreparation(frame,tmp_path)
    contract={'evidence':'x'*(frames._SIDECAR_SPLIT_LIMIT_BYTES+1)}
    value={'decision_contract':contract,'semantic_review_lens_contract':contract}
    def split(used):return frames._split_sidecar_payload_for_cas(value,frame=frame,frames_dir=tmp_path,
        json_path='runtime.request_phase_graph',depth=0,split_counter=[used],size_preparation=reuse)
    full=split(0);bounded=split(frames._SIDECAR_SPLIT_MAX_REFS-1)
    with patch.object(frames,'_should_split_sidecar_child',wraps=frames._should_split_sidecar_child) as decisions:
        repeated=split(0)
    assert full==repeated and decisions.call_count>0
    assert len(full[1])>len(bounded[1])


@pytest.mark.parametrize('change',['missing','corrupt'])
def test_size_hits_preserve_fresh_cas_repair(tmp_path,change):
    value=payload();frame=copy.deepcopy(FRAME);reuse=frames._SnapshotSizePreparation(frame,tmp_path)
    def save():return frames._write_snapshot_ref(value,frame=frame,frames_dir=tmp_path,
        json_path='runtime.request_phase_graph',_size_preparation=reuse)
    first=save();target=tmp_path/first['path']
    if change=='missing':target.unlink()
    else:target.write_bytes(b'corrupt')
    seen,note=counts()
    with patch.object(frames,'state_flow_note',note):second=save()
    assert first==second and seen['prepared_size_hits']>0
    assert seen['sidecar_verification_reads']>0 and seen['cas_actual_writes']>=1
    assert frames._read_snapshot_ref_payload(second,frames_dir=tmp_path) is not None


def test_postwrite_verification_failure_is_unchanged(tmp_path):
    frame=copy.deepcopy(FRAME);value=payload();reuse=frames._SnapshotSizePreparation(frame,tmp_path)
    def broken(target,payload):target.write_bytes(b'corrupt')
    with patch.object(frames,'_atomic_replace_file_bytes',broken):
        with pytest.raises(OSError,match='post-write verification'):
            frames._write_snapshot_ref(value,frame=frame,frames_dir=tmp_path,json_path='runtime.graph',_size_preparation=reuse)


def test_media_provenance_and_changed_saved_bytes_are_rechecked(tmp_path):
    raw=b'opaque media bytes'*200;media=tmp_path/'one.png';media.write_bytes(raw)
    other=tmp_path/'two.png';other.write_bytes(raw);root=tmp_path/'frames';root.mkdir()
    frame=copy.deepcopy(FRAME);reuse=frames._SnapshotSizePreparation(frame,root)
    value=payload();value.update(saved_image_path=str(media),artifact_ref='artifact:one',image=base64.b64encode(raw).decode())
    def save():return frames._write_snapshot_ref(value,frame=frame,frames_dir=root,json_path='runtime.graph',_size_preparation=reuse)
    first=save();value.update(saved_image_path=str(other),artifact_ref='artifact:two');second=save()
    other.write_bytes(b'changed artifact bytes');third=save()
    a=frames._read_snapshot_ref_payload(first,frames_dir=root);b=frames._read_snapshot_ref_payload(second,frames_dir=root);c=frames._read_snapshot_ref_payload(third,frames_dir=root)
    assert a['image']['artifact_ref']=='artifact:one' and b['image']['artifact_ref']=='artifact:two'
    assert c['image']['kind']=='ollmo.snapshot_stripped_raw_media_payload'
    assert len({first['sha256'],second['sha256'],third['sha256']})==3


def test_concurrent_compactions_are_exact_and_private(tmp_path):
    value=payload()
    def compact(index):
        root=tmp_path/str(index);root.mkdir();frame=copy.deepcopy(FRAME)
        frame['runtime']={'request_phase_graph':copy.deepcopy(value)}
        result=frames.compact_response_frame_for_ledger(frame,frames_dir=root)
        return result,{str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*.json')}
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(compact,[0,1]))
    assert results[0]==results[1]
