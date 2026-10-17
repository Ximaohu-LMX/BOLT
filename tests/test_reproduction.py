"""Protect integrity checks that decide whether an expensive run may be skipped."""
import io
import json
from pathlib import Path
import tarfile
from unittest.mock import patch
import numpy as np
import pytest
from repro.common import digest, identity, atomic_json
from repro.runner import completed
from repro.metrics import qerrors, ordering
from repro.datasets import safe_members


def test_skip_requires_identity_and_every_output(tmp_path):
    output = tmp_path/'prediction.csv'
    output.write_text('query,true,est\na,1,1\n')
    receipt = tmp_path/'DONE.json'
    atomic_json(receipt, dict(status='COMPLETE', identity=identity({'lambda':5}), outputs={str(output):digest(output)}))
    assert completed(receipt, identity({'lambda':5}))
    assert not completed(receipt, identity({'lambda':0}))
    output.write_text('changed')
    assert not completed(receipt, identity({'lambda':5}))
    output.unlink()
    assert not completed(receipt, identity({'lambda':5}))


def test_partial_receipt_is_not_success(tmp_path):
    p=tmp_path/'DONE.json';p.write_text('{')
    assert not completed(p,'unused')
    atomic_json(p,dict(status='RUNNING',identity='x',outputs={}))
    assert not completed(p,'x')


@pytest.mark.parametrize('name,link', [('../escape',False),('/absolute',False),('safe',True)])
def test_archive_rejects_paths_and_links(name,link):
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w') as tar:
        member=tarfile.TarInfo(name)
        if link:
            member.type=tarfile.SYMTYPE;member.linkname='../outside'
        tar.addfile(member)
    buffer.seek(0)
    with tarfile.open(fileobj=buffer,mode='r') as tar, pytest.raises(ValueError):
        list(safe_members(tar))


def test_qerror_uses_floor_and_continuous_predictions():
    np.testing.assert_allclose(qerrors([0,2,10],[0,3,2]),[1,1.5,5])
    with pytest.raises(ValueError):qerrors([1],[float('nan')])
    with pytest.raises(ValueError):qerrors([1,2],[1])


def test_integer_ties_are_distinct_from_continuous_inversions():
    result=ordering({'tight':1.4,'loose':1.2},[['tight','loose']])
    assert result['R_int']==1 and result['R_tie']==1 and result['raw_zero_slack']==0


def test_filters_do_not_schedule_job_as_a_main_dataset():
    from repro.common import select_tasks
    selected=select_tasks('accuracy',['AUTHOR'],['A0'],['BOLT'],0)
    assert [t['id'] for t in selected]==['AUTHOR_A0_BOLT_S0']
    assert all(t['suite']!='job' for t in select_tasks('all'))
