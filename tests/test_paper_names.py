"""Exercise old-archive installation through the current paper interfaces."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import pytest
from repro.common import digest, select_tasks
from repro.names import display_method
from repro.archive_compat import materialized_bytes, paper_names


def test_paper_filters_select_the_matching_components():
    cases = {'UB+Order': 'UB_ORDER', 'UB+LB': 'UB_LB', 'LB+Order': 'LB_ORDER',
             'BOLT (λ=0)': 'BOLT_L0', 'Order + endpoints': 'ANCHORED_MID_ORDER'}
    for label, method in cases.items():
        tasks = select_tasks('all', ['AUTHOR'], ['A0'], [label], 0)
        assert len(tasks) == 1 and tasks[0]['method'] == method
        assert display_method(method) == label
    assert display_method('CLIQUE', ablation=True) == 'UB'


def test_name_migration_preserves_queries_and_numeric_observations():
    old = {'arm': 'BIMO_A0', 'query': 'BIMO%MONO', 'est': 12.375, 'true': 15}
    raw = (json.dumps(old) + '\n').encode()
    result = json.loads(materialized_bytes('artifacts/job/timings.jsonl', raw))
    assert result == dict(old, arm='BOLT_A0')
    predictions = {'BIMO_A0': {'title.title': {'BIMO%MONO': 12.375}}}
    result = json.loads(materialized_bytes('artifacts/job/positive_predictions.json', json.dumps(predictions).encode()))
    assert result == {'BOLT_A0': predictions['BIMO_A0']}
    assert materialized_bytes('data/AUTHOR/training/A0_BIMO/train.txt', b'BIMO%MONO,12\n') == b'BIMO%MONO,12\n'
    assert materialized_bytes('artifacts/checkpoints/A_BIMO/model.pth', b'BIMO MONO weights') == b'BIMO MONO weights'


def _archive_case(tmp_path):
    root = tmp_path/'checkout'; root.mkdir()
    mirror = tmp_path/'mirror'; mirror.mkdir()
    payloads = {'data/AUTHOR/training/A0_BIMO/train.txt': b'BIMO%MONO,12\n',
                'artifacts/job/timings.jsonl': b'{"arm":"BIMO_A0","query":"BIMO%MONO","total_ms":1.25}\n'}
    archive = mirror/'paper.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name, payload in payloads.items():
            item = tarfile.TarInfo(name); item.size = len(payload)
            tar.addfile(item, io.BytesIO(payload))
    hashes = {k: hashlib.sha256(v).hexdigest() for k, v in payloads.items()}
    original = dict(name=archive.name, url='unused', sha256=digest(archive), files=hashes)
    path = root/'manifests/archive.json';path.parent.mkdir();path.write_text(json.dumps({'bundles': [original]}))
    current = dict(original, archive_manifest='manifests/archive.json',
                   files={paper_names(k): hashlib.sha256(materialized_bytes(k, v)).hexdigest() for k, v in payloads.items()})
    return root, mirror, current, payloads


def test_old_archive_installs_current_names_and_restores_damage(tmp_path, monkeypatch):
    from repro import datasets
    root, mirror, current, payloads = _archive_case(tmp_path)
    monkeypatch.setattr(datasets, 'ROOT', root)
    datasets.install_bundle(current, str(mirror))
    view = root/'data/AUTHOR/training/A0_BOLT/train.txt'
    assert view.read_bytes() == payloads['data/AUTHOR/training/A0_BIMO/train.txt']
    assert not (root/'data/AUTHOR/training/A0_BIMO').exists()
    path = root/'artifacts/job/timings.jsonl'
    assert json.loads(path.read_text())['arm'] == 'BOLT_A0'
    view.write_bytes(b'damaged')
    datasets.install_bundle(current, str(mirror))
    assert view.read_bytes() == payloads['data/AUTHOR/training/A0_BIMO/train.txt']


def test_migration_rejects_incorrect_materialized_hash(tmp_path, monkeypatch):
    from repro import datasets
    root, mirror, current, _ = _archive_case(tmp_path)
    monkeypatch.setattr(datasets, 'ROOT', root)
    current['files']['artifacts/job/timings.jsonl'] = '0' * 64
    with pytest.raises(ValueError, match='Materialized checksum mismatch'):
        datasets.install_bundle(current, str(mirror))


@pytest.mark.parametrize('method,extension', [('BOLT','.pth'), ('BOLT_L0','.pth'), ('UB_ORDER','.pth'), ('UB_LB','.pth'), ('LBS','.pkl')])
def test_checkpoint_type_uses_method_not_bolt_run_prefix(tmp_path, monkeypatch, method, extension):
    from src.util import get_common_pathes_for_estimator
    monkeypatch.chdir(tmp_path)
    paths = get_common_pathes_for_estimator('TOY', 'BOLT_RUN/attempt/' + method, 'A0_BOLT', 0)
    assert Path(paths[2]).suffix == extension
