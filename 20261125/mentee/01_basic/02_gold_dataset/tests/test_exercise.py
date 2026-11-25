"""Day28 실습 02: TODO 전에는 실패하고 완성 후 통과해야 하는 실행 테스트."""
from pathlib import Path
import subprocess, sys, json, importlib.util
import pytest

HERE=Path(__file__).resolve().parents[1]


def test_offline_student_solution():
    
    proc=subprocess.run([sys.executable,str(HERE/'main.py'),'--offline'],cwd=HERE,
                        text=True,capture_output=True,timeout=45)
    assert proc.returncode==0, (proc.stdout+'\n'+proc.stderr)[-2500:]
    path=HERE/'outputs'/'result.json'
    assert path.exists()
    data=json.loads(path.read_text(encoding='utf-8'))
    assert data['exercise']==2
    assert data['level']=='01_basic'
    assert data['mode']=='offline_fixture'
    assert data['dataset_version']=='d28-eval-v1'
    assert 'annotations' in data


def test_project_scaffold_has_expected_files():
    assert (HERE/'main.py').exists()
    assert (HERE/'README.md').exists()
    assert (HERE/'expected'/'EXPECTED.md').exists()
