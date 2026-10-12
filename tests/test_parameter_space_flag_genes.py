"""GA genome must not mutate boolean/flag comparison values or dead period genes."""
import json
from pathlib import Path

from app.optimize.parameter_space import extract_genome

REPO = Path(__file__).resolve().parents[1]
NQ = REPO / "tests/fixtures/nq_orb_manual.json"


def test_nq_orb_genome_has_only_live_genes():
    cfg = json.loads(NQ.read_text())
    paths = [tuple(g.path) for g in extract_genome(cfg)]
    assert len(paths) == 4
    for p in paths:
        assert p[-1] != "value" or p[0] != "entry_conditions"
    names = {p[-1] for p in paths}
    assert "value" not in names and "lookback" not in names
