import pandas as pd

from app.backtest.risk import RiskConfig
from app.strategy.base import apply_strategy_trade_cap, resolve_max_trades_per_day
from app.strategy.manual import ManualStrategy


def test_manual_cap_tightens_only():
    s = ManualStrategy({"max_trades_per_day": 1})
    assert resolve_max_trades_per_day(s) == 1
    assert apply_strategy_trade_cap(RiskConfig(max_trades_per_day=10), s).max_trades_per_day == 1
    assert apply_strategy_trade_cap(RiskConfig(max_trades_per_day=1), s).max_trades_per_day == 1
    s3 = ManualStrategy({"max_trades_per_day": 5})
    assert apply_strategy_trade_cap(RiskConfig(max_trades_per_day=2), s3).max_trades_per_day == 2


def test_no_declaration_is_unchanged():
    r = RiskConfig(max_trades_per_day=7)
    assert apply_strategy_trade_cap(r, ManualStrategy({})) is r


def test_python_strategy_declares_cap(tmp_path):
    from app.strategy.python import PythonStrategy
    f = tmp_path / "s.py"
    f.write_text("MAX_TRADES_PER_DAY = 1\ndef generate_signals(df):\n    import pandas as pd\n    return pd.Series(0, index=df.index)\n")
    assert resolve_max_trades_per_day(PythonStrategy(f)) == 1
