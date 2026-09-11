"""
Hedge Fund Manager -- a four-desk research -> portfolio -> execution ->
oversight pipeline, built the same way every other tab in this app is
built: closed-form math where possible, honest documented approximations
where not, and glue code between desks that already exist in this
codebase rather than a bolted-on fifth engine.

Desk mapping (see docs/HEDGE_FUND_MANAGER.md for the full writeup this
module was commissioned from):

  Research   -> app.hedge_fund.research        (the "Kronos" seat)
  Portfolio  -> app.hedge_fund.black_litterman (the "skfolio" seat)
  Execution  -> app.hedge_fund.rebalancer      (the "NautilusTrader" seat)
  Oversight  -> app.hedge_fund.oversight       (the "Vibe-Trading" seat)

app.hedge_fund.pipeline.run_hedge_fund_manager() wires all four into one
walk-forward rebalance backtest, which is what the /hedge-fund web tab
calls.
"""
