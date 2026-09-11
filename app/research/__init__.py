"""
T58 Research Director -- the diagnostic layer that sits ABOVE search/
evolution/forge and turns every strategy (winner or failure) into
information about where an edge does, or does not, live.

See app.research.director for the actual engines:
    - edge_decomposition   -- add one rule layer at a time, watch the metric move
    - ablation_test        -- remove one rule at a time from a full strategy
    - null_baselines       -- compare against dumb reference strategies
    - signal_degradation   -- stress the strategy's timing/costs/fills
    - trade_contribution   -- leave-X-out dependency analysis
    - conditional_expectancy -- when does this edge actually fire
    - regime_discovery     -- what separates winners from losers, tested OOS
    - research_director_report -- cross-run "what have we learned" synthesis
"""
