Submission for an RL competition involving gold mining agents.

A3C agent seems to converge at around -15 points in rewards. Bots are still based upon heuristics. Will incorporate smarter bots which are also actor-critic agents. The idea right now is to use parameters of winning networks per local environment to update the global network, then synchronize. The bots (who are also AC agents) will then take on the 2nd best agents' parameters per local environment and the games start again.

Might also incorporate an ML-fused prediction scheme to predict the states of other players and use that to make actions. The idea is that the main agent must learn to consider the other agents when making its own prediction to go for gold or not. 
