# Online, Dynamic, and Continuous Reoptimization Research Series

This file maps repositories that respond to changing information, state, demand, or constraints after deployment. It is an index only: each project remains independent because the event model, state representation, decision horizon, and adaptation mechanism differ.

## Event-driven reoptimization

- `event-driven-continuous-reoptimization` — domain-neutral event/trigger/solve/validate architecture for replacing a current feasible solution only when conditions justify reoptimization.
- `dynamic-cvrptw-online-reoptimization-python` — online/dynamic routing with time windows.
- `adaptive-production-scheduling-python` — production scheduling that adapts to changing operating conditions.
- `fjsp-ml-rescheduling` — machine-learning-assisted flexible-job-shop rescheduling.

## Learning-augmented online optimization

- `learning-augmented-online-machine-scheduling-python` — online scheduling with learned predictions augmenting an algorithmic decision rule.
- `contextual-bandits-dynamic-procurement` — sequential procurement decisions with contextual exploration/exploitation.
- `dynamic-pricing-revenue-management-rl` — dynamic pricing as a sequential RL problem.

## Dynamic control and digital twins

- `dynamic-manufacturing-digital-twin-rl` — closed-loop adaptive decisions in a manufacturing digital-twin setting.
- `production-control-with-mpc-vs-rl` — receding-horizon model-based control compared with learned control.
- `safe-rl-constrained-production-control` — adaptive control under explicit safety/constraint considerations.
- `offline-rl-industrial-process-control` — dynamic policy learning from logged data rather than online exploration.

## Sequential-decision bridge

- `sequential-decision-analytics` — foundational state/action/information framework.
- `sequential-decision-analytics-policy-classes` — PFA/CFA/VFA/DLA policy classes.
- `approximate-dynamic-programming-fleet-inventory` — approximate dynamic programming for multi-period operational decisions.

## Why these repositories stay separate

Dynamic decision systems differ according to when information arrives and how decisions adapt:

- event-triggered reoptimization;
- periodic rolling-horizon optimization;
- prediction-augmented online algorithms;
- contextual bandits;
- reinforcement learning;
- offline RL;
- model-predictive control.

These are related but not interchangeable paradigms, so the series is intended for navigation and methodological comparison rather than consolidation.
