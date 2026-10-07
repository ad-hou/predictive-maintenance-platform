# Decisions and trade-offs

| Decision | Why | Limit |
|---|---|---|
| Chronological blocks, no random split | Random splits leak the future into training for time series | Fewer rows per block; metrics vary between runs |
| Isotonic calibration on its own block | Cost-based decisions need probabilities that mean something | Needs enough positives in the block; with few it can be coarse |
| Threshold chosen on a separate block, by cost | Accuracy and F1 ignore that a missed failure costs far more than an inspection | Costs are hypothetical |
| Champion chosen on calibration-block PR-AUC | Keeps the test block untouched for the final report | In one run logistic regression had a lower test cost than the selected model. Selecting on cost would be a valid alternative, left as a conscious choice |
| PSI threshold 0.2 plus KS test | Common rule of thumb for "significant shift" | A heuristic, not a law; the value is configurable (`PSI_THRESHOLD`) |
| Promotion needs a cost drop of at least 2 % on an unseen window | Avoids swapping models for noise | The 2 % is a judgment call (`MIN_COST_GAIN`) |
| PostgreSQL + DBT for transformation | SQL transformations with tests and lineage | Features for the model are still built in Python; the DBT marts serve analytics and the risk table |
| Fast numpy serving path + skew test | Pandas rolling is too slow per request | Two implementations to keep aligned; the test guards this |
| Dashboard on a snapshot | Can be hosted for free without a backend | It shows a recorded scenario, not live production |
| EC2 instead of ECS/EKS | Cheap, simple, enough for a demo | No autoscaling or multi-AZ |
| Locust results kept but caveated | Shows how the API was profiled and optimised | Run on a small shared machine |
