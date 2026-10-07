# Data dictionary

All data is synthetic unless the Azure PdM adapter is used.

## Raw tables

| Table | Columns |
|---|---|
| telemetry | machine_id, timestamp (hourly), voltage, rotation, pressure, vibration |
| errors | machine_id, timestamp, error_code |
| maintenance | machine_id, timestamp, component |
| failures | machine_id, timestamp, component |
| machines | machine_id, model, age |

## Target

`failure_next_24h` = 1 if the machine has a failure in the 24 hours after the observation time, else 0 (about 2.8 % positives in the synthetic data).

## Features (25)

| Group | Features |
|---|---|
| Sensor means | `{voltage,rotation,pressure,vibration}_mean_3h`, `..._mean_24h` |
| Variability | `..._std_24h` |
| Trend | `..._delta_3h` |
| Peaks | `vibration_max_6h`, `pressure_max_6h` |
| Errors | `errors_last_24h`, `errors_last_72h`, `hours_since_last_error` |
| History | `hours_since_last_maintenance`, `hours_since_last_failure` (capped at 720 h) |
| Machine | `age`, `model_code` |

Features use only past data at the observation time.

## Outputs

`ml.machine_risk_predictions`: machine_id, run_date, probability (calibrated), risk_rank, flagged (above the cost-optimal threshold), model_version.
