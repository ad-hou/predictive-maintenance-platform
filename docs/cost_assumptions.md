# Cost assumptions (hypothetical)

These euro values are invented to make the trade-off concrete. Replace them with real figures from a maintenance team; the whole pipeline reads them from environment variables.

| Event | Cost | Variable |
|---|---|---|
| Missed failure (unplanned breakdown) | 10 000 € | `COST_MISSED_FAILURE` |
| Useless inspection (false alarm) | 500 € | `COST_USELESS_INSPECTION` |
| Useful inspection (failure prevented) | 500 € | `COST_USEFUL_INSPECTION` |

Total cost = missed × 10 000 + false alarms × 500 + true alarms × 500. Reference points: "do nothing" (every failure missed) and "inspect everything".

The ratio 20:1 between a miss and an inspection is the assumption that drives the threshold. Changing it changes the optimal threshold and can change which model is cheapest.

Inspection capacity is separate: `INSPECTIONS_PER_DAY` (default 5) defines the precision@N metric.
