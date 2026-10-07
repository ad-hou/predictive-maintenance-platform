{#-
  Rolling sensor features in SQL, for analytics and BI. They use the same names and definitions
  as src/features/build_features.py (windows are expressed in rows = hours). The features used by
  the model are computed by the Python code, which is the single source of truth for training and
  serving; tests/test_sql_python_parity.py checks that both agree on this subset.
-#}
select
    machine_id,
    "timestamp",
    voltage, rotation, pressure, vibration,
    {% for s in ['voltage', 'rotation', 'pressure', 'vibration'] %}
    avg({{ s }}) over (partition by machine_id order by "timestamp" rows between 2 preceding and current row)  as {{ s }}_mean_3h,
    avg({{ s }}) over (partition by machine_id order by "timestamp" rows between 23 preceding and current row) as {{ s }}_mean_24h,
    coalesce(stddev_samp({{ s }}) over (partition by machine_id order by "timestamp" rows between 23 preceding and current row), 0) as {{ s }}_std_24h,
    coalesce({{ s }} - lag({{ s }}, 3) over (partition by machine_id order by "timestamp"), 0)                  as {{ s }}_delta_3h,
    {% endfor %}
    max(vibration) over (partition by machine_id order by "timestamp" rows between 5 preceding and current row) as vibration_max_6h,
    max(pressure)  over (partition by machine_id order by "timestamp" rows between 5 preceding and current row) as pressure_max_6h
from {{ ref('stg_telemetry') }}
