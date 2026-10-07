with fails as (
    select machine_id, count(*) as n_failures, max("timestamp") as last_failure_at
    from {{ ref('stg_failures') }} group by machine_id
),
maint as (
    select machine_id, max("timestamp") as last_maintenance_at
    from {{ ref('stg_maintenance') }} group by machine_id
),
errs as (
    select machine_id, count(*) as n_errors
    from {{ ref('stg_errors') }} group by machine_id
)
select
    m.machine_id,
    m."model",
    m.age,
    coalesce(f.n_failures, 0) as n_failures,
    f.last_failure_at,
    mt.last_maintenance_at,
    coalesce(e.n_errors, 0) as n_errors
from {{ ref('stg_machines') }} m
left join fails f  on f.machine_id = m.machine_id
left join maint mt on mt.machine_id = m.machine_id
left join errs e   on e.machine_id = m.machine_id
