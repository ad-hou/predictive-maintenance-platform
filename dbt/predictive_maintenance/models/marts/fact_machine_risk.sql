{#- Latest published prediction per machine, joined with machine attributes. -#}
with ranked as (
    select
        p.*,
        row_number() over (partition by machine_id order by run_date desc, "timestamp" desc) as rn
    from {{ source('ml', 'machine_risk_predictions') }} p
)
select
    r.machine_id,
    d."model",
    d.age,
    d.n_failures,
    d.last_failure_at,
    d.last_maintenance_at,
    r.run_date,
    r."timestamp"          as scored_at,
    r.failure_probability,
    r.risk_level,
    r.top_factor,
    r.model_version
from ranked r
join {{ ref('dim_machine') }} d on d.machine_id = r.machine_id
where r.rn = 1
