select
    cast(trim(machine_id) as text)  as machine_id,
    cast("timestamp" as timestamp)  as "timestamp",
    cast(component as text)         as component
from {{ source('raw', 'raw_failures') }}
