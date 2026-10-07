select
    cast(trim(machine_id) as text)  as machine_id,
    cast("model" as text)           as "model",
    cast(age as integer)            as age
from {{ source('raw', 'raw_machines') }}
