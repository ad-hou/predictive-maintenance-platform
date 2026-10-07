select
    cast(trim(machine_id) as text)  as machine_id,
    cast("timestamp" as timestamp)  as "timestamp",
    cast(error_code as text)        as error_code
from {{ source('raw', 'raw_errors') }}
