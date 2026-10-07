select
    cast(trim(machine_id) as text)       as machine_id,
    cast("timestamp" as timestamp)       as "timestamp",
    cast(voltage as double precision)    as voltage,
    cast(rotation as double precision)   as rotation,
    cast(pressure as double precision)   as pressure,
    cast(vibration as double precision)  as vibration
from {{ source('raw', 'raw_telemetry') }}
