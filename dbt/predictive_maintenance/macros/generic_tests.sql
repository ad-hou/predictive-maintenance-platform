{# (machine_id, timestamp)-style uniqueness over several columns #}
{% test unique_combination(model, columns) %}
select {{ columns | join(', ') }}, count(*) as n
from {{ model }}
group by {{ columns | join(', ') }}
having count(*) > 1
{% endtest %}

{# Values must lie within [min_value, max_value]; nulls are handled by not_null tests. #}
{% test accepted_range(model, column_name, min_value=none, max_value=none) %}
select *
from {{ model }}
where {{ column_name }} is not null
  and (
    false
    {% if min_value is not none %} or {{ column_name }} < {{ min_value }} {% endif %}
    {% if max_value is not none %} or {{ column_name }} > {{ max_value }} {% endif %}
  )
{% endtest %}

{# Freshness: the newest value of the column must be younger than max_hours. #}
{% test recent_enough(model, column_name, max_hours) %}
select max({{ column_name }}) as newest
from {{ model }}
having max({{ column_name }}) < current_timestamp - ({{ max_hours }} * interval '1 hour')
{% endtest %}
