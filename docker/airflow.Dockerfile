FROM apache/airflow:2.10.5-python3.12
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
USER airflow
# The project runs in its own virtualenv so that its dependencies never conflict with Airflow's pinned ones.
COPY --chown=airflow requirements.txt /tmp/requirements.txt
RUN python -m venv /home/airflow/pmp-venv && /home/airflow/pmp-venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt
ENV PMP_HOME=/opt/pmp PMP_PYTHON=/home/airflow/pmp-venv/bin/python PMP_DBT=/home/airflow/pmp-venv/bin/dbt
COPY --chown=airflow . /opt/pmp
COPY --chown=airflow airflow/dags /opt/airflow/dags
