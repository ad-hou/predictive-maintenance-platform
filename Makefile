.PHONY: install data test lint demo api dashboard dbt-test loadtest up down
PY ?= python

install:
	$(PY) -m pip install -r requirements-dev.txt

data:            ## synthetic dataset in data/raw
	$(PY) -m src.cli generate-data --machines 60 --days 120

test:            ## unit tests (add PMP_TEST_DATABASE_URL for the PostgreSQL integration tests)
	$(PY) -m pytest -q

lint:
	ruff check . && ruff format --check .

demo:            ## train -> drift -> retrain -> compare -> promote, then export dashboard/demo_data
	$(PY) -m src.cli demo

api:             ## needs a champion model: run `make demo` or `python -m src.cli train` first
	uvicorn api.main:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py

dbt-test:
	cd dbt/predictive_maintenance && dbt build

loadtest:        ## API must be running; results in reports/loadtest_*
	mkdir -p reports && locust -f load_tests/locustfile.py --headless -u 50 -r 10 -t 60s --host http://localhost:8000 --csv reports/loadtest

up:
	docker compose up --build

down:
	docker compose down
