PY = .venv/bin/python

setup:        ## create venv and install dependencies
	python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt

ingest:       ## download O*NET, learn skill embeddings, load the graph
	$(PY) -m career_kg ingest

app:          ## Streamlit UI on http://localhost:8501
	.venv/bin/streamlit run app.py

eval:         ## held-out evaluation vs. exact-match baseline
	$(PY) -m career_kg eval

test:
	$(PY) -m pytest -q

.PHONY: setup ingest app eval test
