.PHONY: train test lint fmt monitor serve clean

# ── Training pipeline ──
train:
	python -m src.deployment.pipeline \
		--input data/raw/cs-training.csv \
		--output data/processed/ \
		--model-dir models/

# ── Tests ──
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html

# ── Code quality ──
lint:
	ruff check src/ tests/

fmt:
	ruff format src/ tests/

# ── Monitoring dashboard ──
monitor:
	streamlit run src/monitoring/dashboard.py \
		--server.port 8501

# ── API serving ──
serve:
	uvicorn src.deployment.api:app --host 0.0.0.0 --port 8000 --reload

# ── Notebooks ──
notebook:
	jupyter lab --no-browser --port=8888

# ── Clean ──
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf htmlcov/ .coverage