.PHONY: install lint format test smoke-train run-pipeline clean

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests

format:
	ruff format src tests

test:
	pytest --cov=src/genai_trainer --cov-report=term-missing

smoke-pipeline:
	python -m genai_trainer.cli run-pipeline --dataset synthetic --epochs 1 --num-samples 16

train-diffusion:
	python -m genai_trainer.cli train-diffusion --config configs/diffusion_config.yaml

train-gan:
	python -m genai_trainer.cli train-gan --config configs/gan_config.yaml

evaluate:
	python -m genai_trainer.cli evaluate --config configs/eval_config.yaml

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache outputs/checkpoints/* outputs/samples/* outputs/reports/*
