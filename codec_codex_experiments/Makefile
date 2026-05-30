.PHONY: test demo confirmatory clean

test:
	pytest -q

demo:
	python -m src.run_all --config configs/demo.json --out results/demo

confirmatory:
	python -m src.run_all --config configs/confirmatory_local.json --out results/confirmatory_local

clean:
	rm -rf results/demo results/confirmatory_local .pytest_cache src/__pycache__ tests/__pycache__
