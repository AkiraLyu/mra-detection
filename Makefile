run:
	python mra.py --epochs 20 --ewaf-alpha 0.15 --threshold-quantile 0.96

run_vae:
	python mra_vae.py --epochs 10 --detector-epochs 4 --score-shift-weight 0.05 --threshold-quantile 0.7

clean:
	rm -rf __pycache__
	rm -rf ablation/__pycache__
	rm -rf baseline/__pycache__
	rm -rf sensitivity/__pycache__
	rm -rf test/__pycache__
	rm -rf utils/__pycache__
	rm -f .codex
