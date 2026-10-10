@echo off
REM SVD++ Hyperparameter Tuning on ML-1M Dataset (Windows)

echo Starting hyperparameter tuning for SVD++ on ML-1M...

python tune.py --algo_name svdpp --dataset ml-1m --gpu "" --hyperopt_trail 20 --optimization_metric ndcg --tune_pack "{\"factors\": {\"min\": 32, \"max\": 128, \"step\": 16}, \"lr\": {\"min\": 0.001, \"max\": 0.01, \"step\": null}, \"reg_1\": {\"min\": 0.0001, \"max\": 0.01, \"step\": null}, \"reg_2\": {\"min\": 0.0001, \"max\": 0.01, \"step\": null}, \"reg_y\": {\"min\": 0.0001, \"max\": 0.01, \"step\": null}}" --epochs 20 --topk 10 --cand_num 100

echo Tuning completed! Check tune_res/ folder for results.
pause
