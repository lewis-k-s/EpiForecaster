rsync -avz --progress . dt:/home/bsc/bsc008913/EpiForecaster \
--exclude="outputs" \
--exclude ".git" \
--exclude ".venv" \
--exclude "__pycache__" \
--exclude ".mypy_cache" \
--exclude ".pytest_cache" \
--exclude ".ruff_cache" \
--exclude "EpiForecaster.egg-info" \
