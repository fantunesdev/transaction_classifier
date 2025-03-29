format:
	@poetry run isort .
	@poetry run blue . --line-length 120
install:
	@poetry init
	@poetry install
run:
	@uvicorn api.main:app --reload --host 0.0.0.0 --port 8001
