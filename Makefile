install:
	@poetry init
	@poetry install
run:
	@uvicorn api.main:app --reload --host 0.0.0.0 --port 8001
