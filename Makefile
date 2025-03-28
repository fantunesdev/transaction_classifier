install:
	@poetry init
	@poetry add scikit-learn pandas fastapi uvicorn
run:
	@poetry shell
	@uvicorn api.main:app --reload --host 0.0.0.0 --port 8001
