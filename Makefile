VENV_PYTHON = .venv/bin/python
VENV_CHAINLIT = .venv/bin/chainlit

install:
	$(VENV_PYTHON) -m pip install -r requirements.txt

download-model:
	huggingface-cli download mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --local-dir models/Llama-3.1-8B-Instruct-4bit

run:
	$(VENV_CHAINLIT) run app.py -w

test:
	python3 tests/test_logic.py

clean:
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
