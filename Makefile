VENV := .venv/bin
export PYTHONPATH := src

.PHONY: setup data yolo-data yolo train eval meta test baselines all

setup:
	python3 -m venv .venv && $(VENV)/pip install -q -r requirements.txt

data:
	./scripts/download_data.sh

yolo-data:
	$(VENV)/python -m fho.prepare_yolo

yolo: yolo-data
	./scripts/setup_yolov5.sh && ./scripts/train_yolo.sh

train:
	$(VENV)/python -m fho.train_landmarks --epochs 400

eval:
	$(VENV)/python -m fho.evaluate --split test

meta:
	$(VENV)/python -m fho.metamorphic

baselines:
	$(VENV)/python -m fho.baselines --split test

test:
	$(VENV)/python -m pytest tests -q

all: data yolo-data baselines test
