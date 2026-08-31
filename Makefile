VENV := .venv/bin
export PYTHONPATH := src

.PHONY: setup data data-external yolo-data yolo train eval meta external no-training roundtrip bench figures test baselines all

setup:
	python3 -m venv .venv && $(VENV)/pip install -q -r requirements.txt

data:
	./scripts/download_data.sh

yolo-data:
	$(VENV)/python -m fho.prepare_yolo

# yolo-data first: focus.yaml records an absolute path, so it goes stale if the
# repository is moved or renamed
yolo: yolo-data
	./scripts/setup_yolov5.sh && ./scripts/train_yolo.sh

train:
	$(VENV)/python -m fho.train_landmarks --epochs 400

eval:
	$(VENV)/python -m fho.evaluate --split test

meta:
	$(VENV)/python -m fho.metamorphic --json runs/metamorphic_focus.json

data-external:
	./scripts/download_external.sh

external:
	$(VENV)/python -m fho.external --n 500 --by-machine --json runs/external.json

no-training:
	$(VENV)/python -m fho.no_training --split test --json runs/no_training.json

roundtrip:
	$(VENV)/python -m fho.roundtrip --json runs/roundtrip.json

bench:
	$(VENV)/python -m fho.bench --detector runs/yolo/focus/weights/best.pt --json runs/bench.json

figures:
	$(VENV)/python -m fho.figures

baselines:
	$(VENV)/python -m fho.baselines --split test

test:
	$(VENV)/python -m pytest tests -q

all: data yolo-data baselines test
