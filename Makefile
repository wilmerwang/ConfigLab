# Makefile for results_helper.py
PY=uv run python
SCRIPT=scripts/results_helper.py

# -----------------------------
# 显示摘要 / 交互式
# 用法: make train/test/predict
# -----------------------------
train:
	$(PY) $(SCRIPT) --task_dir logs/train --task train

test:
	$(PY) $(SCRIPT) --task_dir logs/test --task test

predict:
	$(PY) $(SCRIPT) --task_dir logs/predict --task predict

# -----------------------------
# 清理失败实验
# -----------------------------
clean:
	$(PY) $(SCRIPT) --task_dir logs/train --clean_failed
	$(PY) $(SCRIPT) --task_dir logs/test --clean_failed
	$(PY) $(SCRIPT) --task_dir logs/predict --clean_failed

# -----------------------------
# 伪目标
# -----------------------------
.PHONY: train test predict clean
