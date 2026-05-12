## v0.3.0 (2026-05-12)

### Feat

- Add git snapshot when runing

### Fix

- 修复使用optuna storage sqlite 错误; 增加linux 系统cuda torch 版本
- add missing __init__.py files to ensure proper package imports

### Refactor

- move optimizer and scheduler out of model for hyperparameter tuning
- remove redundant code
- reorganize configuration for data/model module
- reorganize data_prepare function to lit_datamodule
- reorganize model configuration paths (encoder -> model/encoder, head -> model/head)

## v0.2.0 (2026-05-01)

### Feat

- release v0.1.0 - hydra + lightning modular training framework
