from pathlib import Path
from typing import Any

import lightning as L  # noqa: N812
from omegaconf import DictConfig

from configlab.utils.git_utils import snapshot_git_state

from .build import build_callbacks, build_data_module, build_loggers, build_model_module, build_trainer


def run_pipeline(config: DictConfig) -> dict[str, Any]:
    """Run the pipeline."""
    if config.get("seed"):
        L.seed_everything(config.seed, workers=True)

    # capture git info
    snapshot_git_state(output_dir=Path(config.paths.output_dir) / "git_snapshot", cwd=Path(config.paths.root_dir))

    # data module
    data_module = build_data_module(config)

    # model module
    model_module = build_model_module(config)

    # logger and callbacks
    callbacks = build_callbacks(config)
    loggers = build_loggers(config)

    # trainer
    trainer = build_trainer(config, callbacks=callbacks, logger=loggers)

    # run
    results = {}

    if config.train:
        trainer.fit(model=model_module, datamodule=data_module, ckpt_path=config.get("ckpt_path"), weights_only=False)
        results["train_metrics"] = trainer.callback_metrics
        config.ckpt_path = trainer.checkpoint_callback.best_model_path
    if config.test:
        trainer.test(model=model_module, datamodule=data_module, ckpt_path=config.get("ckpt_path"), weights_only=False)
        results["test_metrics"] = trainer.callback_metrics
    if config.predict:
        output = trainer.predict(
            model=model_module, datamodule=data_module, ckpt_path=config.get("ckpt_path"), weights_only=False
        )
        results["predict_output"] = output

    return results
