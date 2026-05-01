from typing import Any

import lightning as L  # noqa: N812
from omegaconf import DictConfig

from configlab.data.data_prepare import mnist_prepare

from .build import build_callbacks, build_data_module, build_loggers, build_model_module, build_trainer


def run_pipeline(config: DictConfig) -> dict[str, Any]:
    """Run the pipeline."""
    if config.get("seed"):
        L.seed_everything(config.seed, workers=True)

    # prepare data
    train, test = mnist_prepare(config.paths.data_dir)

    # data module
    data_module = build_data_module(config, train_dataset=train, test_dataset=test)

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
