"""Finetune a pretrained POYO+ model on new sessions (e.g. a brand new animal).

The new recording brings neurons the model has never seen. Their unit embeddings (and
the session embedding) are created fresh, while the rest of the pretrained network is
re-used. With ``freeze_perceiver_until_epoch: N`` the transformer stays frozen for the
first N epochs ("unit identification": only the new embeddings are learned), after
which the whole model is finetuned.

    python finetune.py --config-name demo_finetune
"""

import logging
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ["HYDRA_FULL_ERROR"] = "1"

from typing import List, Optional

import hydra
import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig

from torch_brain.registry import MODALITIY_REGISTRY
from torch_brain.utils import callbacks as tbrain_callbacks
from torch_brain.utils import seed_everything
from torch_brain.utils.datamodules import DataModule
from torch_brain.utils.stitcher import MultiTaskDecodingStitchEvaluator
from torch_brain.utils.terminal import TerminalNotifier

from train import POYOTrainWrapper, build_logger, build_trainer, build_ui_callbacks

# higher speed on machines with tensor cores
torch.set_float32_matmul_precision("medium")


class GradualUnfreezing(L.Callback):
    r"""A Lightning callback to handle freezing and unfreezing of the model for the
    purpose of finetuning the model to new sessions. If this callback is used,
    most of the model weights will be frozen initially.
    The only parts of the model that will be left unforzen are the unit, and session embeddings.
    One we reach the specified epoch (`unfreeze_at_epoch`), the entire model will be unfrozen.
    """

    _has_been_frozen: bool = False
    frozen_params: Optional[List[nn.Parameter]] = None

    def __init__(self, unfreeze_at_epoch: int):
        self.enabled = unfreeze_at_epoch != 0
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.cli_log = logging.getLogger(__name__)

    @classmethod
    def freeze(cls, model):
        r"""Freeze the model weights, except for the unit and session embeddings, and
        return the list of frozen parameters.
        """
        layers_to_freeze = [
            model.enc_atn,
            model.enc_ffn,
            model.proc_layers,
            model.dec_atn,
            model.dec_ffn,
            model.readout,
            model.token_type_emb,
            model.task_emb,
            model.latent_emb,
        ]

        frozen_params = []
        for layer in layers_to_freeze:
            for param in layer.parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    frozen_params.append(param)

        return frozen_params

    def on_train_start(self, trainer, pl_module):
        if self.enabled:
            self.frozen_params = self.freeze(pl_module.model)
            self._has_been_frozen = True
            self.cli_log.info(
                f"POYO+ Perceiver frozen at epoch 0. "
                + (
                    f"Will stay frozen until epoch {self.unfreeze_at_epoch}."
                    if self.unfreeze_at_epoch > 0
                    else "Will stay frozen (unit identification only)."
                )
            )

    def on_train_epoch_start(self, trainer, pl_module):
        if self.enabled and (trainer.current_epoch == self.unfreeze_at_epoch):
            if not self._has_been_frozen:
                raise RuntimeError("Model has not been frozen yet.")
            for param in self.frozen_params:
                param.requires_grad = True
            self.frozen_params = None
            self.cli_log.info(
                f"POYO+ Perceiver unfrozen at epoch {trainer.current_epoch}"
            )


def load_model_from_ckpt(model: nn.Module, ckpt_path: str) -> None:
    if ckpt_path is None:
        raise ValueError("Must provide a checkpoint path to finetune the model.")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    state_dict = {
        k[len("model.") :]: v for k, v in state_dict.items() if k.startswith("model.")
    }
    model.load_state_dict(state_dict)


@hydra.main(version_base="1.3", config_path="./configs", config_name="demo_finetune")
def main(cfg: DictConfig):
    # fix random seed, skipped if cfg.seed is None
    seed_everything(cfg.seed)

    log = logging.getLogger(__name__)
    logger = build_logger(cfg)

    # make model and load the pretrained weights
    model = hydra.utils.instantiate(cfg.model, readout_specs=MODALITIY_REGISTRY)
    ckpt_path = hydra.utils.to_absolute_path(cfg.ckpt_path)
    load_model_from_ckpt(model, ckpt_path)
    log.info(f"Loaded model weights from {ckpt_path}")

    # setup data module
    data_module = DataModule(cfg, model.unit_emb.tokenizer, model.session_emb.tokenizer)
    data_module.setup()

    # register the new units and sessions: brand new neurons get fresh embeddings,
    # and the vocabulary is then restricted to the recordings we finetune on
    unit_ids, session_ids = data_module.get_unit_ids(), data_module.get_session_ids()
    model.unit_emb.extend_vocab(unit_ids, exist_ok=True)
    model.unit_emb.subset_vocab(unit_ids)
    model.session_emb.extend_vocab(session_ids, exist_ok=True)
    model.session_emb.subset_vocab(session_ids)

    # Lightning train wrapper
    wrapper = POYOTrainWrapper(
        cfg=cfg,
        model=model,
        dataset_config_dict=data_module.get_recording_config_dict(),
        steps_per_epoch=len(data_module.train_dataloader()),
    )

    evaluator = MultiTaskDecodingStitchEvaluator(
        dataset_config_dict=data_module.get_recording_config_dict()
    )

    callbacks = [
        evaluator,
        *build_ui_callbacks(cfg),
        ModelCheckpoint(
            save_last=True,
            monitor="average_val_metric",
            mode="max",
            save_on_train_epoch_end=True,
            every_n_epochs=cfg.eval_epochs,
        ),
        LearningRateMonitor(logging_interval="step"),
        tbrain_callbacks.MemInfo(),
        tbrain_callbacks.EpochTimeLogger(),
        GradualUnfreezing(cfg.freeze_perceiver_until_epoch),
        TerminalNotifier(run_name=cfg.wandb.run_name or "POYO+ finetune"),
    ]

    trainer = build_trainer(cfg, callbacks, logger)

    log.info(
        f"Local rank/node rank/world size/num nodes: "
        f"{trainer.local_rank}/{trainer.node_rank}/{trainer.world_size}/{trainer.num_nodes}"
    )

    # Train (the checkpoint only provides initial weights; we do not resume from it)
    trainer.fit(wrapper, data_module)

    # optionally copy the final finetuned checkpoint to a stable location (the final
    # model, not the "best" epoch: picking the best epoch on the held-out trials would
    # leak test information into the reported score)
    export = cfg.get("export_ckpt")
    last = trainer.checkpoint_callback.last_model_path
    if export and last and trainer.is_global_zero:
        import shutil

        export = hydra.utils.to_absolute_path(export)
        os.makedirs(os.path.dirname(export), exist_ok=True)
        shutil.copyfile(last, export)
        score = trainer.callback_metrics.get("average_val_metric")
        score = f" (final val R² = {float(score):.3f})" if score is not None else ""
        log.info(f"Final checkpoint{score} copied to {export}")


if __name__ == "__main__":
    main()
