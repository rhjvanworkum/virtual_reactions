import warnings
from typing import Optional, Dict, List, Type, Any

import pytorch_lightning as pl
import torch
from torch import nn as nn
from torchmetrics import Metric

from schnetpack.model.base import AtomisticModel

from schnetpack.task import AtomisticTask, ModelOutput, UnsupervisedModelOutput

DATALOADER_LIST = [
    "OOD",
    "IID",
    "VIRTUAL"
]

class SimulatedModelOutput(ModelOutput):

    def __init__(
        self,
        name: str,
        loss_fn: Optional[nn.Module] = None,
        loss_weight: float = 1.0,
        metrics: Optional[Dict[str, Metric]] = None,
        constraints: Optional[List[torch.nn.Module]] = None,
        target_property: Optional[str] = None,
    ):
        super().__init__(name, loss_fn, loss_weight, metrics, constraints, target_property)
        for key in ['OOD_test', 'IID_test', 'VIRTUAL_test']:
            self.metrics[key] = nn.ModuleDict({k: v.clone() for k, v in metrics.items()})


class SimulatedAtomisticTask(AtomisticTask):

    def __init__(
        self,
        model: AtomisticModel,
        outputs: List[ModelOutput],
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_args: Optional[Dict[str, Any]] = None,
        scheduler_cls: Optional[Type] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        scheduler_monitor: Optional[str] = None,
        warmup_steps: int = 0,
    ):
        super().__init__(model, outputs, optimizer_cls, optimizer_args, scheduler_cls, scheduler_args, scheduler_monitor, warmup_steps)

    def log_metrics(self, pred, targets, subset):
        for output in self.outputs:
            output.update_metrics(pred, targets, subset)
            for metric_name, metric in output.metrics[subset].items():
                if subset in ['OOD_test', 'IID_test', 'VIRTUAL_test']:
                    self.log(
                        f"{subset}_{output.name}_{metric_name}",
                        metric,
                        on_step=(subset == "train"),
                        on_epoch=(subset != "train"),
                        prog_bar=False,
                        metric_attribute='outputs.0.test_metrics.MAE'
                    )
                else:
                    self.log(
                        f"{subset}_{output.name}_{metric_name}",
                        metric,
                        on_step=(subset == "train"),
                        on_epoch=(subset != "train"),
                        prog_bar=False
                    )

    def test_step(self, batch, batch_idx, dataloader_idx):
        torch.set_grad_enabled(self.grad_enabled)

        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except:
            pass

        pred = self.predict_without_postprocessing(batch)
        pred, targets = self.apply_constraints(pred, targets)

        loss = self.loss_fn(pred, targets)

        self.log(f"{DATALOADER_LIST[dataloader_idx]}_test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_metrics(pred, targets, f"{DATALOADER_LIST[dataloader_idx]}_test")
        return {f"{DATALOADER_LIST[dataloader_idx]}_test_loss": loss}