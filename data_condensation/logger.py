import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional
import numpy as np
import structlog
import wandb
from PIL import Image


# default json encoder for numpy objects
def default_json_encoder(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError


LOG: structlog.BoundLogger = structlog.get_logger(__name__)


class Logger:

    _experiment_name: str
    _logdir: str
    _save_metrics: bool
    _verbose: bool
    _metrics_buffer: Dict[str, List[float]]
    _params: Optional[Dict[str, float]]

    def __init__(
        self,
        experiment_name: str,
        save_metrics: bool = True,
        root_dir: str = "logs",
        verbose: bool = True,
        with_timestamp: bool = True,
    ):
        self._save_metrics = save_metrics
        self._verbose = verbose
        self._raw_experiment_name = experiment_name
        # add timestamp to prevent unintentional overwrites
        while True:
            if with_timestamp:
                date = datetime.now().strftime("%Y%m%d%H%M%S")
                self._experiment_name = experiment_name + "_" + date
            else:
                self._experiment_name = experiment_name

            if self._save_metrics:
                self._logdir = os.path.join(root_dir, self._experiment_name)
                if not os.path.exists(self._logdir):
                    os.makedirs(self._logdir)
                    LOG.info(f"Directory is created at {self._logdir}")
                    break
                if with_timestamp:
                    time.sleep(1.0)
                else:
                    raise ValueError("%s already exists." % self._logdir)
            else:
                break

        self._metrics_buffer = {}
        self._params = None
        self._wandb = False

    def wandb_init(self, project, name=None):
        wandb.init(
            project=project,
            config=self._params,
            name=name,
        )
        self._wandb = True

    def add_params(self, params: Dict[str, Any]) -> None:
        assert self._params is None, "add_params can be called only once."

        if self._save_metrics:
            # save dictionary as json file
            params_path = os.path.join(self._logdir, "params.json")
            with open(params_path, "w") as f:
                json_str = json.dumps(
                    params, default=default_json_encoder, indent=2
                )
                f.write(json_str)

            if self._verbose:
                LOG.info(
                    f"Parameters are saved to {params_path}", params=params
                )
        elif self._verbose:
            LOG.info("Parameters", params=params)

        # remove non-scaler values for HParams
        self._params = {k: v for k, v in params.items() if np.isscalar(v)}

    def add_metric(self, name: str, value: float) -> None:
        if name not in self._metrics_buffer:
            self._metrics_buffer[name] = []
        self._metrics_buffer[name].append(value)

    def commit(self, epoch: int, step: int, wandb_log: bool = True) -> Dict[str, float]:
        metrics = {}
        for name, buffer in self._metrics_buffer.items():

            metric = sum(buffer) / len(buffer)

            if self._save_metrics:
                path = os.path.join(self._logdir, f"{name}.csv")
                with open(path, "a") as f:
                    print("%d,%d,%f" % (epoch, step, metric), file=f)

            metrics[name] = metric

        if self._verbose:
            LOG.info(
                f"{self._experiment_name}: epoch={epoch} step={step}",
                epoch=epoch,
                step=step,
                metrics=metrics,
            )

        if wandb_log and self._wandb:
            metrics['epoch'] = epoch
            metrics['step'] = step
            wandb.log(metrics)

        # initialize metrics buffer
        self._metrics_buffer = {}
        return metrics

    @contextmanager
    def measure_time(self, name: str) -> Iterator[None]:
        name = "time_" + name
        start = time.time()
        try:
            yield
        finally:
            self.add_metric(name, time.time() - start)

    @property
    def logdir(self) -> str:
        return self._logdir

    @property
    def experiment_name(self) -> str:
        return self._experiment_name

    def clear_metric_buffer(self):
        self._metrics_buffer = {}

    def savez(self, file, *args, **kwds):
        if not os.path.exists(os.path.join(self._logdir, 'recs')):
            os.makedirs(os.path.join(self._logdir, 'recs'))
        self.info('Saving {}'.format(os.path.join(self._logdir, 'recs', file)))
        np.savez(os.path.join(self._logdir, 'recs', file), *args, **kwds)

    def save_image(self, file, image, wandb_log=True):
        assert isinstance(image, Image.Image)
        if not os.path.exists(os.path.join(self._logdir, 'img')):
            os.makedirs(os.path.join(self._logdir, 'img'))
        self.info('Saving {}'.format(os.path.join(self._logdir, 'img', file)))
        image.save(os.path.join(self._logdir, 'img', file))
        if wandb_log and self._wandb:
            wandb.log({file: [wandb.Image(image, caption=f"{file}")]})

    def info(self, text, file=None):
        if self._verbose:
            LOG.info(text)
