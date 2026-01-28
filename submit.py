
import logging
import os
import sys
import traceback
from typing import Any

import hydra
import numpy as np
from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)
CONFIG_PATH = "configs"
NAME_MAX = 255

@hydra.main(version_base=None, config_name="conf", config_path=CONFIG_PATH)
def my_app(cfg: DictConfig) -> None:
    log.info("Running locally.")
    
    main_fn = __import__("src." + cfg.experiment, fromlist=[None]).main
    log.info(f"Beginning experiment [{cfg.experiment}].")
    try:
        main_fn(cfg)
    except Exception as e:
        raise e 
    log.info(f"Completed experiment.")

class LogJobReturnCallback(Callback):
    def __init__(self) -> None:
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def on_job_end(
        self, config: DictConfig, job_return: JobReturn, **kwargs: Any
    ) -> None:
        if job_return.status == JobStatus.COMPLETED:
            self.log.info(f"Succeeded with return value: {job_return.return_value}")
        elif job_return.status == JobStatus.FAILED:
            e_str = "".join(traceback.format_exception(None, # <- type(e) by docs, but ignored 
                                                                job_return._return_value,
                                                                job_return._return_value.__traceback__))
            # We use sys.stderr to ensure that it always prints as some code overwrites print.
            sys.stderr.write(e_str) 
            self.log.error("", exc_info=job_return._return_value)
        else:
            self.log.error("Status unknown. This should never happen.")

if __name__ == "__main__":
    # Omegaconf resolvers for managing configs.
    OmegaConf.register_new_resolver("prod", lambda *args: np.prod(args).item()) # product arguments
    OmegaConf.register_new_resolver("replace_slash", lambda s: s.replace("/", ".")) # replaces slashes in str
    OmegaConf.register_new_resolver("join_path", lambda *args: os.path.join(*args)) # joins paths
    OmegaConf.register_new_resolver("join_overlays", lambda *args: ','.join(filter(None, args))) # joins overlays")
    OmegaConf.register_new_resolver("limit_path_length", lambda s: s[:NAME_MAX]) # limits path length

    my_app()
