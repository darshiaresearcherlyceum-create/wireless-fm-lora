import os
import csv
import json
import logging
from typing import Dict, Any, List

class ExperimentLogger:
    """
    Handles training logs, metrics CSV files, and run manifests.
    """
    def __init__(
        self,
        log_dir: str = "logs",
        exp_name: str = "run_default",
        log_filename: str = "run_log.txt",
        csv_filename: str = "metrics.csv"
    ):
        self.log_dir = log_dir
        self.exp_name = exp_name
        self.run_dir = os.path.join(log_dir, exp_name)
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.run_dir, log_filename)
        self.csv_file = os.path.join(self.run_dir, csv_filename)
        self.csv_writer = None
        self.csv_handle = None
        
        self.logger = logging.getLogger(f"{exp_name}_{log_filename}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            fh = logging.FileHandler(self.log_file, encoding="utf-8")
            sh = logging.StreamHandler()
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            fh.setFormatter(fmt)
            sh.setFormatter(fmt)
            self.logger.addHandler(fh)
            self.logger.addHandler(sh)

    def log(self, message: str):
        self.logger.info(message)

    def log_metrics(self, epoch: int, metrics: Dict[str, Any]):
        row = {"epoch": epoch, **metrics}
        if self.csv_writer is None:
            self.csv_handle = open(self.csv_file, mode="w", newline="")
            self.csv_writer = csv.DictWriter(self.csv_handle, fieldnames=list(row.keys()))
            self.csv_writer.writeheader()
        self.csv_writer.writerow(row)
        self.csv_handle.flush()

    def close(self):
        if self.csv_handle is not None:
            self.csv_handle.close()
