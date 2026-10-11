"""Portable paths for the externally supplied time-series datasets."""
import os
from pathlib import Path


def data_root():
    configured = os.environ.get("ML_TIMESERIES_DATA")
    return Path(configured).expanduser() if configured else Path(__file__).resolve().parent


def training_workbook(identifier, normalized=False):
    suffix = "_normalized" if normalized else ""
    return str(data_root() / "train" / ("M0{}{}.xlsx".format(identifier, suffix)))


def feature_csv(identifier):
    return str(data_root() / "NewDataM0{}.csv".format(identifier))
