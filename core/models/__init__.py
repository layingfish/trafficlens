from .base import REGISTRY, Monitor, StopTraining  # noqa: F401
from . import stats, ml, deep  # noqa: F401  (registers models)

ORDER = ["ha", "snaive", "arima", "holtwinters", "ridge", "knn", "svr", "rf", "lightgbm",
         "mlp", "lstm", "gru", "seq2seq", "tcn", "transformer", "dlinear", "nbeats", "tgcn"]


def catalog():
    return [REGISTRY[k].spec() for k in ORDER if k in REGISTRY]
