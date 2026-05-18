import torch
import numpy as np
import random
from tensorboard.backend.event_processing import event_accumulator

def deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_from_tensorboard(experiment_dir, tag):
    event_files = sorted(experiment_dir.rglob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)

    latest_event_file = event_files[-1]
    ea = event_accumulator.EventAccumulator(str(latest_event_file))
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return None, None
    scalar_events = ea.Scalars(tag)
    steps = [e.step for e in scalar_events]
    values = [e.value for e in scalar_events]
    return steps, values
