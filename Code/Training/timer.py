

"""
models should log the time taken for various components of their pipeline

once the component name is repeated, the timer assumes the model is on the next training example
"""
from typing import Dict, Set

component_logs: Dict[str, float] = {}  # maps a components name to the sum of the times taken
component_names: Set[str] = set()
component_log_counts: Dict[str, float] = {}  # maps a components name to the number of passes so far


def log_time(component_name, time, increment_counter=True):
    """
    set increment counter to false if you need to log the same component from multiple places. Set true for final call
    """
    if component_name not in component_names:
        component_names.add(component_name)
        component_log_counts[component_name] = 0
        component_logs[component_name] = 0

    if increment_counter:
        component_log_counts[component_name] += 1
    component_logs[component_name] += time


def get_component_times():
    """returns the average times spent in each component logged"""
    global component_log_counts
    global component_names
    global component_logs

    av_times = {name.replace(" ", "_")+"_time": t / component_log_counts[name] for name, t in component_logs.items()}

    component_logs = {}  # maps a components name to the sum of the times taken
    component_names = set()
    component_log_counts = {}  # maps a components name to the number of passes so far
    return av_times