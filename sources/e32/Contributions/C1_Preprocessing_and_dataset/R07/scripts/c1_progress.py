"""Small console progress estimates; no scheduler, network or scientific execution."""
import math
import statistics
import time


def duration_text(seconds):
    if seconds is None:
        return "estimation en cours"
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def remaining_seconds(total, completed, durations, active_elapsed=0):
    if (not isinstance(total, int) or not isinstance(completed, int) or isinstance(total, bool) or
            isinstance(completed, bool) or total < 0 or not 0 <= completed <= total or active_elapsed < 0):
        raise ValueError("Invalid progress counts or elapsed time")
    if completed == total:
        return 0.0
    observed = [float(value) for value in durations if math.isfinite(value) and value > 0]
    if not observed:
        return None
    typical = statistics.median(observed)
    # An overrun is not zero remaining time. Reserve one typical unit until it returns.
    current = typical - active_elapsed if 0 < active_elapsed < typical else typical
    return max(0, total - completed - 1) * typical + current


class Progress:
    def __init__(self, clock=time.monotonic, emit=print):
        self.clock, self.emit, self.started = clock, emit, clock()

    def update(self, phase, counts, totals, durations, active_elapsed=None, states=None):
        active_elapsed = active_elapsed or {}
        states = states or {"PC1": "NON_PRECISE", "PC2": "NON_PRECISE"}
        estimates = [remaining_seconds(totals[role], counts[role], durations.get(role, []), active_elapsed.get(role, 0))
                     for role in ("PC1", "PC2")]
        eta = None if any(value is None for value in estimates) else max(estimates)
        elapsed = self.clock() - self.started
        remaining = duration_text(eta) if phase == "ENTRAINEMENT" else (
            "termine" if phase.startswith("TERMINE") else "validation/analyse non estimee")
        self.emit(f"[AVANCEMENT] {phase} | PC1 {counts['PC1']}/{totals['PC1']} taches retournees [{states['PC1']}]"
                  f" | PC2 {counts['PC2']}/{totals['PC2']} taches retournees [{states['PC2']}]"
                  f" | ecoule {duration_text(elapsed)} | restant ~{remaining}"
                  " | gain: non mesure", flush=True)
        return {"phase": phase, "returned": counts, "total": totals, "pc_states": states, "elapsed_seconds": elapsed,
                "estimated_remaining_execution_seconds": eta,
                "estimate_assumption": "one sequential unit per host; median comparable observed unit durations; final analysis excluded",
                "speedup_measured": None, "completion_requires_verified_barrier": True}
