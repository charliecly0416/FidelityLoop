"""Policy/guard adapters; dispatch remains a first-class operation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Mapping, Any
import math
from fidelityloop.legacy.maxopt_v2.engine import TargetPolicy
from fidelityloop.legacy.maxopt_v3.policy import DeadlinePolicy


@dataclass
class TargetPolicyAdapter:
    spec: Mapping[str, Any] | str = "all1"
    max_devices: int = 2
    def __post_init__(self):
        if not 1 <= self.max_devices <= 8:
            raise ValueError("max_devices outside bounded facade")
        raw = {"name": self.spec} if isinstance(self.spec, str) else dict(self.spec)
        self.spec = raw
        self.state = {"target": min(self.max_devices, int(raw.get("target", 1))), "last_change_at": -10**9}
        if self.max_devices == 2 and raw.get("name") in {"all1", "all2", "hpa", "hysteresis"}:
            self._legacy = TargetPolicy(raw)
            self.state = self._legacy.state
        else:
            self._legacy = None
            name = raw.get("name")
            if name not in {"all1", "all2", "allN", "hpa", "hysteresis"}:
                raise ValueError("unsupported policy for bounded adapter")
            if name == "all2" and self.max_devices < 2:
                raise ValueError("all2 requires two devices")
            if name == "allN":
                target = raw.get("target", self.max_devices)
                if type(target) is not int or not 1 <= target <= self.max_devices:
                    raise ValueError("allN target outside device bound")
            self._cooldown = int(raw.get("cooldown_seconds", 0))

    def decide(self, observation):
        if self._legacy is not None:
            target = self._legacy.decide(_base_observation(observation))
            return {"target": target, "reserve": False, "base_target": target,
                    "risky_ids": [], "schema": "target-adapter-v1"}
        _validate_base_observation(observation, self.max_devices)
        now = observation["time"]
        old = self.state["target"]
        name = self.spec["name"]
        if name == "all1": target = 1
        elif name == "all2": target = 2
        elif name == "allN": target = self.spec.get("target", self.max_devices)
        else:
            if now - self.state["last_change_at"] < self._cooldown:
                target = old
            else:
                pressure = observation["queue_online"] + observation["queue_offline"] + observation["local_running"]
                if name == "hpa":
                    target = math.ceil(pressure / max(1, int(self.spec.get("target_pressure", 4))))
                elif pressure > int(self.spec.get("up_threshold", 6)) * max(1, old):
                    target = old + 1
                elif pressure <= int(self.spec.get("down_threshold", 1)) * max(1, old - 1):
                    target = old - 1
                else: target = old
        target = max(int(self.spec.get("min_devices", 0)), min(self.max_devices, int(target)))
        if target != old: self.state = {"target": target, "last_change_at": now}
        return {"target": target, "reserve": False, "base_target": target,
                "risky_ids": [], "schema": "target-adapter-v1"}

    def dispatch_kind(self, observation, reserve=False):
        online, offline = observation.get("queue_online", []), observation.get("queue_offline", [])
        if not isinstance(online, list): online = []
        if not isinstance(offline, list): offline = []
        if not online: return "offline" if offline else None
        if not offline or not reserve: return "online"
        jobs = [j for d in observation.get("devices", []) for j in d.get("jobs", [])]
        if any(j.get("job_type") == "offline" for j in jobs): return "online"
        on = online[0].get("deadline_s", 0) - observation["time"]
        off = offline[0].get("deadline_s", 0) - observation["time"]
        return "online" if on <= off else "offline"


class GuardPolicyAdapter:
    """Use frozen DeadlinePolicy for historical two-device runs; bounded fallback otherwise."""
    def __init__(self, spec, model, safety, *, max_devices=2, estimator=None,
                 scale=True, reservation=True):
        self.spec, self.model, self.safety = spec, model, safety
        self.max_devices, self.scale, self.reservation = max_devices, scale, reservation
        self._legacy = None
        if max_devices == 2 and estimator is not None and hasattr(estimator, "model"):
            self._legacy = DeadlinePolicy(spec, model, safety, scale=scale, reservation=reservation)
        self._target = TargetPolicyAdapter(spec, max_devices=max_devices)
        self.latched = set()
        self.state = self._target.state
        self.estimator = estimator

    def decide(self, observation):
        if self._legacy is not None:
            return self._legacy.decide(observation)
        _validate_base_observation(observation, self.max_devices)
        base = self._target.decide(_base_observation(observation))["target"]
        pending = observation.get("queue_offline", [])
        ids = {j["request_id"] for j in pending}
        self.latched.intersection_update(ids)
        risky = {j["request_id"] for j in pending if j.get("deadline_s", 0) - observation["time"] <= 1}
        self.latched.update(risky)
        target = base
        if self.latched and self.scale: target = max(base, min(self.max_devices, base + 1))
        return {"base_target": base, "target": target,
                "reserve": bool(self.latched and self.reservation),
                "risky_ids": sorted(self.latched), "infeasible": False,
                "schema": "bounded-guard-v1"}

    def dispatch_kind(self, observation, reserve):
        if self._legacy is not None: return self._legacy.dispatch_kind(observation, reserve)
        online, offline = observation.get("queue_online", []), observation.get("queue_offline", [])
        if not online: return "offline" if offline else None
        if not offline or not reserve: return "online"
        running = [j for d in observation.get("devices", []) for j in d.get("jobs", [])]
        if any(j.get("job_type") == "offline" for j in running): return "online"
        now = observation["time"]
        def eta(job):
            try: return float(self.estimator.seconds(job, 1))
            except (AttributeError, TypeError): return 0.0
        online_slack = online[0]["deadline_s"] - now - eta(online[0])
        offline_slack = offline[0]["deadline_s"] - now - eta(offline[0])
        return "online" if online_slack <= 5.0 and online_slack <= offline_slack else "offline"


class PPOActionAdapter:
    """Map existing PPO scale actions (0/1/2) to bounded target actions."""
    def __init__(self, action_source: Callable | int, *, max_devices=2, min_devices=0, dispatch=None):
        self.action_source, self.max_devices, self.min_devices = action_source, max_devices, min_devices
        self.dispatch = dispatch
        if not 1 <= max_devices <= 8 or not 0 <= min_devices <= max_devices: raise ValueError("invalid PPO bounds")
        self.state = {"target": min_devices}
    def decide(self, observation):
        raw = self.action_source(observation) if callable(self.action_source) else self.action_source
        if isinstance(raw, Mapping): raw = raw.get("scale_action", raw.get("action"))
        if type(raw) is not int or raw not in (0, 1, 2): raise ValueError("PPO scale action must be 0, 1, or 2")
        current = sum(d.get("state") in ("active", "starting") for d in observation.get("devices", []))
        if raw == 0: target = current
        elif raw == 1: target = min(self.max_devices, current + 1)
        else: target = max(self.min_devices, current - 1)
        self.state["target"] = target
        return {"target": target, "reserve": False, "scale_action": raw,
                "base_target": target, "risky_ids": [], "schema": "ppo-action-adapter-v1"}
    def dispatch_kind(self, observation, reserve=False):
        if self.dispatch is None: return TargetPolicyAdapter("all1", self.max_devices).dispatch_kind(observation, reserve)
        return self.dispatch(observation, reserve)


def _base_observation(obs):
    return {"time": obs["time"], "queue_online": len(obs["queue_online"]) if isinstance(obs.get("queue_online"), list) else obs.get("queue_online", 0),
            "queue_offline": len(obs["queue_offline"]) if isinstance(obs.get("queue_offline"), list) else obs.get("queue_offline", 0),
            "local_running": sum(len(d.get("jobs", [])) for d in obs.get("devices", [])) if "devices" in obs else obs.get("local_running", 0),
            **{s: sum(d.get("state") == s for d in obs.get("devices", [])) for s in ("active", "starting", "off")},
            "draining": sum(d.get("state") in ("draining", "stopping") for d in obs.get("devices", []))}


def _validate_base_observation(obs, max_devices):
    if type(obs.get("time")) is not int or len(obs.get("devices", [])) != max_devices:
        raise ValueError("observation/device count mismatch")
