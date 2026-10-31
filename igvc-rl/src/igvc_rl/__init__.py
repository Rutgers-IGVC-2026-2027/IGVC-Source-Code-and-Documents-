"""IGVC navigation; register the standalone environment when Gymnasium is installed."""

from importlib.util import find_spec

__version__ = "0.1.0"
if find_spec("gymnasium") is not None:
    from gymnasium.envs.registration import register, registry

    if "IGVCNav-v0" not in registry:
        register(id="IGVCNav-v0", entry_point="igvc_rl.envs.nav_env:IGVCNavEnv")
