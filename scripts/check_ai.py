"""Safe provider health diagnostic. Never prints credentials or provider exceptions."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from config.environment import load_environment

load_environment()

from services.ai.manager import AIManager


def main():
    manager = AIManager()
    status = manager.health(generation_test=True)
    print(f"AI Provider: {status.provider.title()}")
    print(f"Configured: {'YES' if status.configured else 'NO'}")
    print(f"Reachable: {'YES' if status.reachable else 'NO'}")
    print(f"Model: {'available' if status.model_available else 'unavailable'}")
    print(f"Generation test: {'PASS' if status.generation_successful else 'FAIL'}")
    print(f"Fallback: {'ACTIVE' if status.fallback_active else 'NOT ACTIVE'}")
    if status.message: print(f"Status: {status.message}")
    return 0 if status.generation_successful else 1


if __name__ == "__main__": raise SystemExit(main())
