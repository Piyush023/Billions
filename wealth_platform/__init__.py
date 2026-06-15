"""AI Wealth Platform.

Importing this package auto-loads `.env` from the project root so API keys
are available without manual exports. Existing environment variables are
never overridden — exported values win over .env values.
"""

import os
from pathlib import Path

from wealth_platform.paths import PROJECT_ROOT


def _load_dotenv(path: Path = PROJECT_ROOT / ".env"):
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()
