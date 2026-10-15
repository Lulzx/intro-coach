"""User settings saved on this machine (~/.intro-coach/settings.json, readable only by the owner)."""
import json
import os

from .paths import HOME

FILE = HOME / "settings.json"


def load() -> dict:
    try:
        return json.loads(FILE.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def save(data: dict) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    tmp = FILE.with_suffix(".tmp")
    # create with owner-only permissions before the key is written
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)
    os.replace(tmp, FILE)


def openrouter_key() -> tuple[str | None, str | None]:
    """(key, source): a key saved in Settings wins over the OPENROUTER_API_KEY environment variable."""
    saved = load().get("openrouter_api_key")
    if saved:
        return saved, "settings"
    env = os.environ.get("OPENROUTER_API_KEY")
    return (env, "environment") if env else (None, None)


def set_openrouter_key(key: str | None) -> None:
    data = load()
    if key:
        data["openrouter_api_key"] = key
    else:
        data.pop("openrouter_api_key", None)
    save(data)


def mask(key: str) -> str:
    return key[:8] + "…" + key[-4:] if len(key) > 16 else "…" + key[-4:]
