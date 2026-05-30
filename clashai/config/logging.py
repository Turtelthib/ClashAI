# clashai/config/logging.py
# Centralised logging + pretty-print for ClashAI.
#
# Two outputs:
#   1. Console — rich-formatted, colored per level + per tag.
#   2. File   — logs/clashai.log, rotated at ~5 MB, 3 backups (plain text).
#
# Theme (colors, tags, etc.) is read from configs/logs/theme.yaml. If the
# file is missing or malformed, sensible defaults kick in — no crash.
# Level configurable via env var CLASHAI_LOG_LEVEL (DEBUG/INFO/WARNING/ERROR).
#
# Usage in modules:
#     from clashai.config.logging import get_logger, pp
#     log = get_logger(__name__)
#     log.info("starting attack")           # standard logger
#     pp("Step 3 [deploy]: golem", tag='deploy')   # pretty inline output
#     pp("EPISODE #1", tag='banner')               # framed banner

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from rich.panel import Panel
from rich.traceback import install as install_rich_tracebacks


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_LEVEL = os.environ.get('CLASHAI_LOG_LEVEL', 'INFO').upper()
LOG_DIR = os.environ.get('CLASHAI_LOG_DIR', 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'clashai.log')
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3

THEME_FILE = os.path.join('configs', 'logs', 'theme.yaml')

# Built-in defaults — used if configs/logs/theme.yaml is missing or invalid.
DEFAULT_THEME = {
    'levels': {
        'DEBUG':    'dim cyan',
        'INFO':     'white',
        'WARNING':  'bold yellow',
        'ERROR':    'bold red',
        'CRITICAL': 'bold red on yellow',
    },
    'tags': {
        'banner':  'bold cyan on black',
        'section': 'bold magenta',
        'step':    'white',
        'deploy':  'green',
        'spell':   'bold blue',
        'ability': 'bold magenta',
        'observe': 'dim white',
        'wait':    'dim white',
        'reward':  'bold yellow',
        'done':    'bold green',
        'warning': 'bold yellow',
        'error':   'bold red',
        'ok':      'bold green',
        'skip':    'dim white',
    },
    'show_path':       False,
    'show_time':       True,
    'markup':          True,
    'rich_tracebacks': True,
}


def _load_theme() -> dict:
    """Read configs/logs/theme.yaml, merge over the built-in defaults.
    Never raises — bad YAML / missing file just gives the defaults."""
    if not os.path.exists(THEME_FILE):
        return DEFAULT_THEME
    try:
        import yaml
        with open(THEME_FILE, encoding='utf-8') as f:
            user = yaml.safe_load(f) or {}
    except Exception:
        return DEFAULT_THEME
    # Shallow merge so user only needs to override what they care about.
    merged = {**DEFAULT_THEME, **user}
    merged['levels'] = {**DEFAULT_THEME['levels'], **(user.get('levels') or {})}
    merged['tags'] = {**DEFAULT_THEME['tags'], **(user.get('tags') or {})}
    return merged


_theme_cfg = _load_theme()


# -----------------------------------------------------------------------------
# Rich Console singleton
# -----------------------------------------------------------------------------

# Build rich Theme so [tagname] / [level] markup auto-applies the style.
_rich_theme_styles = {
    f'logging.level.{k.lower()}': v for k, v in _theme_cfg['levels'].items()
}
_rich_theme_styles.update(_theme_cfg['tags'])

_console = Console(
    theme=Theme(_rich_theme_styles),
    markup=_theme_cfg['markup'],
    soft_wrap=False,
    emoji=False,
)


def console() -> Console:
    """Get the project-wide rich Console (use for direct prints)."""
    return _console


# -----------------------------------------------------------------------------
# Setup (idempotent)
# -----------------------------------------------------------------------------

_configured = False


def _configure_once() -> None:
    """Install the rich console handler + file handler on the project root
    logger. Idempotent."""
    global _configured
    if _configured:
        return

    if _theme_cfg.get('rich_tracebacks'):
        install_rich_tracebacks(console=_console, show_locals=False)

    root = logging.getLogger('clashai')
    root.setLevel(DEFAULT_LEVEL)
    root.propagate = False

    # Rich console handler — colors per level, optional time + path
    rich_handler = RichHandler(
        console=_console,
        show_time=_theme_cfg['show_time'],
        show_level=True,
        show_path=_theme_cfg['show_path'],
        markup=_theme_cfg['markup'],
        rich_tracebacks=_theme_cfg.get('rich_tracebacks', True),
        omit_repeated_times=False,
    )
    rich_handler.setLevel(DEFAULT_LEVEL)
    root.addHandler(rich_handler)

    # File handler — plain text, full info for post-mortem
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        fh = RotatingFileHandler(
            LOG_FILE,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding='utf-8',
        )
        fh.setLevel(DEFAULT_LEVEL)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)-7s %(name)s | %(message)s',
            '%Y-%m-%d %H:%M:%S',
        ))
        root.addHandler(fh)
    except OSError as e:
        root.warning(f"Could not open log file {LOG_FILE}: {e}")

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Returns a logger named under the `clashai` hierarchy."""
    _configure_once()
    if not name.startswith('clashai'):
        name = f'clashai.{name}'
    return logging.getLogger(name)


def set_level(level: str) -> None:
    """Change the log level at runtime (e.g. from a CLI flag)."""
    _configure_once()
    root = logging.getLogger('clashai')
    root.setLevel(level)
    for h in root.handlers:
        h.setLevel(level)


# -----------------------------------------------------------------------------
# Pretty-print helpers (for in-game output that isn't really "logging")
# -----------------------------------------------------------------------------

def pp(message: str, tag: Optional[str] = None) -> None:
    """
    Print a styled inline message using the theme tag. Rich markup
    (e.g. produced by styled()) inside `message` is still interpreted —
    inner styles win over the outer tag for the spans they cover.
    """
    _configure_once()
    style = _theme_cfg['tags'].get(tag) if tag else None
    _console.print(message, style=style)


def banner(title: str, subtitle: Optional[str] = None,
           tag: str = 'banner') -> None:
    """Print a framed banner for major sections (episode start, summary…)."""
    _configure_once()
    style = _theme_cfg['tags'].get(tag, 'bold cyan')
    text = f"[{style}]{title}[/{style}]"
    if subtitle:
        text += f"\n[dim]{subtitle}[/dim]"
    _console.print(Panel(text, border_style=style, expand=False))


def section(title: str) -> None:
    """Print a section header (lighter than banner)."""
    _configure_once()
    style = _theme_cfg['tags'].get('section', 'bold magenta')
    _console.rule(f"[{style}]{title}[/{style}]", style=style)


def priority_tag(prio: int, max_prio: int = 10) -> str:
    """
    Map a priority value (1..max_prio) to a theme tag for color coding.
    Higher = more critical (red), lower = safe (green).
    """
    if max_prio <= 0:
        return 'step'
    ratio = prio / max_prio
    if ratio >= 0.85:
        return 'prio_max'
    if ratio >= 0.65:
        return 'prio_high'
    if ratio >= 0.35:
        return 'prio_med'
    return 'prio_low'


def styled(text: str, tag: str) -> str:
    """Wrap `text` with rich markup for the given tag (for use inside
    larger pp/log messages where you want a single token highlighted)."""
    _configure_once()
    style = _theme_cfg['tags'].get(tag)
    if not style:
        return text
    return f"[{style}]{text}[/{style}]"
