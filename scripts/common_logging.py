#!/usr/bin/env python3
"""Shared logging helpers for ChainBench pipeline scripts."""

from __future__ import annotations

import re
import sys
from typing import Any

from loguru import logger
from tqdm.auto import tqdm


LOG_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <7}</level> | "
    "<cyan>{extra[component]}</cyan> | "
    "<level>{message}</level>"
)

STREAM_LOG_PREFIX_PATTERNS = (
    re.compile(r"^\d{2}:\d{2}:\d{2}\s+\|\s+[A-Z]+\s+\|\s+[^|]+\|\s+"),
    re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:,\d+)?\s+[A-Z]+\s+"),
)


def _tqdm_sink(message: Any) -> None:
    tqdm.write(str(message).rstrip("\n"), file=sys.stderr)


def setup_logging(level: str) -> None:
    logger.remove()
    logger.add(
        _tqdm_sink,
        level=level.upper(),
        colorize=sys.stderr.isatty(),
        format=LOG_FORMAT,
    )


class LoggerAdapter:
    def __init__(self, component: str) -> None:
        self._logger = logger.bind(component=component)

    def _render(self, message: str, *args: Any) -> str:
        if not args:
            return message
        try:
            return message % args
        except Exception:
            return " ".join([message, *[str(arg) for arg in args]])

    def info(self, message: str, *args: Any) -> None:
        self._logger.info(self._render(message, *args))

    def warning(self, message: str, *args: Any) -> None:
        self._logger.warning(self._render(message, *args))

    def error(self, message: str, *args: Any) -> None:
        self._logger.error(self._render(message, *args))

    def success(self, message: str, *args: Any) -> None:
        self._logger.success(self._render(message, *args))


def get_logger(component: str):
    return LoggerAdapter(component)


def format_elapsed(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def progress_bar(done: int, total: int, width: int = 18) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled = min(width, round((done / total) * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def clean_stream_line(line: str) -> str:
    cleaned = line.strip()
    for pattern in STREAM_LOG_PREFIX_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned
