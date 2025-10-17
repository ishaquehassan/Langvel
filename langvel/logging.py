"""
Structured logging system for Langvel framework.

Provides JSON-formatted logging with proper log levels, context,
and integration with centralized logging systems.

Usage:
    from langvel.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Agent started", extra={"agent": "CustomerSupport", "user_id": "123"})
"""

import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.

    Outputs logs in JSON format compatible with logging aggregators
    like ELK, Datadog, CloudWatch, etc.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        # Base log data
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add process and thread info
        if record.process:
            log_data['process_id'] = record.process
        if record.thread:
            log_data['thread_id'] = record.thread

        # Add extra fields if present
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                # Skip standard logging fields
                if key not in [
                    'name', 'msg', 'args', 'created', 'filename', 'funcName',
                    'levelname', 'levelno', 'lineno', 'module', 'msecs',
                    'pathname', 'process', 'processName', 'relativeCreated',
                    'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
                    'getMessage', 'message'
                ]:
                    try:
                        # Serialize complex objects
                        if isinstance(value, (str, int, float, bool, type(None))):
                            log_data[key] = value
                        else:
                            log_data[key] = str(value)
                    except:
                        pass

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add stack trace if present
        if record.stack_info:
            log_data['stack_trace'] = self.formatStack(record.stack_info)

        return json.dumps(log_data, default=str)


class ColoredConsoleFormatter(logging.Formatter):
    """
    Colored console formatter for human-readable logs.

    Used in development mode for better readability.
    """

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')

        # Build log line
        log_line = (
            f"{color}[{record.levelname}]{reset} "
            f"{timestamp} "
            f"{record.name} "
            f"- {record.getMessage()}"
        )

        # Add exception if present
        if record.exc_info:
            log_line += f"\n{self.formatException(record.exc_info)}"

        return log_line


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = True,
    console: bool = True
) -> logging.Logger:
    """
    Setup structured logging for Langvel.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        json_format: Use JSON format (True) or colored console format (False)
        console: Enable console output

    Returns:
        Configured root logger

    Example:
        # Production: JSON logs to file
        setup_logging(log_level="INFO", log_file="langvel.log", json_format=True)

        # Development: Colored console output
        setup_logging(log_level="DEBUG", json_format=False)
    """
    # Get root logger for langvel
    logger = logging.getLogger('langvel')
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()  # Remove existing handlers

    # Choose formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = ColoredConsoleFormatter()

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        # Always use JSON format for file output
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Agent started", extra={"agent_name": "CustomerSupport"})
    """
    # Ensure it's under langvel namespace
    if not name.startswith('langvel'):
        name = f'langvel.{name}'

    return logging.getLogger(name)


def configure_from_config():
    """
    Configure logging from Langvel config.

    Reads configuration from config/langvel.py and sets up logging accordingly.
    Should be called once at application startup.
    """
    try:
        from config.langvel import config

        setup_logging(
            log_level=getattr(config, 'LOG_LEVEL', 'INFO'),
            log_file=getattr(config, 'LOG_FILE', None),
            json_format=not getattr(config, 'DEBUG', False),  # Use colors in debug mode
            console=True
        )
    except ImportError:
        # Config not available, use defaults
        setup_logging(log_level='INFO', json_format=True)


# Initialize logging on import (can be reconfigured later)
_default_logger = setup_logging(log_level='INFO', json_format=True, console=True)
