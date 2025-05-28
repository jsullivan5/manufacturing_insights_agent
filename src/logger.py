# src/logger.py (revised)
import logging
import logging.config
import os
from rich.logging import RichHandler
# from rich.traceback import Traceback # Not directly used, can be removed if not needed for other funcs
import argparse # Added to suppress RichHandler tracebacks for this module if needed
import click # Added to suppress RichHandler tracebacks for this module if needed

# Define a default log file path if MCP_LOG_FILE is not set
DEFAULT_LOG_FILE = "orchestrator_debug.log"

def setup_logging(
    default_level=logging.INFO,
    console_formatter: str = "flow",
    file_formatter: str = "verbose",
    log_file_path: str = None
):
    """
    Set up logging configuration for the application using RichHandler for the console.

    Args:
        default_level: The default logging level for the root logger and console.
        console_formatter: Formatter to use for console output ('flow' or 'verbose').
        file_formatter: Formatter to use for file output ('flow' or 'verbose').
        log_file_path: Optional path for the debug log file. Uses MCP_LOG_FILE env var
                       or 'orchestrator_debug.log' if not provided.
    """
    if log_file_path is None:
        log_file_path = os.getenv("MCP_LOG_FILE", DEFAULT_LOG_FILE)

    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "flow": {
                # "()": "rich.logging.RichFormatter", # RichHandler applies its own formatter
                "format": "%(message)s",
                "datefmt": "[%X]",
            },
            "verbose": {
                # "()": "rich.logging.RichFormatter",
                "format": "%(asctime)s [%(name)s:%(lineno)d] %(levelname)-8s %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "rich.logging.RichHandler",
                "level": default_level,
                "formatter": console_formatter, # This name refers to a formatter defined above
                "show_path": False, 
                "rich_tracebacks": True,
                "markup": True, # Enable markup in log messages for the handler
                "tracebacks_show_locals": True,
                "tracebacks_suppress": [argparse, click], 
            },
            "file_debug": {
                "class": "logging.handlers.RotatingFileHandler", 
                "level": logging.DEBUG, 
                "formatter": file_formatter, # This name refers to a formatter defined above
                "filename": log_file_path,
                "maxBytes": 1024 * 1024 * 5,  # 5 MB
                "backupCount": 2,
                "encoding": "utf-8",
            },
        },
        "root": { 
            "handlers": ["console", "file_debug"],
            "level": logging.DEBUG, 
        },
    }
    logging.config.dictConfig(LOGGING_CONFIG)
    # Use a specific logger for this message to avoid it being processed before config is fully applied
    # if root logger is immediately used.
    config_logger = logging.getLogger("mcp.config.logger") 
    config_logger.info("Logging setup complete. Console level: %s, File: %s", default_level, log_file_path)

# Example usage (typically called once at the start of your application)
if __name__ == '__main__':
    setup_logging(default_level=logging.DEBUG, console_formatter="verbose") 
    
    logger = logging.getLogger("my_app") 
    
    logger.debug("This is a [bold blue]debug[/bold blue] message from my_app.")
    logger.info("This is an [italic green]info[/italic green] message from my_app.")
    logger.warning("This is a [yellow]warning[/yellow] message from my_app.")
    logger.error("This is a [bold red]error[/bold red] message from my_app.")
    logger.critical("This is a [underline bold red]critical[/underline bold red] message from my_app.")

    # Test traceback
    try:
        x = 1 / 0
    except ZeroDivisionError:
        logger.exception("A handled exception (ZeroDivisionError) occurred:")

    other_logger = logging.getLogger("another.module")
    other_logger.info("Info from another module. It should also use Rich logging.")

    print(f"File logs should be in: {os.getenv('MCP_LOG_FILE', DEFAULT_LOG_FILE)}")