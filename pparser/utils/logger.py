"""
Logging utilities for PPARSER system
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from ..config import config


def setup_logger(
    name: str = "pparser",
    level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """Setup logger with file and console handlers"""
    
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Set level
    log_level = level or config.log_level
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file or config.log_file:
        file_path = Path(log_file or config.log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Global logger instance
logger = setup_logger()


def get_logger(name: str = "pparser") -> logging.Logger:
    """Get a logger instance with the specified name"""
    return setup_logger(name)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure global logging settings"""
    setup_logger("pparser", level, log_file)
