"""
Centralized error handling utilities for agents.
"""

import functools
import traceback
from typing import Dict, Any, Optional, Callable
from enum import Enum

from ..utils.logger import get_logger


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentError(Exception):
    """Base exception for agent errors."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 agent_name: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.severity = severity
        self.agent_name = agent_name
        self.context = context or {}


class ProcessingError(AgentError):
    """Error during content processing."""
    pass


class LLMError(AgentError):
    """Error during LLM interaction."""
    pass


class ValidationError(AgentError):
    """Error during content validation."""
    pass


class ErrorHandler:
    """Centralized error handling for agents."""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = get_logger(agent_name)
        self.error_counts = {}
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None, 
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> Dict[str, Any]:
        """Handle error and return standardized error response."""
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        error_info = {
            'success': False,
            'error_type': error_type,
            'error_message': str(error),
            'severity': severity.value,
            'agent': self.agent_name,
            'context': context or {},
            'traceback': traceback.format_exc() if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] else None
        }
        
        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"Critical error in {self.agent_name}: {error}", extra=error_info)
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(f"High severity error in {self.agent_name}: {error}", extra=error_info)
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"Medium severity error in {self.agent_name}: {error}")
        else:
            self.logger.info(f"Low severity error in {self.agent_name}: {error}")
        
        return error_info
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics for this agent."""
        return self.error_counts.copy()


def with_error_handling(fallback_value: Any = None, 
                       severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """Decorator for agent methods to handle errors consistently."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                if hasattr(self, 'error_handler'):
                    error_response = self.error_handler.handle_error(
                        e, 
                        context={'method': func.__name__, 'args': str(args)[:100]},
                        severity=severity
                    )
                    
                    if fallback_value is not None:
                        error_response['fallback_result'] = fallback_value
                    
                    return error_response
                else:
                    # Fallback if no error handler
                    return {
                        'success': False,
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'agent': getattr(self, 'name', 'UnknownAgent')
                    }
        
        return wrapper
    return decorator


def create_fallback_response(agent_name: str, 
                            original_input: Dict[str, Any], 
                            error_message: str,
                            partial_result: Any = None) -> Dict[str, Any]:
    """Create standardized fallback response for failed operations."""
    response = {
        'success': False,
        'agent': agent_name,
        'error': error_message,
        'input_data': original_input,
        'timestamp': None  # Could add timestamp here
    }
    
    if partial_result is not None:
        response['partial_result'] = partial_result
    
    return response


class RetryManager:
    """Manages retry logic for agent operations."""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.5):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    def retry_on_failure(self, operation: Callable, 
                        *args, 
                        retry_exceptions: tuple = (Exception,),
                        **kwargs) -> Any:
        """Retry operation with exponential backoff."""
        import time
        
        for attempt in range(self.max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except retry_exceptions as e:
                if attempt == self.max_retries:
                    raise e
                
                wait_time = self.backoff_factor ** attempt
                time.sleep(wait_time)
        
        return None


def with_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 1.5):
    """Decorator for retry logic."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        time.sleep(wait_time)
                    else:
                        raise e
            
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


# Add class method to ErrorHandler
ErrorHandler.with_retry = staticmethod(with_retry)
