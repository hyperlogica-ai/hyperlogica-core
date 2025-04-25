# hyperlogica/error_handling.py
from typing import Dict, Tuple, Any, Callable, TypeVar, Union

# Type variables for generic functions
T = TypeVar('T')
E = TypeVar('E')
R = TypeVar('R')

# Result type for functions that might fail
Result = Union[Tuple[T, None], Tuple[None, E]]

def success(value: T) -> Result[T, Any]:
    """Create a success result."""
    return (value, None)

def error(err: E) -> Result[Any, E]:
    """Create an error result."""
    return (None, err)

def is_success(result: Result[T, E]) -> bool:
    """Check if a result is successful."""
    return result[1] is None

def is_error(result: Result[T, E]) -> bool:
    """Check if a result is an error."""
    return result[1] is not None

def get_value(result: Result[T, E]) -> T:
    """Get the value from a successful result."""
    if is_error(result):
        raise ValueError(f"Cannot get value from error result: {result[1]}")
    return result[0]

def get_error(result: Result[T, E]) -> E:
    """Get the error from an error result."""
    if is_success(result):
        raise ValueError("Cannot get error from success result")
    return result[1]

def map_success(result: Result[T, E], fn: Callable[[T], R]) -> Result[R, E]:
    """Apply a function to the value if the result is successful."""
    if is_success(result):
        return success(fn(get_value(result)))
    return result  # Return the error unchanged

def handle_error(result: Result[T, E], handler: Callable[[E], R]) -> Union[T, R]:
    """Handle the error if the result is an error, otherwise return the value."""
    if is_success(result):
        return get_value(result)
    return handler(get_error(result))