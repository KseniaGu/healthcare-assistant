from functools import wraps
from typing import Any, Callable, TypeVar, Awaitable

from backend.utils.exceptions import CriticalDatabaseSideError

# A type variable for the return type of the wrapped function
T = TypeVar('T')


def connection(method: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Decorator for database session management in async methods.
    
    This decorator handles the database session lifecycle for async methods that require
    database access.
    
    Args:
        method: The async method to be wrapped. Must accept a `session` parameter.
    """

    @wraps(method)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> T:
        """Wrapper function that manages the database session."""
        session_factory = getattr(self, 'async_session', None)
        if not session_factory:
            raise CriticalDatabaseSideError("Database client is not properly initialized with async_session")

        # Create a new session
        async with session_factory() as session:
            try:
                result = await method(self, *args, session=session, **kwargs)
                await session.commit()
                return result

            except CriticalDatabaseSideError:
                raise

            except Exception as e:
                if session.in_transaction():
                    await session.rollback()
                raise CriticalDatabaseSideError(f"Unexpected database error in {method.__name__}: {str(e)}") from e

    return wrapper
