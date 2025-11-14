import sys
import functools
from datetime import datetime
from pathlib import Path
import os

class Tee:
    """Redirect stdout or stderr to both terminal and a log file."""
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class SuppressStdout:
    """
    A context manager to temporarily suppress stdout by redirecting it to devnull.
    stderr is not affected.
    """
    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self.original_stdout


def gui_log_output(default_log_dir="logs", per_run=True, capture_stderr=True, get_log_dir_from_args=None):
    """
    Decorator for GUI or long-running functions to duplicate stdout/stderr.

    Args:
        default_log_dir (str): Fallback directory for logs if dynamic path fails.
        per_run (bool): If True, create a new log file per call.
        capture_stderr (bool): If True, also duplicate stderr to the same log.
        get_log_dir_from_args (callable): A function that receives the decorated
                                          function's (*args, **kwargs) and returns
                                          the desired log directory path.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log_dir = default_log_dir
            
            if callable(get_log_dir_from_args):
                try:
                    dynamic_log_dir = get_log_dir_from_args(*args, **kwargs)
                    if dynamic_log_dir:
                        log_dir = dynamic_log_dir
                except Exception as e:
                    print(
                        f"Warning: Could not determine dynamic log directory. Using default '{log_dir}'. Error: {e}",
                        file=sys.__stderr__
                    )

            Path(log_dir).mkdir(exist_ok=True)

            if per_run:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                logfile = Path(log_dir) / f"{func.__name__}_{timestamp}.log"
            else:
                logfile = Path(log_dir) / f"{func.__name__}.log"

            old_stdout = sys.stdout
            old_stderr = sys.stderr

            tee_stream = Tee(logfile)
            sys.stdout = tee_stream
            if capture_stderr:
                sys.stderr = tee_stream

            print(f"--- Log started for {func.__name__} at {datetime.now()} ---")
            print(f"--- Log file: {logfile.resolve()} ---")
            try:
                result = func(*args, **kwargs)
                print(f"\n--- Finished {func.__name__} successfully at {datetime.now()} ---")
                return result
            except Exception as e:
                import traceback
                print(f"\n!!! EXCEPTION in {func.__name__}: {e}\n", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                raise
            finally:
                sys.stdout = old_stdout
                if capture_stderr:
                    sys.stderr = old_stderr
                tee_stream.log.close()
        return wrapper
    return decorator


class TerminalOnlyStdout:
    """
    A context manager to temporarily redirect stdout and stderr to the 
    terminal *only*, bypassing the Tee (log file) if it's active.
    
    This assumes stdout/stderr are Tee objects with a .terminal attribute.
    """
    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Handle stdout
        if hasattr(self.original_stdout, 'terminal'):
            sys.stdout = self.original_stdout.terminal
        
        # Handle stderr
        if hasattr(self.original_stderr, 'terminal'):
            sys.stderr = self.original_stderr.terminal

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original streams
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr