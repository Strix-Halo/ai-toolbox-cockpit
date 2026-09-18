"""Foreground server lifecycle shared by backend server panels."""

import shlex
import signal
import subprocess
from contextlib import nullcontext

from .isolated_api import IsolatedAPIRelay
from .terminal import command_failed, pause_after_failure


def redact_command(
    command: list[str],
    secret_options: tuple[str, ...] = ("--api-key",),
    secret_environment: tuple[str, ...] = ("HF_TOKEN",),
) -> list[str]:
    """Redact every separate or equals-form value for sensitive CLI options."""
    redacted = list(command)
    index = 0
    while index < len(redacted):
        argument = redacted[index]
        if argument in secret_options and index + 1 < len(redacted):
            redacted[index + 1] = "<redacted>"
            index += 2
            continue
        for option in secret_options:
            if argument.startswith(f"{option}="):
                redacted[index] = f"{option}=<redacted>"
                break
        for variable in secret_environment:
            if argument.startswith(f"{variable}="):
                redacted[index] = f"{variable}=<redacted>"
                break
        index += 1
    return redacted


def run_foreground_server(
    command: list[str],
    engine: str,
    container_name: str,
    *,
    display_command: list[str] | None = None,
    isolated_api: tuple[str, int] | None = None,
) -> int:
    """Run a server until exit/Ctrl+C and always remove its named container."""
    print(f"\nStarting server:\n{shlex.join(display_command or command)}\n")
    print(
        "Press Ctrl+C to stop the server and return to the cockpit. "
        "If startup fails, press Enter after reviewing the error.\n"
    )
    try:
        subprocess.run([engine, "rm", "-f", container_name], capture_output=True)
    except OSError as error:
        pause_after_failure(f"Could not prepare the server command: {error}")
        return 127
    old_handler = signal.signal(signal.SIGINT, signal.default_int_handler)
    process: subprocess.Popen | None = None
    try:
        relay = (IsolatedAPIRelay(engine, container_name, *isolated_api)
                 if isolated_api is not None else nullcontext())
        with relay:
            if isolated_api is not None:
                print(f"Isolated API relay: {isolated_api[0]}:{isolated_api[1]} "
                      "-> container loopback (no container network).\n")
            process = subprocess.Popen(command)
            return_code = process.wait()
        if command_failed(return_code):
            pause_after_failure(f"Server exited with status {return_code}.")
        return return_code
    except OSError as error:
        pause_after_failure(f"Could not start the server command: {error}")
        return 127
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        subprocess.run([engine, "rm", "-f", container_name], capture_output=True)
        if process is not None:
            process.kill()
            process.wait()
        return 130
    finally:
        try:
            if isolated_api is not None:
                # Also remove the isolated server on ordinary exit or relay failure.
                subprocess.run([engine, "rm", "-f", container_name], capture_output=True)
        except OSError as error:
            print(f"Could not remove the server container: {error}")
        finally:
            signal.signal(signal.SIGINT, old_handler)
