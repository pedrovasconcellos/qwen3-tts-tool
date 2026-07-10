"""CLI entry point for qwen3-tts-tool.

Provides local text-to-speech using Qwen3-TTS with a simple command surface
(synthesize, init, info, list-voices).
"""

import click

from qwen3_tts_tool import __version__
from qwen3_tts_tool.commands import (
    info_command,
    init_command,
    list_voices_command,
    synthesize,
)
from qwen3_tts_tool.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


@click.group()
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Verbose output: -v (INFO), -vv (DEBUG), -vvv (TRACE with library logs)",
)
@click.version_option(version=__version__)
def main(verbose: int) -> None:
    """Local Text-to-Speech CLI using Qwen3-TTS.

    A command-line tool for converting text to speech using the Qwen3-TTS
    CustomVoice model running locally on your machine.

    Quick Start:

    \b
        # Initialize (downloads models on first run)
        qwen3-tts-tool init

    \b
        # Basic text-to-speech
        qwen3-tts-tool synthesize "Hello, world!"

    \b
        # List available speakers
        qwen3-tts-tool list-voices

    \b
        # Show configuration
        qwen3-tts-tool info

    \b
    Examples:

        # Save to file
        qwen3-tts-tool synthesize "developer" --speed 0.8 --voice aiden \\
            --silence 800 --output audio_generate.wav

    \b
    Model Info:
        - Default model: Qwen3-TTS-12Hz-0.6B-CustomVoice
        - Runs locally (CPU by default; QWEN3_TTS_DEVICE=mps to opt into MPS)
        - Languages: English, Chinese, Japanese, Korean, German, French,
          Russian, Portuguese, Spanish, Italian
    """
    setup_logging(verbose)


main.add_command(synthesize)
main.add_command(list_voices_command, name="list-voices")
main.add_command(info_command, name="info")
main.add_command(init_command, name="init")


if __name__ == "__main__":
    main()
