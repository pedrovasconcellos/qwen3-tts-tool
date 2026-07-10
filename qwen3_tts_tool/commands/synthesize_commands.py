"""Text-to-speech synthesis CLI command.

Implements the primary TTS functionality, handling user input (text or
stdin), speaker/language selection, speed and silence controls, and output
routing (speakers or WAV file).
"""

import sys
from pathlib import Path

import click

from qwen3_tts_tool.engine import (
    DEFAULT_SILENCE_MS,
    read_from_stdin,
    synthesize_and_play,
    synthesize_to_file,
)
from qwen3_tts_tool.logging_config import get_logger
from qwen3_tts_tool.voices import (
    DEFAULT_LANGUAGE,
    DEFAULT_VOICE,
    validate_language,
    validate_voice,
)

logger = get_logger(__name__)


@click.command()
@click.argument("text", required=False)
@click.option("--stdin", "-s", is_flag=True, help="Read text from stdin instead of argument")
@click.option(
    "--voice",
    "-v",
    default=DEFAULT_VOICE,
    help=f"Speaker to use (default: {DEFAULT_VOICE})",
)
@click.option(
    "--language",
    "-l",
    default=DEFAULT_LANGUAGE,
    help=f"Synthesis language, or 'Auto' (default: {DEFAULT_LANGUAGE})",
)
@click.option(
    "--instruct",
    "-i",
    default=None,
    help="Optional natural-language style instruction (e.g. 'Very happy.')",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Save audio to file instead of playing through speakers (WAV format)",
)
@click.option(
    "--speed",
    type=click.FloatRange(0.5, 2.0),
    default=1.0,
    help="Speech speed from 0.5 (slow) to 2.0 (fast) (default: 1.0)",
)
@click.option(
    "--silence",
    type=click.IntRange(0, 5000),
    default=DEFAULT_SILENCE_MS,
    help=f"Trailing silence in milliseconds to avoid audio cutoff (default: {DEFAULT_SILENCE_MS})",
)
def synthesize(
    text: str | None,
    stdin: bool,
    voice: str,
    language: str,
    instruct: str | None,
    output: Path | None,
    speed: float,
    silence: int,
) -> None:
    """Convert text to speech using local Qwen3-TTS.

    Synthesizes text using the Qwen3-TTS CustomVoice model running locally.
    Audio can be played through the speakers or saved to a WAV file.

    The model weights are downloaded automatically on first use (or run
    'qwen3-tts-tool init' beforehand).

    Examples:

    \b
        # Play text with the default speaker (aiden)
        qwen3-tts-tool synthesize "Hello world"

    \b
        # Save to a file
        qwen3-tts-tool synthesize "developer" --speed 0.8 --voice aiden \\
            --silence 800 --output audio_generate.wav

    \b
        # Read from stdin
        echo "Hello world" | qwen3-tts-tool synthesize --stdin

    \b
    Available Speakers:
        Use 'qwen3-tts-tool list-voices' to see all 9 speakers.
        Examples: aiden, ryan (English)
    """
    try:
        if stdin:
            logger.debug("Reading text from stdin")
            input_text = read_from_stdin()
        elif text:
            input_text = text
        else:
            logger.error("No text provided")
            click.echo(
                "Error: No text provided.\n\n"
                "Provide text as argument or use --stdin:\n"
                "  qwen3-tts-tool synthesize 'your text'\n"
                "  echo 'text' | qwen3-tts-tool synthesize --stdin\n\n"
                "Use --help for more examples.",
                err=True,
            )
            sys.exit(1)

        if not input_text.strip():
            click.echo(
                "Error: Empty text provided.\n\nProvide non-empty text to synthesize.",
                err=True,
            )
            sys.exit(1)

        logger.debug(f"Validating voice: {voice}")
        speaker = validate_voice(voice)
        logger.debug(f"Validating language: {language}")
        validated_language = validate_language(language)

        truncated = input_text[:50] + "..." if len(input_text) > 50 else input_text
        click.echo(f"Synthesizing: {truncated}", err=True)
        click.echo(
            f"Speaker: {speaker}, Language: {validated_language}, Speed: {speed}x",
            err=True,
        )

        if output:
            if not str(output).lower().endswith(".wav"):
                click.echo(
                    f"Error: Output file must have .wav extension. Got: {output}\n\n"
                    "What to do:\n"
                    "  Change output to: --output speech.wav",
                    err=True,
                )
                sys.exit(1)

            logger.info(f"Synthesizing audio to file: {output}")
            char_count = synthesize_to_file(
                input_text,
                output,
                speaker,
                validated_language,
                speed,
                silence,
                instruct,
            )
            click.echo(f"\nAudio saved to: {output}", err=True)
        else:
            logger.info("Synthesizing audio for playback")
            char_count = synthesize_and_play(
                input_text,
                speaker,
                validated_language,
                speed,
                silence,
                instruct,
            )
            click.echo("\nPlayback complete!", err=True)

        click.echo(f"Processed {char_count} characters", err=True)

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        logger.debug("Full traceback:", exc_info=True)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        logger.debug("Full traceback:", exc_info=True)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.debug("Full traceback:", exc_info=True)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
