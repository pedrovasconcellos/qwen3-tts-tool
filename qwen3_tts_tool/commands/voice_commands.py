"""Voice commands for qwen3-tts-tool.

Provides commands for listing the available TTS speakers.
"""

import click

from qwen3_tts_tool.logging_config import get_logger
from qwen3_tts_tool.voices import list_languages, list_voices

logger = get_logger(__name__)


def format_table_row(
    voice_id: str,
    name: str,
    gender: str,
    language: str,
    description: str,
) -> str:
    """Format a row for table display with consistent column widths."""
    return f"  {voice_id:<12} {name:<12} {gender:<8} {language:<28} {description:<52}"


@click.command("list-voices")
@click.option(
    "--language",
    "-l",
    help="Filter by native language (e.g., English, Chinese, Japanese)",
)
@click.option(
    "--gender",
    "-g",
    type=click.Choice(["Male", "Female"], case_sensitive=False),
    help="Filter by gender",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output as JSON for scripting",
)
def list_voices_command(
    language: str | None,
    gender: str | None,
    as_json: bool,
) -> None:
    """List all available Qwen3-TTS speakers.

    Displays speaker information including id, name, gender, native language,
    and description. There are 9 premium CustomVoice speakers.

    Examples:

    \b
        # List all speakers
        qwen3-tts-tool list-voices

    \b
        # Filter by language
        qwen3-tts-tool list-voices --language English

    \b
        # Filter by gender
        qwen3-tts-tool list-voices --gender Male

    \b
        # JSON output for scripting
        qwen3-tts-tool list-voices --json
    """
    logger.info("Listing available Qwen3-TTS speakers")

    voices = list_voices(language=language, gender=gender)

    if not voices:
        click.echo("No speakers found matching the criteria.", err=True)
        return

    if as_json:
        import json

        click.echo(json.dumps(voices, indent=2))
        return

    click.echo("\nQwen3-TTS Speakers")
    click.echo("=" * 116)
    click.echo(format_table_row("Voice ID", "Name", "Gender", "Native Language", "Description"))
    click.echo("-" * 116)

    for voice in voices:
        click.echo(
            format_table_row(
                voice["id"],
                voice["name"],
                voice["gender"],
                voice["language"],
                voice["description"],
            )
        )

    click.echo("-" * 116)
    click.echo(f"\nTotal: {len(voices)} speakers")

    languages = list_languages()
    click.echo(f"\nSupported synthesis languages: {', '.join(languages)}")

    logger.info(f"Listed {len(voices)} speakers")
