"""Info command for qwen3-tts-tool.

Displays configuration status and system information.
"""

import click

from qwen3_tts_tool.engine import select_device_and_dtype
from qwen3_tts_tool.logging_config import get_logger
from qwen3_tts_tool.models import get_model_info
from qwen3_tts_tool.voices import DEFAULT_LANGUAGE, DEFAULT_VOICE, list_languages

logger = get_logger(__name__)


@click.command("info")
def info_command() -> None:
    """Display configuration and model status.

    Shows information about the Qwen3-TTS installation including model
    download status, configured device, and default options.

    Examples:

    \b
        # Show status
        qwen3-tts-tool info
    """
    logger.info("Displaying configuration info")

    info = get_model_info()
    device, dtype_name = select_device_and_dtype()

    click.echo("\nQwen3-TTS Tool - Configuration")
    click.echo("=" * 50)

    click.echo("\nModel Status:")
    if info["ready"]:
        click.echo("  Status: Ready")
    else:
        click.echo("  Status: Not downloaded")
        click.echo("  Run 'qwen3-tts-tool init' to download models")
    click.echo(f"  Tokenizer: {info['tokenizer_id']}")
    click.echo(f"  Model: {info['model_id']}")

    if info.get("model_path"):
        click.echo("\nCache Locations:")
        click.echo(f"  Tokenizer: {info.get('tokenizer_path', 'N/A')}")
        click.echo(f"  Model: {info['model_path']}")

    click.echo("\nConfig Directory:")
    click.echo(f"  {info['config_dir']}")

    click.echo("\nRuntime:")
    click.echo(f"  Device: {device} (dtype {dtype_name}, attn sdpa)")
    click.echo("  Override device with QWEN3_TTS_DEVICE=mps (experimental)")
    click.echo("  Override model with QWEN3_TTS_MODEL=<hf-model-id>")

    click.echo("\nDefaults:")
    click.echo(f"  Voice: {DEFAULT_VOICE}")
    click.echo(f"  Language: {DEFAULT_LANGUAGE}")
    click.echo("  Speed: 1.0x")
    click.echo("  Output: Speakers (or WAV file)")

    languages = list_languages()
    click.echo(f"\nSupported Languages ({len(languages)}):")
    click.echo(f"  {', '.join(languages)}")

    click.echo("\nQuick Start:")
    click.echo('  qwen3-tts-tool synthesize "Hello world"')
    click.echo("  qwen3-tts-tool list-voices")

    logger.info("Info display complete")
