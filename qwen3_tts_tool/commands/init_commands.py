"""Init command for qwen3-tts-tool.

Handles pre-downloading the tokenizer and CustomVoice model weights.
"""

import sys

import click

from qwen3_tts_tool.logging_config import get_logger
from qwen3_tts_tool.models import download_models, get_model_info, models_exist

logger = get_logger(__name__)


@click.command("init")
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Re-download models even if they already exist",
)
def init_command(force: bool) -> None:
    """Download and initialize the Qwen3-TTS models.

    Downloads the Qwen3-TTS speech tokenizer and the configured CustomVoice
    model into the Hugging Face cache. The default model is the lightweight
    0.6B CustomVoice checkpoint; override it with the QWEN3_TTS_MODEL env var.

    Examples:

    \b
        # Download models (skips if already present)
        qwen3-tts-tool init

    \b
        # Force re-download
        qwen3-tts-tool init --force
    """
    logger.info("Initializing Qwen3-TTS models")

    info = get_model_info()

    if models_exist() and not force:
        click.echo("\nModels already downloaded!")
        click.echo(f"  Tokenizer: {info['tokenizer_id']}")
        click.echo(f"  Model: {info['model_id']}")
        click.echo("\nUse --force to re-download.")
        return

    click.echo("\nDownloading Qwen3-TTS models...")
    click.echo(f"  Tokenizer: {info['tokenizer_id']}")
    click.echo(f"  Model: {info['model_id']}")
    click.echo("This may take a while and download several GB on first run.\n")

    try:
        tokenizer_path, model_path = download_models(force=force)
        click.echo("\nModels downloaded successfully!")
        click.echo(f"  Tokenizer: {tokenizer_path}")
        click.echo(f"  Model: {model_path}")
        click.echo("\nYou're ready to go!")
        click.echo('  qwen3-tts-tool synthesize "Hello world"')
    except RuntimeError as e:
        logger.error(f"Download failed: {e}")
        click.echo(f"\nError: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.debug("Full traceback:", exc_info=True)
        click.echo(f"\nError: {e}", err=True)
        sys.exit(1)
