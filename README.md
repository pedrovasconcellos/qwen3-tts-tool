# qwen3-tts-tool

A CLI that provides local text-to-speech using [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS).
It exposes a simple command surface (`synthesize`, `init`, `info`,
`list-voices`) so you can use it straight from your terminal:

#### Developed by Pedro Vasconcellos in 2026

```bash
qwen3-tts-tool synthesize "term" --speed 0.8 --voice aiden --silence 800 --output generated_audio.wav
```

## Runtime notes

Qwen3-TTS is a 0.6B/1.7B model, so:

- The first run downloads several GB of weights (tokenizer + model).
- Generation on **macOS CPU is slow**. This is expected for a model this size.
- The model is loaded on **CPU by default** with `float32` and `sdpa`
  attention. Apple GPU (MPS) is opt-in and experimental (see below); the
  CUDA + FlashAttention 2 path from the upstream docs does not apply on a Mac.

## Requirements

- Python 3.12 (Qwen3-TTS recommends a clean 3.12 environment).
- [`ffmpeg`](https://ffmpeg.org/) — only required when using `--speed` other
  than `1.0` (speed is applied as a pitch-preserving `atempo` post-process).
  Install on macOS with `brew install ffmpeg`.

## Installation

Using [`uv`](https://docs.astral.sh/uv/) (recommended):

```bash
uv tool install ./qwen3-tts-tool
uv tool update-shell   # then reopen the terminal if needed
```

Or with pip, in an isolated Python 3.12 environment:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ./qwen3-tts-tool
```

Then download the model weights once:

```bash
qwen3-tts-tool init
```

## Usage

```bash
# Play text with the default speaker (aiden)
qwen3-tts-tool synthesize "Hello world"

# Save to a WAV file
qwen3-tts-tool synthesize "developer" --speed 0.8 --voice aiden \
    --silence 800 --output generated_audio.wav

# Read from stdin
echo "Hello world" | qwen3-tts-tool synthesize --stdin

# List speakers / show config
qwen3-tts-tool list-voices
qwen3-tts-tool info
```

### `synthesize` options

| Option | Default | Meaning |
|--------|---------|---------|
| `--voice`, `-v` | `aiden` | Speaker id (see `list-voices`) |
| `--language`, `-l` | `English` | Synthesis language, or `Auto` |
| `--instruct`, `-i` | none | Natural-language style instruction, e.g. `"Very happy."` |
| `--speed` | `1.0` | Speech speed `0.5`-`2.0` (applied via `ffmpeg atempo`) |
| `--silence` | `500` | Trailing silence in ms so short terms are not cut off |
| `--output`, `-o` | none | Save to a `.wav` file instead of playing |
| `--stdin`, `-s` | off | Read text from stdin |

### Speakers

There are 9 CustomVoice speakers. The English ones are `aiden` (default,
sunny American male) and `ryan` (dynamic male). Run `qwen3-tts-tool list-voices`
for the full list, including Chinese, Japanese, and Korean speakers.

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `QWEN3_TTS_MODEL` | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | Hugging Face model id to load (e.g. the 1.7B variant for higher quality) |
| `QWEN3_TTS_DEVICE` | `cpu` | Set to `mps` to try Apple GPU acceleration (experimental; falls back to CPU if unavailable) |

## Notes on `--speed`

Qwen3-TTS has no native speed parameter. For file output, `--speed` is
applied as a pitch-preserving `ffmpeg atempo` post-process (valid range
`0.5`-`2.0`). For live playback, speed is approximated by adjusting the
playback sample rate (which shifts pitch slightly).

## Development

```bash
pip install -e ".[dev]"
pytest
```

The unit tests mock the model, so they run without downloading any weights.

## Sponsor

[![Vasconcellos Solutions](https://vasconcellos.solutions/assets/open-source/images/company/vasconcellos-solutions-small-icon.jpg)](https://vasconcellos.solutions)

### Vasconcellos IT Solutions
