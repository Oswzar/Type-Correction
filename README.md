# Type Correction

A lightweight real-time text correction tool for macOS. Select any text in any app, press a hotkey, and get instant typo corrections powered by DistilBERT.

## Quick Start

```bash
# 1. Setup
bash setup.sh
source .venv/bin/activate

# 2. Run
python run.py
```

**Usage:**
1. Select text in any application (TextEdit, VS Code, Browser, etc.)
2. Press `Alt + Shift`
3. Corrected text replaces the original automatically

## Requirements

- macOS
- Python 3.10+
- ~8GB RAM (model runs on CPU/MPS)

## First Run Setup

The first time you run the tool, macOS will ask for permissions:

1. **System Settings → Privacy & Security → Input Monitoring** → Enable terminal
2. **System Settings → Privacy & Security → Accessibility** → Enable terminal

If text capture doesn't work, grant these permissions and restart the app.

## Changing the Hotkey

Edit `run.py`, find this line and change the shortcut:

```python
listener = MacTextSelectionListener(logger=logger, shortcut="<alt>+<shift>")
```

**Format:** Use `<cmd>`, `<ctrl>`, `<alt>`, `<shift>` combined with letters.

**Examples:**
```python
shortcut="<cmd>+<shift>+x"      # Cmd+Shift+X
shortcut="<ctrl>+<alt>+c"       # Ctrl+Alt+C  
shortcut="<cmd>+d"              # Cmd+D
```

If the shortcut conflicts with another app, choose a different combination.

## How It Works

1. **Listener** (`src/listener/hotkey_listener.py`): Monitors global keyboard shortcut
2. **Capture**: Simulates Cmd+C to copy selected text
3. **Correction** (`src/model/bert_corrector.py`): Uses DistilBERT masked-language model to predict corrections based on context
4. **Replace**: Pastes corrected text back via Cmd+V

## Customizing the Typo Dictionary

Edit `src/model/bert_corrector.py` to add your own common typos:

```python
self.common_typos = {
    "helo": "hello",
    "worl": "world",
    "teh": "the",
    # Add your own:
    "youre": "you're",
    "dont": "don't",
}
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No selectable text captured" | Make sure text is highlighted before pressing hotkey |
| "Clipboard unchanged after Cmd+C" | Check Accessibility permissions in System Settings |
| Model too slow | First run downloads model (~250MB), subsequent runs are faster |
| Out of memory | Tool automatically falls back to simple typo-map mode |

## Testing

```bash
python test.py
# Expected: input "helo worl" → output "hello world" → TEST PASSED
```

## Stopping the App

Press `Ctrl + C` in the terminal running the app.

## Tech Details

- **Model:** DistilBERT (distilbert-base-uncased)
- **Auto-download:** Model downloads from HuggingFace on first run
- **Memory optimization:** batch_size=1, torch.no_grad(), float16 on Apple Silicon MPS
- **Fallback:** If model fails to load, falls back to simple typo-map correction
