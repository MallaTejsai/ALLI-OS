# ALLI OS

## Personal AI Operating System

### Owner
Tejsai (Binnu)

## Current Version

v0.6 - Complete Memory Management

## Memory System

- `memory/conversation.json` — conversation history only
- `memory/long_term_memory.json` — durable personal facts
- `src/memory/` — Python-owned memory engine

### Architecture

1. WRITE: extract fact → normalize → duplicate/contradiction check → ADD/UPDATE/UNCHANGED
2. History: replaced facts become `status=inactive`; current facts stay `active`
3. READ: semantic intent → category-aware retrieval over active memories → grounded answer
4. General chat: Ollama/Qwen without forcing personal memories

### Running

```bash
ollama pull qwen2.5:7b
ollama pull all-minilm
cd ALLI-OS/src
python main.py
```

### Multi-Terminal Launch

ALLI runs from a single project — no separate copies. Endpoint is auto-discovered:

**Windows PowerShell** (Ollama runs natively on Windows):
```powershell
cd E:\AI - projects\ALLI-OS\src
python main.py
```

**WSL Ubuntu** (Ollama runs on Windows host — auto-detected):
```bash
cd /mnt/e/AI\ -\ projects/ALLI-OS/src
python3 main.py
```

**VS Code terminal** — works automatically with either PowerShell or WSL backend.

Override endpoint if needed:
```bash
export OLLAMA_HOST=http://custom-host:11434
```

### Ollama Auto-Start (Windows)

For ALLI to work after a normal PC restart, Ollama must start automatically on Windows.
Run this **once** in an **elevated PowerShell** (Run as Administrator):

```powershell
New-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Run" `
  -Name "Ollama" `
  -Value '"C:\Users\tejsa\AppData\Local\Programs\Ollama\ollama.exe" serve' `
  -PropertyType String -Force
```

### Tests

```bash
cd ALLI-OS
python tests/test_memory_manager.py
```

Debug:

```bash
set ALLI_MEMORY_DEBUG=1
python main.py
```

## Mission

Build a powerful local AI assistant that grows step by step and remains useful for years.
