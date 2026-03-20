# TTSNS — Arquitectura del Sistema

> Nuestro sistema TTS híbrido: AR para streaming + difusión para calidad.
> Documento de decisiones de diseño.

---

## Filosofía

1. **Ser críticos** — saber dónde invertir complejidad y dónde no.
2. **Clasificar antes de procesar** — el sistema debe entender qué tipo de audio recibe.
3. **Híbrido pragmático** — AR para planificación semántica, difusión para calidad acústica.
4. **Incremental** — construir pieza por pieza, validar cada componente.

---

## Arquitectura general

```
                         Texto + Audio referencia
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
          ┌─────────────────┐           ┌─────────────────┐
          │  Text Frontend   │           │  Audio Analyzer  │
          │  ──────────────  │           │  ──────────────  │
          │  • Normalización │           │  • Clasificación │
          │    de texto      │           │    de ruido      │
          │  • Detección de  │           │  • SNR estimation│
          │    idioma        │           │  • VAD           │
          │  • Tokenización  │           │  • Speaker       │
          │    (sin G2P)     │           │    diarization   │
          └────────┬────────┘           │  • Denoising     │
                   │                     │    condicional   │
                   │                     └────────┬────────┘
                   │                              │
                   │                    ┌─────────┴─────────┐
                   │                    │  Audio Tokenizer   │
                   │                    │  (DAC/codec)       │
                   │                    │  Audio → tokens    │
                   │                    └─────────┬─────────┘
                   │                              │
                   └──────────────┬───────────────┘
                                 ▼
                   ┌──────────────────────────────┐
                   │     Slow AR (LLM Backbone)    │
                   │     ═══════════════════════   │
                   │     • Modelo de lenguaje      │
                   │       preentrenado            │
                   │     • Genera tokens           │
                   │       semánticos (codebook 1) │
                   │     • Streaming nativo        │
                   │     • Planifica prosodia      │
                   └──────────────┬───────────────┘
                                 │ hidden states + tokens semánticos
                                 ▼
                   ┌──────────────────────────────┐
                   │     Buffer de chunks          │
                   │     Acumula N frames          │
                   └──────────────┬───────────────┘
                                 │ cada N frames
                                 ▼
                   ┌──────────────────────────────┐
                   │     Acoustic Decoder          │
                   │     ═════════════════         │
                   │     OPCIÓN A: Fast AR         │
                   │       (tokens → codebooks     │
                   │        2-10, como Fish Audio)  │
                   │                               │
                   │     OPCIÓN B: Mini-Difusión   │
                   │       (flow matching, 5-10    │
                   │        pasos por chunk)        │
                   │                               │
                   │     → Decisión: empezar con   │
                   │       Fast AR (más simple),   │
                   │       luego experimentar con  │
                   │       difusión                │
                   └──────────────┬───────────────┘
                                 │
                                 ▼
                   ┌──────────────────────────────┐
                   │     Vocoder / Codec Decoder   │
                   │     Tokens → Waveform         │
                   │     44.1 kHz                  │
                   └──────────────┬───────────────┘
                                 │
                                 ▼
                            Audio WAV
                         (streaming por chunks)
```

---

## Componentes y prioridades

### Fase 1: Fundamentos (AHORA)

| # | Componente | Qué hace | Complejidad | Prioridad |
|---|-----------|----------|-------------|-----------|
| 1 | **Audio Analyzer** | Clasifica ruido, estima SNR, VAD, decide si el audio es usable | Media | ALTA |
| 2 | **Audio Preprocessor** | Denoising, normalización, segmentación | Media | ALTA |
| 3 | **Audio Tokenizer** | Convierte audio limpio → tokens discretos (DAC/EnCodec) | Baja (usar existente) | ALTA |
| 4 | **Text Frontend** | Normaliza texto, detecta idioma, tokeniza | Baja | ALTA |

### Fase 2: Modelo core

| # | Componente | Qué hace | Complejidad | Prioridad |
|---|-----------|----------|-------------|-----------|
| 5 | **Slow AR** | LLM backbone que genera tokens semánticos | Alta | CRÍTICA |
| 6 | **Fast AR** | Transformer ligero para codebooks acústicos | Media | CRÍTICA |
| 7 | **Vocoder** | Tokens → waveform (DAC decoder o HiFi-GAN) | Baja (usar existente) | ALTA |

### Fase 3: Calidad y refinamiento

| # | Componente | Qué hace | Complejidad | Prioridad |
|---|-----------|----------|-------------|-----------|
| 8 | **Mini-Difusión** | Reemplaza/complementa Fast AR con flow matching | Alta | MEDIA |
| 9 | **RL Alignment** | GRPO para mejorar calidad post-training | Alta | MEDIA |
| 10 | **Speaker Encoder** | Embedding de hablante para clonación | Media | MEDIA |

---

## Decisiones de diseño

### 1. Sin G2P (grafema-a-fonema)

**Decisión**: No usar G2P. El LLM backbone entiende texto directamente.

**Por qué**:
- G2P requiere reglas por idioma (frágil, no escala)
- Los LLMs ya resuelven polisemia, homógrafos, code-switching
- Fish Audio demostró que esto funciona mejor

### 2. Clasificación de audio ANTES de todo

**Decisión**: Clasificar el ruido/calidad del audio de referencia como primer paso.

**Por qué**:
- No tiene sentido tokenizar audio ruidoso — basura entra, basura sale
- Clasificar primero permite decidir: ¿denoising agresivo? ¿rechazar? ¿aceptar?
- El clasificador guía qué nivel de preprocesamiento aplicar

```
Audio entrada
      │
      ▼
┌─────────────────────────────────────┐
│  Noise Classifier                    │
│  ────────────────                    │
│  Categorías:                         │
│  • clean (SNR > 30dB)      → pasar  │
│  • mild_noise (20-30dB)    → suave  │
│  • moderate_noise (10-20dB) → fuerte│
│  • heavy_noise (<10dB)     → alertar│
│  • music_background        → separar│
│  • multiple_speakers       → diarize│
│  • non_speech              → reject │
│                                      │
│  También detecta:                    │
│  • Reverberación                     │
│  • Clipping/distorsión              │
│  • Sample rate original             │
└─────────────────────────────────────┘
```

### 3. Empezar con Fast AR, iterar hacia difusión

**Decisión**: El acoustic decoder empieza como Fast AR (estilo Fish Audio).

**Por qué**:
- Más simple de implementar y depurar
- Streaming nativo sin buffers complicados
- Si funciona bien, la difusión es mejora opcional
- Si la calidad no es suficiente, tenemos el camino híbrido documentado

### 4. Audio tokenizer: usar DAC existente

**Decisión**: Usar Descript Audio Codec (DAC) como tokenizer.

**Por qué**:
- Open source, bien probado
- 44.1 kHz, alta calidad
- RVQ multi-codebook (necesario para Dual-AR)
- Se puede fine-tunear después si necesitamos

### 5. LLM Backbone: empezar small

**Decisión**: Empezar con un modelo pequeño (Qwen2.5-0.5B o similar) para prototipar.

**Por qué**:
- Iterar rápido en GPU consumer (8-16GB VRAM)
- Validar la arquitectura sin necesitar un cluster
- Escalar después es trivial (misma arquitectura, más params)

---

## Stack técnico

```
Lenguaje:        Python 3.11+
Framework ML:    PyTorch 2.x
Audio:           torchaudio, librosa, soundfile
Tokenizer:       DAC (descript-audio-codec)
LLM:             transformers (HuggingFace)
Training:        accelerate, deepspeed (cuando escalemos)
Serving:         SGLang (producción), uvicorn (dev)
Audio analysis:  torchaudio + modelos custom
Testing:         pytest
Config:          YAML (hydra o simple)
```

---

## Estructura del proyecto

```
ttsns/
├── doc/                    # Documentación (ya existe)
├── src/
│   └── ttsns/
│       ├── __init__.py
│       ├── audio/
│       │   ├── __init__.py
│       │   ├── analyzer.py      # Clasificación de ruido, SNR, VAD
│       │   ├── preprocessor.py  # Denoising, normalización, segmentación
│       │   └── tokenizer.py     # Audio → tokens (wrapper de DAC)
│       ├── text/
│       │   ├── __init__.py
│       │   └── frontend.py      # Normalización, detección idioma, tokenización
│       ├── model/
│       │   ├── __init__.py
│       │   ├── slow_ar.py       # LLM backbone (Slow AR)
│       │   ├── fast_ar.py       # Acoustic decoder (Fast AR)
│       │   └── hybrid.py        # Orquestador Slow AR + decoder
│       ├── vocoder/
│       │   ├── __init__.py
│       │   └── decoder.py       # Tokens → waveform
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py       # Loop de entrenamiento
│       │   └── data.py          # Dataset y dataloaders
│       ├── inference/
│       │   ├── __init__.py
│       │   └── generate.py      # Pipeline de inferencia
│       └── config/
│           └── default.yaml     # Configuración por defecto
├── tests/
│   ├── test_audio_analyzer.py
│   ├── test_preprocessor.py
│   └── test_text_frontend.py
├── scripts/
│   └── prepare_data.py
├── pyproject.toml
└── .gitignore
```

---

## Métricas objetivo (V1)

| Métrica | Objetivo V1 | SOTA (Fish Audio S2) |
|---------|-------------|----------------------|
| RTF | < 1.0 (tiempo real) | 0.195 |
| TTFA | < 500ms | < 100ms |
| WER (EN) | < 5% | 0.99% |
| Streaming | Sí (por chunks) | Sí (nativo) |
| Sample rate | 44.1 kHz | 44.1 kHz |
| Idiomas (V1) | ES + EN | 80+ |

---

## Qué NO hacer (donde NO invertir complejidad)

1. **No construir un codec custom** — usar DAC tal cual, fine-tunear solo si es necesario
2. **No G2P** — el LLM ya entiende texto
3. **No serving framework custom** — usar SGLang cuando llegue el momento
4. **No multi-speaker en V1** — una voz a la vez, multi-speaker después
5. **No RL en V1** — primero que funcione, después optimizar con GRPO
6. **No difusión en V1** — Fast AR primero, difusión como upgrade

---

## Siguiente paso

Empezar por **Audio Analyzer** — el clasificador de ruido y calidad de audio.
Es el primer componente del pipeline y nos da la base para todo el preprocesamiento.
