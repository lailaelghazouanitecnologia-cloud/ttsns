# TTSNS V0.1 — Plan de Desarrollo

> TTS bilingüe (ES + EN) con Dual-AR, anotación emocional, y Mimi codec.
> Backbone: Qwen3.5-0.8B. Sin G2P. Streaming nativo.

---

## Estado actual

```
HECHO:
  ✓ doc/01-07: investigación de arquitecturas, Fish Audio, decisiones
  ✓ doc/08: pipeline de anotación (SenseVoice + Emotion2Vec + GLM-4-Voice)
  ✓ doc/09: comparativa de codecs (Mimi elegido)
  ✓ src/ttsns/audio/analyzer.py — clasificador de calidad de audio
  ✓ pyproject.toml, estructura de carpetas, config base

POR HACER:
  Todo el resto.
```

---

## Decisiones técnicas V0.1

| Decisión | Elección | Alternativa para V0.2+ |
|----------|----------|------------------------|
| Backbone LLM | Qwen3.5-0.8B-Base | Escalar a 2B, 4B, 9B |
| Codec | Mimi (96M, 24kHz, 12.5Hz) | WavTokenizer (1 codebook) o codec custom |
| Fast AR | Transformer 4 capas (~50M) | Más capas o mini-difusión |
| Anotación bulk | SenseVoice + Emotion2Vec | + GLM-4-Voice router (V0.2) |
| Sample rate | 24 kHz | 44.1 kHz (cambiar codec) |
| Idiomas | ES + EN | Más idiomas |
| G2P | No. El LLM entiende texto. | — |
| Entrenamiento | Single GPU o multi-GPU con accelerate | DeepSpeed, FSDP |
| Serving | CLI + uvicorn (dev) | SGLang (producción) |

---

## Arquitectura V0.1

```
                    Texto (ES/EN) + Audio referencia (10-30s)
                          │                    │
                          ▼                    ▼
                 ┌────────────────┐   ┌────────────────────┐
                 │ Text Tokenizer │   │  Audio Analyzer     │
                 │ (Qwen3.5 tok.) │   │  (SNR, VAD, ruido)  │
                 │ vocab: 248K    │   └─────────┬──────────┘
                 └───────┬────────┘             │
                         │                      ▼
                         │             ┌────────────────────┐
                         │             │  Audio Preprocessor │
                         │             │  (resample 24kHz,   │
                         │             │   normalize, denoise)│
                         │             └─────────┬──────────┘
                         │                       │
                         │                       ▼
                         │             ┌────────────────────┐
                         │             │  Mimi Encoder       │
                         │             │  (96M params)       │
                         │             │  Audio → 8 codebooks│
                         │             │  @ 12.5 Hz          │
                         │             │                     │
                         │             │  CB1: semántico     │
                         │             │  CB2-8: acústico    │
                         │             └─────────┬──────────┘
                         │                       │
                         │              tokens de audio
                         │              (CB1 = IDs semánticos)
                         │                       │
                         └───────────┬───────────┘
                                     │
                                     ▼
                      ┌───────────────────────────────┐
                      │  Slow AR (Qwen3.5-0.8B)       │
                      │  ════════════════════════      │
                      │                                │
                      │  Input:                        │
                      │    [text tokens] + [CB1 ref]   │
                      │                                │
                      │  Output:                       │
                      │    tokens CB1 (semánticos)     │
                      │    autoregresivos en tiempo    │
                      │                                │
                      │  800M params                   │
                      │  24 capas (18 DeltaNet + 6 GA) │
                      │  Contexto: 262K tokens         │
                      │  VRAM: ~1.6GB FP16             │
                      │                                │
                      │  Vocab extendido:              │
                      │    248K texto + 2048 audio CB1 │
                      │    = 250,368 tokens             │
                      └──────────────┬────────────────┘
                                     │
                            hidden states + CB1 tokens
                                     │
                                     ▼
                      ┌───────────────────────────────┐
                      │  Fast AR (~50M params)         │
                      │  ═════════════════════         │
                      │                                │
                      │  Por cada timestep t:          │
                      │  hidden_state[t] → CB2..CB8   │
                      │                                │
                      │  4 capas Transformer           │
                      │  Embedding compartido          │
                      │  Genera 7 codebooks acústicos  │
                      │  (secuencial en profundidad)   │
                      │                                │
                      │  VRAM: ~200MB FP16             │
                      └──────────────┬────────────────┘
                                     │
                            8 codebooks completos por frame
                                     │
                                     ▼
                      ┌───────────────────────────────┐
                      │  Mimi Decoder                  │
                      │  (parte del codec, ~48M)       │
                      │                                │
                      │  8 codebooks → waveform 24kHz  │
                      │  Streaming: sí (causal, 80ms)  │
                      └──────────────┬────────────────┘
                                     │
                                     ▼
                                Audio WAV 24kHz
                              (streaming por chunks)
```

---

## Fases de desarrollo

### Fase 0: Datos y anotación

Sin datos anotados no hay entrenamiento. Esto va primero.

```
OBJETIVO: Pipeline que toma audio crudo → produce JSON con
          transcripción + emoción + eventos + calidad

COMPONENTES:
  0.1  Audio Preprocessor
       ├── Resample a 24kHz (target de Mimi)
       ├── Normalización de loudness (LUFS)
       ├── Mono conversion
       └── Segmentación por VAD (del AudioAnalyzer existente)

  0.2  Wrapper SenseVoice
       ├── Carga FunAudioLLM/SenseVoiceSmall
       ├── Inferencia: audio → {transcripción, emoción, idioma, eventos}
       └── Batch processing

  0.3  Wrapper Emotion2Vec
       ├── Carga emotion2vec/emotion2vec_plus_large
       ├── Inferencia: audio → {emoción, confianza por clase}
       └── Batch processing

  0.4  Annotator (merge + export)
       ├── Combina resultados de SenseVoice + Emotion2Vec
       ├── Confidence check (detecta desacuerdos)
       ├── Marca para escalación si necesario
       ├── Exporta JSON en el formato de doc/08
       └── Genera estadísticas del dataset

  0.5  Script CLI: annotate
       └── python -m ttsns.annotate --input-dir data/raw --output-dir data/annotated

ARCHIVOS:
  src/ttsns/audio/preprocessor.py
  src/ttsns/annotation/sensevoice.py
  src/ttsns/annotation/emotion2vec.py
  src/ttsns/annotation/annotator.py
  src/ttsns/annotation/__init__.py
  scripts/annotate.py
  tests/test_preprocessor.py
  tests/test_annotation.py

DEPENDENCIAS NUEVAS:
  funasr (para SenseVoice)
  modelscope o torch (para Emotion2Vec)
```

---

### Fase 1: Codec (Mimi)

```
OBJETIVO: Wrapper limpio para encodear/decodear audio con Mimi.
          Verificar que el roundtrip mantiene calidad.

COMPONENTES:
  1.1  MimiCodec wrapper
       ├── Carga kyutai/mimi desde HuggingFace
       ├── encode(waveform) → tokens (8 codebooks × T frames)
       ├── decode(tokens) → waveform
       ├── encode_semantic(waveform) → tokens CB1 only
       └── Manejo de device (CPU/GPU), dtype

  1.2  Tests de roundtrip
       ├── encode → decode → comparar con original
       ├── Medir: PESQ, STOI, mel distance
       └── Verificar CB1 captura prosodia

ARCHIVOS:
  src/ttsns/codec/__init__.py
  src/ttsns/codec/mimi.py
  tests/test_codec.py

DEPENDENCIAS NUEVAS:
  moshi (incluye Mimi)
```

---

### Fase 2: Datos de entrenamiento

```
OBJETIVO: Dataset que combina anotaciones + tokens de codec
          para entrenar el Dual-AR.

COMPONENTES:
  2.1  Dataset class
       ├── Carga audio + anotación JSON
       ├── On-the-fly: audio → Mimi encode → tokens
       │   (o pre-computa tokens para velocidad)
       ├── Tokeniza texto con Qwen3.5 tokenizer
       ├── Construye secuencias de entrenamiento:
       │     [BOS] [text_tokens] [SEP] [audio_CB1_tokens] [EOS]
       └── Collate con padding

  2.2  Pre-cómputo de tokens
       ├── Script que procesa todo el dataset
       ├── Guarda tokens Mimi en disco (.pt o .npy)
       └── Evita re-encodear en cada epoch

ARCHIVOS:
  src/ttsns/training/dataset.py
  scripts/precompute_tokens.py
  tests/test_dataset.py

FORMATO DE ENTRADA:
  data/
  ├── annotated/
  │   ├── audio_00001.json    # anotación (doc 08)
  │   ├── audio_00001.wav     # audio original
  │   └── ...
  └── tokens/                 # pre-computado
      ├── audio_00001.pt      # {cb1: [...], cb2: [...], ...}
      └── ...
```

---

### Fase 3: Modelo (Slow AR + Fast AR)

```
OBJETIVO: El modelo Dual-AR que genera audio tokens desde texto.

COMPONENTES:
  3.1  Slow AR (Qwen3.5-0.8B adaptado)
       ├── Carga Qwen3.5-0.8B-Base
       ├── Extiende vocab: +2048 tokens para CB1 de Mimi
       │   (resize_token_embeddings)
       ├── Tokens especiales: <|audio_start|>, <|audio_end|>,
       │   <|text_start|>, <|text_end|>, <|sep|>
       ├── Forward: text_tokens + ref_cb1 → next CB1 token
       ├── Loss: CrossEntropy sobre CB1 tokens
       └── KV cache para inferencia

  3.2  Fast AR (~50M params)
       ├── 4-layer Transformer decoder
       ├── Input: hidden_state del Slow AR en timestep t
       ├── Output: CB2, CB3, ..., CB8 (secuencial)
       ├── Embedding compartido entre codebooks
       ├── Loss: CrossEntropy sobre CB2-8
       └── Corre una vez por timestep

  3.3  Modelo unificado (Hybrid)
       ├── Orquesta Slow AR + Fast AR
       ├── Forward conjunto para training
       │   (loss = loss_slow + λ * loss_fast)
       ├── generate() para inferencia
       └── Manejo de streaming (yield chunks)

ARCHIVOS:
  src/ttsns/model/slow_ar.py
  src/ttsns/model/fast_ar.py
  src/ttsns/model/hybrid.py
  tests/test_model.py

SECUENCIA DE ENTRENAMIENTO:
  ┌─────────────────────────────────────────────────────────────┐
  │ <|text_start|> tokens_texto <|text_end|>                   │
  │ <|sep|>                                                     │
  │ <|audio_start|> ref_CB1_tokens <|audio_end|>              │
  │ <|sep|>                                                     │
  │ <|audio_start|> target_CB1_tokens <|audio_end|>           │
  │                                                             │
  │ El modelo aprende a predecir target_CB1 dado texto + ref   │
  └─────────────────────────────────────────────────────────────┘

  Para cada target_CB1[t], el Fast AR genera CB2-8[t]
  condicionado en el hidden_state[t] del Slow AR.
```

---

### Fase 4: Entrenamiento

```
OBJETIVO: Loop de entrenamiento funcional, single o multi-GPU.

COMPONENTES:
  4.1  Trainer
       ├── HuggingFace accelerate para multi-GPU
       ├── Mixed precision (BF16)
       ├── Gradient accumulation
       ├── Learning rate schedule (cosine con warmup)
       ├── Checkpointing
       ├── Logging (tensorboard o wandb)
       └── Validación periódica

  4.2  Entrenamiento en 2 etapas:
       ETAPA A: Solo Slow AR
         ├── Congela Fast AR
         ├── Entrena predicción de CB1
         ├── LR: 1e-4, warmup 1K steps
         └── Objetivo: que el modelo aprenda la estructura

       ETAPA B: Slow AR + Fast AR conjunto
         ├── Descongela Fast AR
         ├── Loss conjunta: L_slow + 0.5 * L_fast
         ├── LR más bajo para Slow AR (1e-5)
         ├── LR normal para Fast AR (1e-4)
         └── Objetivo: calidad acústica

  4.3  Datos de entrenamiento V0.1
       ├── LibriSpeech (EN): ~960 horas, limpio
       ├── Common Voice ES: ~500+ horas
       ├── Anotado con nuestro pipeline (Fase 0)
       └── Total estimado: ~1500 horas para V0.1

ARCHIVOS:
  src/ttsns/training/trainer.py
  src/ttsns/training/config.py
  scripts/train.py
  configs/train_v01.yaml

HIPERPARÁMETROS V0.1:
  batch_size: 16 (por GPU)
  gradient_accumulation: 4
  effective_batch: 64
  max_audio_length: 30s (375 tokens CB1 @ 12.5Hz)
  max_text_length: 512 tokens
  total_steps: 200K (etapa A: 100K, etapa B: 100K)
  optimizer: AdamW (β1=0.9, β2=0.95)
  weight_decay: 0.1
  warmup: 2K steps
  lr_slow: 1e-4 → 1e-5
  lr_fast: 1e-4
  precision: BF16
```

---

### Fase 5: Inferencia

```
OBJETIVO: Pipeline end-to-end: texto + audio ref → audio generado.

COMPONENTES:
  5.1  Generator
       ├── Carga modelo entrenado (checkpoint)
       ├── Carga Mimi codec
       ├── Pipeline:
       │     texto → tokenize
       │     audio_ref → Mimi encode → CB1 ref
       │     [text + CB1_ref] → Slow AR → CB1 gen
       │     CB1 gen → Fast AR → CB2-8 gen
       │     [CB1-8] → Mimi decode → audio 24kHz
       ├── Streaming: yield chunks cada N frames
       ├── Sampling: temperature, top_p, top_k
       └── Guardar WAV

  5.2  CLI
       ├── python -m ttsns.generate \
       │     --text "Hola, ¿cómo estás?" \
       │     --ref audio_ref.wav \
       │     --output output.wav \
       │     --language es
       └── Flags: --temperature, --stream, --device

  5.3  Server básico (uvicorn)
       ├── POST /v1/tts
       │   body: {text, ref_audio_base64, language}
       │   response: audio/wav (streaming)
       └── Para demo/testing

ARCHIVOS:
  src/ttsns/inference/generator.py
  src/ttsns/inference/server.py
  scripts/generate.py
```

---

## Estructura final del proyecto V0.1

```
ttsns/
├── doc/                          # Documentación (10 docs)
├── src/
│   └── ttsns/
│       ├── __init__.py
│       ├── audio/
│       │   ├── __init__.py
│       │   ├── analyzer.py       # ✓ HECHO — SNR, VAD, calidad
│       │   └── preprocessor.py   # Fase 0.1 — resample, normalize
│       ├── annotation/
│       │   ├── __init__.py
│       │   ├── sensevoice.py     # Fase 0.2 — ASR + emoción
│       │   ├── emotion2vec.py    # Fase 0.3 — emoción fina
│       │   └── annotator.py      # Fase 0.4 — merge + export
│       ├── codec/
│       │   ├── __init__.py
│       │   └── mimi.py           # Fase 1 — encode/decode
│       ├── text/
│       │   ├── __init__.py
│       │   └── frontend.py       # Fase 3 — normalización texto
│       ├── model/
│       │   ├── __init__.py
│       │   ├── slow_ar.py        # Fase 3.1 — Qwen3.5-0.8B
│       │   ├── fast_ar.py        # Fase 3.2 — Transformer 4L
│       │   └── hybrid.py         # Fase 3.3 — orquestador
│       ├── training/
│       │   ├── __init__.py
│       │   ├── dataset.py        # Fase 2 — datos
│       │   ├── trainer.py        # Fase 4 — loop
│       │   └── config.py         # Fase 4 — config training
│       └── inference/
│           ├── __init__.py
│           ├── generator.py      # Fase 5.1 — pipeline
│           └── server.py         # Fase 5.3 — API básica
├── scripts/
│   ├── annotate.py               # Fase 0.5 — CLI anotación
│   ├── precompute_tokens.py      # Fase 2.2 — pre-cómputo
│   ├── train.py                  # Fase 4 — CLI training
│   └── generate.py               # Fase 5.2 — CLI inferencia
├── configs/
│   ├── default.yaml              # Config base
│   └── train_v01.yaml            # Config entrenamiento V0.1
├── tests/
│   ├── test_audio_analyzer.py
│   ├── test_preprocessor.py
│   ├── test_annotation.py
│   ├── test_codec.py
│   ├── test_dataset.py
│   └── test_model.py
├── pyproject.toml
└── .gitignore
```

---

## Dependencias V0.1

```toml
[project]
dependencies = [
    # Core
    "torch>=2.1",
    "torchaudio>=2.1",
    "numpy>=1.24",
    "pyyaml>=6.0",
    # Audio
    "librosa>=0.10",
    "soundfile>=0.12",
    # LLM
    "transformers>=4.48",
    "accelerate>=0.30",
    # Codec
    "moshi",
    # Annotation
    "funasr",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "ruff>=0.1"]
train = ["datasets>=2.16", "tensorboard>=2.15", "wandb"]
serve = ["uvicorn>=0.27", "fastapi>=0.110"]
```

---

## Métricas objetivo V0.1

| Métrica | Objetivo V0.1 | SOTA (Fish S2) | Notas |
|---------|---------------|----------------|-------|
| WER (EN) | < 10% | 0.99% | Inteligibilidad básica |
| WER (ES) | < 12% | — | Menos datos disponibles |
| RTF | < 2.0 | 0.195 | No necesita ser rápido en V0.1 |
| MOS | > 3.0/5 | ~4.4/5 | Calidad aceptable, no perfecta |
| Streaming | Sí (por chunks) | Sí (nativo) | Latencia no crítica |
| Sample rate | 24 kHz | 44.1 kHz | Suficiente para V0.1 |
| Clonación de voz | Básica (10-30s ref) | Avanzada | Zero-shot con ref |

---

## Orden de ejecución

```
SEMANA 1-2: Fase 0 (Anotación)
  → Preprocessor + SenseVoice + Emotion2Vec + Annotator
  → Empezar a procesar LibriSpeech + CommonVoice

SEMANA 3: Fase 1 (Codec Mimi)
  → Wrapper encode/decode
  → Tests de roundtrip

SEMANA 4: Fase 2 (Dataset)
  → Dataset class + pre-cómputo de tokens
  → Verificar que el dataloader funciona

SEMANA 5-6: Fase 3 (Modelo)
  → Slow AR (Qwen3.5-0.8B con vocab extendido)
  → Fast AR (4-layer transformer)
  → Hybrid (forward conjunto)

SEMANA 7-10: Fase 4 (Entrenamiento)
  → Etapa A: Solo Slow AR, 100K steps
  → Etapa B: Slow + Fast AR, 100K steps
  → Iterar según resultados

SEMANA 11-12: Fase 5 (Inferencia)
  → Pipeline end-to-end
  → CLI + server básico
  → Primeras muestras de audio

Total estimado: ~3 meses para V0.1 funcional
```

---

## Riesgos y mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|---------|------------|
| Mimi 24kHz calidad insuficiente para ES | Media | Alto | Probar SNAC 44kHz como fallback |
| Qwen3.5-0.8B demasiado pequeño para TTS | Media | Alto | Escalar a 2B/4B (misma familia) |
| SenseVoice malo en español | Baja | Medio | Whisper-v3 como fallback para ASR |
| Datos insuficientes en español | Media | Alto | Añadir Mozilla Common Voice, Voxpopuli |
| Fast AR no converge | Baja | Alto | Empezar sin Fast AR (solo CB1 + Mimi) |
| OOM en training | Baja | Bajo | Gradient checkpointing, reduce batch |

---

## Qué NO hacer en V0.1

1. No RL / GRPO — primero que funcione
2. No multi-speaker — una voz a la vez
3. No difusión — Fast AR es suficiente
4. No codec custom — Mimi tal cual
5. No serving en producción — CLI y dev server
6. No router aLLM (GLM-4-Voice) — solo pipeline ligero de anotación
7. No fine-tune del codec — usar Mimi pre-entrenado
8. No 44.1kHz — 24kHz con Mimi
