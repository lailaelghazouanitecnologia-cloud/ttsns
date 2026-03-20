# Fish Audio — Análisis Completo

> Fish Audio es actualmente el sistema TTS más avanzado en calidad, velocidad y capacidades.
> Este documento cubre toda su evolución, arquitectura y detalles técnicos.

---

## Índice

1. [Visión general y evolución](#1-visión-general-y-evolución)
2. [Arquitectura Dual-AR](#2-arquitectura-dual-ar)
3. [Audio Tokenizer / Codec](#3-audio-tokenizer--codec)
4. [Firefly-GAN Vocoder (v1.x)](#4-firefly-gan-vocoder-v1x)
5. [Pipeline de datos](#5-pipeline-de-datos)
6. [Entrenamiento y alineación con RL](#6-entrenamiento-y-alineación-con-rl)
7. [Inferencia y optimización](#7-inferencia-y-optimización)
8. [Control de prosodia y emociones](#8-control-de-prosodia-y-emociones)
9. [Multi-speaker y clonación de voz](#9-multi-speaker-y-clonación-de-voz)
10. [Benchmarks y resultados](#10-benchmarks-y-resultados)
11. [Comparativa con otros modelos](#11-comparativa-con-otros-modelos)
12. [API, SDK y herramientas](#12-api-sdk-y-herramientas)
13. [Referencias](#13-referencias)

---

## 1. Visión general y evolución

| Versión | Fecha | Hito principal |
|---------|-------|----------------|
| Fish Speech v1.0 | 2024 Abr | Primera release open-source |
| Fish Speech v1.2 | 2024 Jul | Auto-reranking, bilingüe, cuantización, Faster Whisper |
| Fish Speech v1.4 | 2024 Sep | Dual-AR + GFSQ + Firefly-GAN. 720K horas. Paper: arXiv:2411.01156 |
| Fish Speech v1.5 | 2024 Q4 | 1M+ horas, ELO 1339 en TTS Arena, WER 3.5%, ONNX export |
| Fish Speech v1.6 | 2025 Mar | Mayor estabilidad, soporte de emociones, mejora multilingüe |
| OpenAudio S1 | 2025 | Rebrand. 4B params. #1 en TTS-Arena2. RLHF online. 48+ emociones |
| OpenAudio S1-mini | 2025 | 0.5B params, open-source (WER 0.011, CER 0.005) |
| **Fish Audio S2** | 2025 | Qwen3-4B backbone, 10M+ horas, RL con GRPO |
| **Fish Audio S2 Pro** | 2026 | 4B params, SOTA en todos los benchmarks. Paper: arXiv:2603.08823 |

**Filosofía de diseño:**
- Sin dependencia de G2P (grafema-a-fonema) — el modelo entiende texto nativamente como un LLM.
- TTS como problema de language modeling, no como pipeline acústico tradicional.
- Inferencia nativa con optimizaciones de LLM (KV cache, batching, CUDA graphs).

---

## 2. Arquitectura Dual-AR

El corazón de Fish Audio es su arquitectura **Dual-AR (Dual Autoregressive)**, que separa la generación temporal (semántica) de la generación en profundidad (acústica).

### Diagrama completo

```
Texto + [instrucciones de prosodia] + Audio de referencia (10-30s)
       │                                        │
       ▼                                        ▼
┌──────────────┐                      ┌────────────────────┐
│ Tokenizer    │                      │ Audio Tokenizer    │
│ de texto     │                      │ (DAC custom)       │
│ (Qwen3)      │                      │ 10 RVQ codebooks   │
└──────┬───────┘                      │ ~21 Hz frame rate  │
       │                              └────────┬───────────┘
       │                                       │
       │    ┌──────────────────────────────┐    │
       └───►│                              │◄───┘
            │   SLOW AR (Eje temporal)     │
            │   ─────────────────────────  │
            │   • Qwen3-4B (pretrained)    │
            │   • Autoregresivo en tiempo  │
            │   • Predice tokens del       │
            │     1er codebook (semántico) │
            │   • Planifica estructura     │
            │     lingüística y prosodia   │
            │   • 4B parámetros            │
            │                              │
            └──────────────┬───────────────┘
                           │
                  Hidden states por timestep
                           │
                           ▼
            ┌──────────────────────────────┐
            │                              │
            │   FAST AR (Eje de profund.)  │
            │   ─────────────────────────  │
            │   • 4-layer Transformer      │
            │   • Genera codebooks 2-10    │
            │     en cada timestep         │
            │   • Condicionado en hidden   │
            │     states del Slow AR       │
            │   • Embedding compartido     │
            │     entre capas de codebook  │
            │   • 400M parámetros          │
            │                              │
            └──────────────┬───────────────┘
                           │
                 10 codebooks por frame
                           │
                           ▼
            ┌──────────────────────────────┐
            │   Audio Codec Decoder        │
            │   (DAC modificado)           │
            │   Tokens → Waveform 44.1kHz  │
            └──────────────┬───────────────┘
                           │
                           ▼
                      Audio WAV
```

### ¿Por qué Dual-AR?

| Aspecto | Single AR (como VALL-E) | Dual-AR (Fish Audio) |
|---------|------------------------|----------------------|
| Velocidad | Genera todos los codebooks secuencialmente → lento | Slow AR en tiempo + Fast AR en profundidad → eficiente |
| Estabilidad | Prone a errores acumulativos | Slow AR planifica semántica estable |
| Uso de codebook | Codebook collapse parcial | 100% utilización de codebooks (GFSQ) |
| Parámetros | Distribuidos uniformemente | Asimétricos: 4B (tiempo) + 400M (profundidad) |

### Diferencia con otros enfoques

```
VALL-E:     [AR sobre codebook 1] → [NAR paralelo codebooks 2-8]
                                     ↑ No autoregresivo, pierde detalle

Bark:       [GPT semántico] → [GPT grueso] → [GPT fino]
             ↑ 3 modelos separados, lento, sin compartir contexto

Fish Audio: [Slow AR: codebook 1 en tiempo] → [Fast AR: codebooks 2-10 en profundidad]
             ↑ Un solo forward pass integrado, asimétrico y eficiente
```

---

## 3. Audio Tokenizer / Codec

### Versión v1.x: GFSQ (Grouped Finite Scalar Vector Quantization)

- **Innovación**: Cuantización escalar agrupada que logra **100% de utilización del codebook**.
- Los codecs tradicionales (como EnCodec con RVQ) sufren de "codebook collapse" donde muchos códigos nunca se usan.
- GFSQ agrupa dimensiones del vector latente y cuantiza cada grupo independientemente.

### Versión S2: DAC Custom (Descript Audio Codec modificado)

```
Audio 44.1 kHz
      │
      ▼
┌─────────────────────────────────────────┐
│  Encoder (Causal Sliding-Window Transf.)│
│  ───────────────────────────────────────│
│  Downsampling total: 2048x              │
│  → ~21 Hz frame rate                    │
│  → 446M parámetros                      │
│                                         │
│  Salida: 10 RVQ codebooks              │
│                                         │
│  Codebook 1: SEMÁNTICO                  │
│    • Entrenado con destilación de       │
│      w2v-BERT 2.0 (regresión)          │
│    • Captura info lingüística/fonética  │
│                                         │
│  Codebooks 2-10: ACÚSTICOS             │
│    • Detalles de timbre, textura,       │
│      micro-prosodia                     │
│                                         │
│  Entrenamiento: 1M steps               │
│  Loss: Composite GAN loss              │
└─────────────────────────────────────────┘
```

**Destilación semántica**: El primer codebook se entrena para regresionar las activaciones de w2v-BERT 2.0, asegurando que capture información lingüística rica. Esto es lo que permite al Slow AR operar solo sobre este codebook y aún así planificar la estructura completa del habla.

---

## 4. Firefly-GAN Vocoder (v1.x)

Usado en Fish Speech v1.2–v1.5 (en S2 se reemplaza por el decoder del DAC).

```
Tokens GFSQ
      │
      ▼
┌──────────────────────────────────┐
│  ParallelBlock (reemplaza MRF)   │
│  ─────────────────────────────── │
│  • Kernels configurables         │
│  • Dilations configurables       │
│  • Stack-and-average mechanism   │
│  • Más eficiente que HiFi-GAN    │
│    para inputs de codec          │
└──────────────┬───────────────────┘
               │
               ▼
          Waveform
```

---

## 5. Pipeline de datos

### S2: 3 etapas sin distribución shift

```
Audio crudo (10M+ horas, 80+ idiomas)
      │
      ▼
┌─ ETAPA 1: Preprocesamiento ────────────────────┐
│  • Separación de fuentes (vocals vs ruido)      │
│  • Eliminación de ruido                          │
│  • Voice Activity Detection (VAD)                │
│  • Segmentación por hablante                     │
└────────────────────┬────────────────────────────┘
                     ▼
┌─ ETAPA 2: Filtrado de calidad ─────────────────┐
│  • Modelo multidimensional de calidad           │
│    (backbone: w2v-BERT 2.0)                     │
│  • Evalúa:                                       │
│    - Signal-to-Noise Ratio (SNR)                │
│    - Consistencia de hablante                    │
│    - Calidad de grabación                        │
│    - Inteligibilidad                             │
└────────────────────┬────────────────────────────┘
                     ▼
┌─ ETAPA 3: Transcripción rica ──────────────────┐
│  • Qwen3-Omni-30B fine-tuned                   │
│  • Transcripción + anotación vocal simultánea   │
│  • Inyecta instrucciones vocales:               │
│    [risa prolongada], [inhalar], [enojado],     │
│    [susurro], [tono profesional], etc.          │
│  • Elimina distribution shift con RL            │
│    (el mismo modelo anota y sirve de reward)    │
└─────────────────────────────────────────────────┘
```

### Datos de entrenamiento por idioma (v1.5)

| Idioma | Horas |
|--------|-------|
| Inglés | 300,000+ |
| Chino | 300,000+ |
| Japonés | 100,000+ |
| Tier 2 (KO, ES, PT, AR, RU, FR, DE) | Extenso |
| Otros (80+ idiomas) | Variable |

---

## 6. Entrenamiento y alineación con RL

### Pre-entrenamiento

```
┌────────────────────────────────────────────┐
│  Pre-Training                               │
│  ──────────                                 │
│  • Base: Qwen3-4B (LLM pretrained)         │
│  • 500B+ tokens totales                    │
│  • Contexto progresivo: 8,192 → 16,384     │
│  • Interleaving modalidad (70% prob):       │
│    texto y audio mezclados a granularidad   │
│    fina → estabiliza pronunciación          │
│  • 30% corpus texto puro →                  │
│    previene olvido catastrófico             │
└────────────────────────────────────────────┘
```

### SFT (Supervised Fine-Tuning)

- Datos internos curados de alta calidad
- Enfoque en instrucciones de control vocal

### RL con GRPO (Group Relative Policy Optimization)

```
┌──────────────────────────────────────────────────┐
│  GRPO Post-Training                               │
│  ─────────────────                                │
│                                                    │
│  R_total = λ_STT · R_STT                         │
│          + λ_Pref · R_Pref                        │
│          + λ_SIM · R_SIM                          │
│                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐│
│  │ R_STT        │  │ R_Pref       │  │ R_SIM    ││
│  │ (Semántica)  │  │ (Acústica)   │  │ (Timbre) ││
│  │              │  │              │  │          ││
│  │ ASR re-trans │  │ Modelo de    │  │ Cosine   ││
│  │ cripción +   │  │ calidad de   │  │ simil.   ││
│  │ penalización │  │ habla        │  │ voiceprint││
│  │ por speaker  │  │              │  │ externo  ││
│  │ tags errados │  │              │  │          ││
│  └──────────────┘  └──────────────┘  └──────────┘│
│                                                    │
│  Optimización: LoRA weight-swap para reducir       │
│  memoria peak durante cómputo de KL divergence     │
└──────────────────────────────────────────────────┘
```

**Clave**: Los mismos modelos usados para limpiar y anotar datos se usan como Reward Models. Esto elimina el mismatch de distribución entre pre-training y post-training que afecta a otros sistemas.

---

## 7. Inferencia y optimización

### Motor de inferencia: SGLang

Fish Audio S2 hereda **todas las optimizaciones nativas de serving de LLMs** porque su Dual-AR es estructuralmente isomorfo a un LLM autoregresivo estándar.

```
┌───────────────────────────────────────────────┐
│  SGLang Serving Engine                         │
│  ─────────────────────                         │
│                                                │
│  • Continuous Batching                         │
│  • Paged KV Cache                              │
│  • CUDA Graph Replay                           │
│  • RadixAttention Prefix Caching               │
│    - Hit rate: 86.4% promedio, >90% pico       │
│    - Reutiliza contexto de audio de referencia  │
│  • Co-scheduled vocoder + LLM via MPS          │
│                                                │
│  Rendimiento (1x NVIDIA H200):                 │
│  ┌─────────────────────────────────────┐       │
│  │ RTF (Real-Time Factor): 0.195      │       │
│  │ TTFA (Time-to-First-Audio): <100ms │       │
│  │ Throughput: 3,000+ tokens/s        │       │
│  │ RTF a max throughput: <0.5         │       │
│  └─────────────────────────────────────┘       │
└───────────────────────────────────────────────┘
```

**RTF 0.195** significa que genera audio **~5x más rápido que tiempo real**.

---

## 8. Control de prosodia y emociones

A diferencia de modelos con tags predefinidos, Fish Audio S2 acepta **descripciones en lenguaje natural libre**:

```
Ejemplos de control inline:
─────────────────────────

"Hola [susurrar en voz baja], ¿cómo estás? [tono profesional de locución]"

"The results are [pitch up] amazing [prolonged laugh]"

"今日は [怒り] なんでそんなことしたの？ [溜め息]"

• 15,000+ tags de lenguaje natural disponibles
• Control a nivel de palabra
• No requiere tags predefinidos — texto libre
• Entrenado con anotaciones de Qwen3-Omni-30B
```

---

## 9. Multi-speaker y clonación de voz

```
┌────────────────────────────────────────────────┐
│  Voice Cloning                                  │
│  ─────────────                                  │
│  • Audio de referencia: 10-30 segundos          │
│  • Zero-shot (sin fine-tuning)                  │
│  • Cross-lingual (referencia en un idioma,      │
│    genera en otro)                               │
│                                                  │
│  Multi-Speaker                                   │
│  ─────────────                                   │
│  • Token <|speaker:i|> para cada hablante       │
│  • Una sola generación con múltiples voces      │
│  • Soporte nativo para diálogos multi-turno     │
│                                                  │
│  Similitud de hablante (v1.4):                  │
│  ┌─────────────────────────────────────┐        │
│  │ Resemblyzer: 0.914 (GT: 0.921)     │        │
│  │ Gap: solo 0.76%                     │        │
│  │ SpeechBrain: 0.762 (GT: 0.770)     │        │
│  │ Gap: solo 1.04%                     │        │
│  └─────────────────────────────────────┘        │
└────────────────────────────────────────────────┘
```

---

## 10. Benchmarks y resultados

### Seed-TTS Eval (WER — menor es mejor)

| Modelo | Chino | Inglés | ZH-Hard |
|--------|-------|--------|---------|
| **Fish Audio S2** | **0.54%** | **0.99%** | **5.99%** |
| Qwen3-TTS | 0.77% | 1.24% | — |
| MiniMax Speech-02 | 0.99% | 1.90% | — |
| Seed-TTS | 1.12% | 2.25% | — |
| Fish Audio S1 | 0.54% | 1.07% | 17.00% |

### Audio Turing Test (mayor es mejor, 0.5 = indistinguible de humano)

| Modelo | Score |
|--------|-------|
| **Fish Audio S2 Pro** | **0.515** |
| Seed-TTS | 0.417 |
| MiniMax Speech | 0.387 |

> **S2 Pro supera la barrera de 0.5**: es estadísticamente indistinguible del habla humana.

### EmergentTTS-Eval

| Métrica | Fish Audio S2 |
|---------|---------------|
| Win rate general | 81.88% |
| Win rate paralingüístico | 91.61% |

### Fish Speech v1.5 (TTS Arena)

| Métrica | Valor |
|---------|-------|
| ELO Score | 1339 |
| WER (inglés) | 3.5% |
| CER (inglés) | 1.2% |
| CER (chino) | 1.3% |

### Calidad e instrucciones (S2 Pro)

| Idioma | TAR (adherencia) | Naturalidad | Expresividad |
|--------|-------------------|-------------|--------------|
| Chino | 0.984 | 4.40/5 | 4.94/5 |
| Inglés | 0.881 | 4.21/5 | 4.50/5 |
| General | 0.933 | — | 4.51/5 |

---

## 11. Comparativa con otros modelos

### ¿Por qué Fish Audio supera a los demás?

| Aspecto | Fish Audio S2 | XTTS v2 | Bark | Tortoise | VALL-E | F5-TTS |
|---------|---------------|---------|------|----------|--------|--------|
| **Params** | 4B + 400M | ~500M | ~1B | ~1B | ~370M | ~300M |
| **Datos** | 10M+ horas | ~10K horas | Desconocido | ~50K horas | 60K horas | ~100K horas |
| **RTF** | 0.195 | ~1.0 | ~3.0 | ~10+ | ~2.0 | ~0.5 |
| **TTFA** | <100ms | ~500ms | ~2s | ~10s | ~1s | ~300ms |
| **WER** | 0.99% (EN) | ~5% | ~8% | ~4% | ~3% | ~3% |
| **Audio Turing** | 0.515 | — | — | — | — | — |
| **Idiomas** | 80+ | 17 | 13 | ~3 | 1 | ~5 |
| **Control emocional** | 15K+ tags NL | No | Limitado | No | No | No |
| **Multi-speaker** | Nativo | No | No | No | No | No |
| **RL alignment** | GRPO | No | No | No | No | No |
| **Sample rate** | 44.1 kHz | 24 kHz | 24 kHz | 24 kHz | 24 kHz | 24 kHz |
| **Backbone LLM** | Qwen3-4B | GPT-2 | GPT custom | GPT custom | — | DiT |

### Ventajas arquitectónicas clave

1. **LLM pretrained como backbone**: Qwen3-4B ya entiende lenguaje → mejor pronunciación, polisemia, code-switching.
2. **Dual-AR asimétrico**: 4B para semántica, 400M para acústica. Los otros modelos distribuyen params uniformemente.
3. **RL post-training (GRPO)**: Único modelo TTS con alineamiento por RL multi-dimensional. Similar a RLHF en LLMs.
4. **10M+ horas de datos**: Orden de magnitud más que cualquier competidor open/closed.
5. **Infraestructura LLM nativa**: SGLang, KV cache, prefix caching. No reinventa serving.
6. **Sin G2P**: No depende de reglas grafema-a-fonema frágiles. El LLM entiende texto directamente.

---

## 12. API, SDK y herramientas

### API REST

```
Endpoint: https://api.fish.audio/v1/tts

Features:
- Streaming de audio en tiempo real
- Clonación de voz con audio de referencia
- Control de emociones/prosodia inline
- Multi-speaker en una sola request
- Formatos: WAV, MP3, OGG, FLAC
```

### SDKs oficiales

| SDK | Instalación | Notas |
|-----|-------------|-------|
| **Python** | `pip install fish-audio-sdk` | Python 3.9+, async, streaming, type hints |
| **JavaScript/TS** | `npm install fish-audio-sdk` | Node 16+ |
| **Go** | `fish-audio-go` | — |
| **n8n** | Community node oficial | Workflow automation |
| **Dify** | Integración nativa | — |

### Open Source

| Recurso | URL |
|---------|-----|
| Código (28K+ stars) | github.com/fishaudio/fish-speech |
| Modelos | huggingface.co/fishaudio |
| Docs | speech.fish.audio |
| Bert-VITS2 (8.6K stars) | github.com/fishaudio/Bert-VITS2 |
| Audio preprocessing | github.com/fishaudio/audio-preprocess |

### Formatos de salida

- MP3 (64/128/192 kbps), WAV, PCM, Opus
- Sample rate: 44.1 kHz

### Despliegue

- Docker containerizado
- WebUI incluido
- SGLang server para producción
- CLI para inferencia local

---

## 13. Referencias

- **Fish Speech v1.4 Paper**: [arXiv:2411.01156](https://arxiv.org/abs/2411.01156) (2024)
- **Fish Audio S2 Technical Report**: [arXiv:2603.08823](https://arxiv.org/abs/2603.08823) (2026)
- **GitHub**: [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)
- **HuggingFace**: [fishaudio](https://huggingface.co/fishaudio)
- **Web**: [fish.audio](https://fish.audio)
- **Blog**: [Introducing Fish Speech](https://fish.audio/blog/introducing-fish-speech/)
