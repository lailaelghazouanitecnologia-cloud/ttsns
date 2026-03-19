# Arquitecturas Modernas de TTS

## 1. VITS (2021) — Conditional Variational Autoencoder with Adversarial Learning

### Arquitectura

```
                    ENTRENAMIENTO
                    ═════════════

Texto (fonemas)                      Audio (waveform)
      │                                     │
      ▼                                     ▼
┌────────────┐                    ┌──────────────────┐
│ Text       │                    │ Posterior Encoder │
│ Encoder    │                    │ (Linear Spectro   │
│ (Transf.)  │                    │  → WaveNet-like)  │
└─────┬──────┘                    └────────┬─────────┘
      │                                    │
      │              ┌─────────┐           │
      │              │ MAS     │◄──────────┤
      │              │(Monotonic│           │
      │              │Alignment │           │
      │              │Search)  │           │
      │              └────┬────┘           │
      │                   │                │
      ▼                   ▼                ▼
┌──────────────────────────────────────────────┐
│         Normalizing Flow (f)                  │
│    Prior z_p ◄══════════════► Posterior z_q   │
│    (del texto)                (del audio)     │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
          ┌────────────────┐
          │  HiFi-GAN      │
          │  Decoder        │
          └────────┬───────┘
                   │
                   ▼
              Audio WAV


                    INFERENCIA
                    ══════════

Texto → Text Encoder → Prior → Flow inverso → z → HiFi-GAN → Audio
```

- **Tipo**: VAE condicional + GAN + Normalizing Flows, end-to-end.
- **Innovación clave**: Primer modelo que genera waveform directamente desde texto en un solo modelo entrenado end-to-end.
- **MAS (Monotonic Alignment Search)**: Algoritmo eficiente para aprender alineación texto-audio sin supervisión externa.
- **Calidad**: MOS comparable a audio real en varios benchmarks.
- **Velocidad**: Tiempo real en GPU, razonablemente rápido en CPU.
- **Variantes**: VITS2, MB-iSTFT-VITS (más rápido), so-vits-svc (canto).

---

## 2. Tortoise TTS (2022) — James Betker

### Arquitectura

```
Texto + Audio de referencia (clonación de voz)
       │              │
       ▼              ▼
┌──────────┐   ┌──────────────┐
│ Text     │   │ CLVP         │
│ Tokens   │   │ (Contrastive │
│          │   │  Language-    │
│          │   │  Voice        │
│          │   │  Pretraining) │
└────┬─────┘   └──────┬───────┘
     │                │
     ▼                ▼
┌──────────────────────────┐
│  Autoregressive          │
│  Transformer (GPT-like)  │
│  → genera MEL tokens     │
│    (múltiples candidatos)│
└───────────┬──────────────┘
            │
     ┌──────┴──────┐
     │  Re-ranking │ ← CLVP selecciona el mejor candidato
     │  (CLVP)     │
     └──────┬──────┘
            ▼
┌──────────────────────┐
│  Diffusion Model     │
│  (genera mel-spectro │
│   detallado)         │
└───────────┬──────────┘
            ▼
┌──────────────────────┐
│  UnivNet Vocoder     │
└───────────┬──────────┘
            ▼
        Audio WAV
```

- **Tipo**: Autoregresivo (GPT) + Difusión + Vocoder.
- **Innovación**: Usa CLVP para re-ranking de candidatos (similar a CLIP pero para voz). Excelente clonación zero-shot.
- **Calidad**: Muy alta naturalidad, especialmente para inglés.
- **Limitación**: Muy lento (múltiples pasadas AR + difusión). No apto para tiempo real sin optimización.

---

## 3. Bark (Suno AI, 2023)

### Arquitectura

```
Texto
  │
  ▼
┌───────────────────────┐
│  GPT Semántico        │
│  (Transformer AR)     │
│  Texto → Tokens       │
│  semánticos           │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  GPT de Grano Grueso  │
│  (Coarse Acoustic)    │
│  Tokens semánticos →  │
│  Tokens acústicos     │
│  gruesos (EnCodec     │
│  primeros 2 codebooks)│
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  GPT de Grano Fino    │
│  (Fine Acoustic)      │
│  Tokens gruesos →     │
│  Tokens acústicos     │
│  finos (EnCodec       │
│  8 codebooks)         │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  EnCodec Decoder      │
│  (Meta)               │
│  Tokens → Waveform    │
└───────────┬───────────┘
            │
            ▼
        Audio WAV
```

- **Tipo**: Cascada de 3 modelos Transformer autoregresivos + codec neuronal.
- **Innovación**: Genera no solo voz sino también risas, pausas, música de fondo. Altamente expresivo.
- **Codec**: Usa EnCodec (Meta) como representación discreta de audio.
- **Multilingüe**: Soporte para múltiples idiomas.
- **Limitación**: Puede ser impredecible. No hay control fino sobre prosodia.

---

## 4. XTTS / Coqui TTS (2023)

### Arquitectura

```
Texto + Audio de referencia (speaker embedding)
       │              │
       ▼              ▼
┌──────────┐   ┌──────────────────┐
│ Text     │   │ Speaker Encoder  │
│ Encoder  │   │ (extrae embedding│
│          │   │  del hablante)   │
└────┬─────┘   └───────┬──────────┘
     │                 │
     └────────┬────────┘
              ▼
┌──────────────────────────┐
│  GPT-2 Autoregressive    │
│  Decoder                 │
│  (genera latent codes)   │
└───────────┬──────────────┘
            ▼
┌──────────────────────────┐
│  Perceiver Resampler     │
│  (comprime secuencia)    │
└───────────┬──────────────┘
            ▼
┌──────────────────────────┐
│  Diffusion Decoder       │
│  (latent → mel-spectro)  │
└───────────┬──────────────┘
            ▼
┌──────────────────────────┐
│  HiFi-GAN Vocoder        │
└───────────┬──────────────┘
            ▼
        Audio WAV
```

- **Tipo**: GPT autoregresivo + Difusión + Vocoder.
- **Innovación**: Clonación de voz multilingüe (17+ idiomas) con solo 6 segundos de audio de referencia.
- **Licencia**: Open-source (Coqui Public Model License).
- **Calidad**: Excelente para clonación cross-lingual.

---

## 5. VALL-E (Microsoft, 2023) y VALL-E X

### Arquitectura

```
Texto (fonemas) + Audio de referencia (3 seg)
       │                    │
       ▼                    ▼
┌──────────┐        ┌──────────────┐
│ Phoneme  │        │ EnCodec      │
│ Embedding│        │ Encoder      │
│          │        │ → 8 codebooks│
└────┬─────┘        └──────┬───────┘
     │                     │
     └─────────┬───────────┘
               ▼
┌──────────────────────────────────┐
│  Etapa 1: AR Transformer        │
│  (genera primer codebook         │
│   token a token)                 │
└───────────────┬──────────────────┘
                ▼
┌──────────────────────────────────┐
│  Etapa 2: NAR Transformer       │
│  (genera codebooks 2-8           │
│   en paralelo, condicionado     │
│   en el primer codebook)         │
└───────────────┬──────────────────┘
                ▼
┌──────────────────────────────────┐
│  EnCodec Decoder                 │
│  (8 codebooks → waveform)        │
└───────────────┬──────────────────┘
                ▼
            Audio WAV
```

- **Tipo**: Híbrido AR + NAR sobre tokens de codec de audio.
- **Innovación**: Trata TTS como un problema de "language modeling" de tokens de audio. Clonación con solo 3 segundos.
- **Entrenamiento**: 60K horas de habla en inglés (LibriLight).
- **VALL-E X**: Extensión cross-lingual.
- **Limitación**: No open-source (solo paper). Requiere dataset masivo.

---

## 6. F5-TTS (2024) — Flow Matching TTS

### Arquitectura

```
Texto + Audio de referencia
       │              │
       ▼              ▼
┌──────────────────────────────┐
│  DiT (Diffusion Transformer)│
│  con Flow Matching           │
│                              │
│  - Sin duración explícita    │
│  - Sin alineamiento forzado  │
│  - Entrenamiento simple      │
│    (solo flow matching loss) │
└──────────────┬───────────────┘
               ▼
           Audio WAV
```

- **Tipo**: Flow matching + Diffusion Transformer.
- **Innovación**: Arquitectura extremadamente simple. Sin predictores de duración ni alineación. Entrenamiento con una sola loss.
- **Calidad**: Competitiva con modelos mucho más complejos.
- **Velocidad**: Más rápido que modelos de difusión tradicionales gracias a flow matching.

---

## Resumen de tendencias

1. **Codec-based**: La tendencia dominante es tratar audio como tokens discretos (EnCodec, DAC, SpeechTokenizer).
2. **LLM-like**: Los modelos más recientes tratan TTS como un problema de modelado de lenguaje.
3. **Zero-shot voice cloning**: Todos los modelos modernos permiten clonación con pocos segundos de audio.
4. **Flow matching**: Está reemplazando a la difusión clásica por ser más rápido y estable.
5. **Simplificación**: La tendencia es hacia arquitecturas más simples con menos componentes (F5-TTS).
