# Codecs Neurales de Audio — Mini Modelo para Análisis de Ondas

> Fish Audio usa un DAC custom de 446M params con destilación semántica.
> Nosotros necesitamos lo mismo pero más pequeño.
> Este documento compara todos los codecs open-source y elige uno.

---

## Qué hace Fish Audio (y por qué funciona)

```
Audio 44.1kHz
      │
      ▼
┌─────────────────────────────────────────────────┐
│  ModifiedDAC (446M params)                       │
│  ═════════════════════════                       │
│                                                   │
│  Encoder:                                         │
│    DAC base + ConvNeXt V2 (4x extra downsample)  │
│    → Total downsampling: 2048x                   │
│    → Frame rate: ~21 Hz                          │
│    → Convolutions causales (para streaming)      │
│    → Causal Sliding-Window Transformer bottleneck│
│                                                   │
│  Cuantización: 10 RVQ codebooks                  │
│                                                   │
│    ┌─────────────────────────────────────────┐   │
│    │  Codebook 1: SEMÁNTICO                  │   │
│    │  ─────────────────────                  │   │
│    │  Prediction head auxiliar que regresa    │   │
│    │  activaciones de la capa 16 de          │   │
│    │  w2v-BERT 2.0 (Google SSL model)        │   │
│    │                                          │   │
│    │  → Captura: fonética, idioma, prosodia  │   │
│    │  → Es lo que el Slow AR predice         │   │
│    └─────────────────────────────────────────┘   │
│                                                   │
│    ┌─────────────────────────────────────────┐   │
│    │  Codebooks 2-10: ACÚSTICOS              │   │
│    │  ─────────────────────────              │   │
│    │  Progresivamente más finos:             │   │
│    │  2-3: estructura espectral gruesa       │   │
│    │  4-6: timbre, formantes                 │   │
│    │  7-10: micro-textura, respiración       │   │
│    │                                          │   │
│    │  → Es lo que el Fast AR predice         │   │
│    └─────────────────────────────────────────┘   │
│                                                   │
│  Decoder: EVA-GAN generator (no el DAC original) │
│                                                   │
│  Discriminadores: 3 simultáneos                  │
│    • Multi-period                                │
│    • Multi-resolution                            │
│    • Multi-scale STFT                            │
│                                                   │
│  Entrenamiento: 1M steps, composite GAN loss     │
└─────────────────────────────────────────────────┘
```

**El truco de Fish**: La destilación semántica en el codebook 1 es lo que permite que el Slow AR (Qwen3-4B) "entienda" el audio. Sin esto, el LLM solo vería códigos acústicos sin significado lingüístico.

---

## Todos los codecs neurales open-source

### Tabla comparativa

```
CODEC            PARAMS  SR      CB   BITRATE   FR      SEMÁNTICO      STREAMING  LIC.
══════════════════════════════════════════════════════════════════════════════════════════

EnCodec (Meta)    23M    24kHz   2-32  1.5-24k   75Hz   ✗              ✓ causal   MIT
DAC (Descript)    74M    44.1k   9     8kbps     86Hz   ✗              ✗          MIT
Mimi (Kyutai)     96M    24kHz   8-16  1.1kbps   12.5Hz ✓ WavLM       ✓ 80ms     CC-BY
SpeechTokenizer   ~35M   16kHz   8     4kbps     50Hz   ✓ HuBERT      ✗          MIT
WavTokenizer      ~85M   24kHz   1     0.6kbps   40Hz   ✓ implícito   ✗          MIT
SNAC              20M    24kHz   3-4   0.98kbps  12-47  ✗ (multi-sc.) ✗          MIT
X-Codec 2.0       ~80M   16kHz   8     1kbps     50Hz   ✓ pre-train   ✗          Apache
BigCodec          160M   16kHz   1     0.53kbps  25Hz   ✓             ✗          Apache
FunCodec          30-80M 16kHz   8-32  1-6kbps   50Hz   parcial       ✗          Apache
Fish ModifiedDAC  446M   44.1k   10    ~2.1kbps  21Hz   ✓ w2v-BERT    ✓ causal   propiet.
```

```
CB = codebooks, FR = frame rate, SR = sample rate
```

### Detalle de cada codec

---

### 1. Mimi (Kyutai) — usado en Moshi

```
Tipo:           RVQ + Transformer bottleneck + destilación semántica
Params:         ~96M
Sample rate:    24 kHz
Bitrate:        1.1 kbps (ultra bajo)
Codebooks:      8-16 (típicamente 8), 2048 entries cada uno
Frame rate:     12.5 Hz (muy bajo → secuencias cortas para el LLM)
Streaming:      Sí, causal, 80ms latencia

Split-RVQ:
  Codebook 1: destilado de WavLM (modelo SSL)
              → captura contenido semántico + prosodia
  Codebooks 2-7: acústicos progresivos
              → timbre, textura, detalle

Entrenamiento:  Solo adversarial (sin spectral loss)
                → mejor calidad subjetiva

Usado en:       Moshi (70+ emociones), Sesame CSM
VRAM:           ~200-450MB
Licencia:       CC-BY 4.0
GitHub:         github.com/kyutai-labs/moshi (Mimi incluido)
```

**Por qué importa**: 12.5 Hz frame rate = 10 segundos de audio son solo 125 tokens. Un LLM con contexto 2048 puede procesar ~2.7 minutos de audio. Con los codecs de 75-86 Hz necesitarías 6x más contexto.

---

### 2. WavTokenizer

```
Tipo:           VQ-GAN, codebook único con vocab grande
Params:         ~85M
Sample rate:    24 kHz
Bitrate:        0.6 kbps (40 Hz) / 0.9 kbps (75 Hz)
Codebooks:      1 (!) — vocab de 4096 o 16384 entries
Frame rate:     40 Hz o 75 Hz

Innovación:     Un solo codebook captura semántica + acústica
                → No necesitas Fast AR
                → Tu LLM predice 1 token por frame
                → Arquitectura radicalmente más simple

                100% utilización del codebook via k-means init
                + random reactivation

Decoder:        Inverse Fourier Transform
                → mejor cobertura de frecuencias

Calidad:        En evaluaciones TTS, 1 codebook de WavTokenizer
                supera 9 codebooks de DAC en calidad Y prosodia

VRAM:           ~300MB
Licencia:       MIT
GitHub:         github.com/jishengpeng/WavTokenizer
Paper:          ICLR 2025
```

---

### 3. SNAC (Multi-Scale Neural Audio Codec)

```
Tipo:           RVQ-GAN multi-escala (basado en DAC)
Params:         19.8M (24kHz) / 54.5M (44kHz)
Sample rate:    24 / 32 / 44 kHz
Bitrate:        0.98 kbps (24kHz) / 2.6 kbps (44kHz)
Codebooks:      3 niveles @ 24kHz, 4 niveles @ 44kHz
                4096 entries por codebook

Multi-escala:
  24kHz:  Nivel 1: 12 Hz (coarse — estructura, prosodia)
          Nivel 2: 23 Hz (mid — formantes)
          Nivel 3: 47 Hz (fine — textura)

  44kHz:  Nivel 1: 14 Hz
          Nivel 2: 29 Hz
          Nivel 3: 57 Hz
          Nivel 4: 115 Hz

  Por cada 1 token coarse → 2 tokens mid → 4 tokens fine
  Secuencia plana interleaved para AR

Sin destilación semántica (la separación es implícita
por resolución temporal)

VRAM:           ~80-220MB
Licencia:       MIT
GitHub:         github.com/hubertsiuzdak/snac
Paper:          NeurIPS 2024 Workshop
```

**El más pequeño**: 19.8M params. Cabe en cualquier cosa.

---

### 4. SpeechTokenizer

```
Tipo:           EnCodec + BiLSTM + RVQ + destilación HuBERT
Params:         ~35M (estimado, basado en EnCodec)
Sample rate:    16 kHz
Bitrate:        4 kbps
Codebooks:      8 (1 semántico + 7 acústicos), 1024 entries
Frame rate:     50 Hz

Split-RVQ:
  Codebook 1: destilado de HuBERT capa 9
              → contenido semántico
  Codebooks 2-8: acústicos
              → timbre, detalle

Debilidad:      Prosodia/F0 parcialmente perdida a bitrates bajos
                → emociones se degradan

Importancia:    PIONERO del split semántico/acústico (ICLR 2024)
                Mimi y Fish S2 adoptaron este diseño después

VRAM:           <200MB
Licencia:       MIT
GitHub:         github.com/ZhangXInFD/SpeechTokenizer
```

---

### 5. EnCodec (Meta)

```
Tipo:           Convolutional encoder-decoder + RVQ
Params:         ~23M
Sample rate:    24 kHz (speech) / 48 kHz (stereo music)
Bitrate:        1.5 / 3 / 6 / 12 / 24 kbps (variable)
Codebooks:      2-32 (configurable), 1024 entries
                Nº de codebooks activos = bitrate
Frame rate:     75 Hz (stride 320 @ 24kHz)

Sin destilación semántica.
Info paralingüística distribuida sin separación.

Incluye:        Transformer entropy coder opcional
                → 25-40% reducción de bitrate adicional

VRAM:           ~100MB (corre en CPU en tiempo real)
Licencia:       MIT
GitHub:         github.com/facebookresearch/encodec

El OG: fundacional, inspiró todo lo demás.
Usado por VALL-E, MusicGen, AudioGen.
```

---

### 6. DAC (Descript Audio Codec) — original

```
Tipo:           RVQ-GAN mejorado
Params:         ~74M (44.1kHz)
Sample rate:    16 / 24 / 44.1 kHz
Bitrate:        ~8 kbps @ 44.1kHz (90x compresión)
Codebooks:      9, 1024 entries, embeddings 8D
                L2-norm + cosine similarity (no euclidean)
Frame rate:     ~86 Hz @ 44.1kHz (stride 512)

Innovaciones:   Snake activations (periódicas)
                Codebook factorization con L2-norm
                Quantizer dropout → multi-bitrate single model

Sin destilación semántica.
Mejor calidad cruda a 44.1kHz.
Base de Fish S2 (pero Fish le añade 370M params extra).

VRAM:           ~300MB
Licencia:       MIT
GitHub:         github.com/descriptinc/descript-audio-codec
```

---

### 7. X-Codec 2.0 (Microsoft)

```
Tipo:           RVQ con conditioning semántico
Params:         ~80M
Sample rate:    16 kHz
Bitrate:        ~1 kbps
Codebooks:      8, semántico inyectado en TODOS los codebooks
Frame rate:     50 Hz

Diferente a Mimi/SpeechTokenizer: en vez de
separar semántico (CB1) vs acústico (CB2-8),
X-Codec inyecta features semánticas de un
modelo pre-entrenado en todos los codebooks.

VRAM:           <500MB
Licencia:       Apache 2.0
GitHub:         github.com/microsoft/X-Codec
```

---

### 8. Otros notables

```
BigCodec (160M)
  1 codebook, 8192 entries, 25Hz, 0.53 kbps
  El bitrate más bajo de todos

AFACodec (basado en DAC)
  Dual-stream adaptive feature-aware
  75% accuracy en emoción (vs 76.76% ground truth)
  → Mejor preservación de emoción de todos los codecs

FACodec
  Descompone explícitamente: contenido, prosodia,
  timbre, detalle acústico en subspaces separados

FreeCodec
  1 codebook, 256 entries, 0.45 kbps
  Supera a FACodec (2.4 kbps) en evaluación subjetiva

FlexiCodec
  Frame rate dinámico, 4.15% WER incluso a 6.25 Hz
```

---

## Comparativa: lo que importa para nuestro proyecto

```
                    PARAMS  SEMÁNTICO  EMOCIÓN  44.1k  STREAMING  SIMPLE
                    ══════  ═════════  ═══════  ═════  ═════════  ══════

Mimi                96M    ✓✓✓✓✓      ✓✓✓✓    ✗      ✓✓✓✓✓      ★★★☆
WavTokenizer        85M    ✓✓✓✓       ✓✓✓     ✗      ✗          ★★★★★
SNAC                20M    ✓✓         ✓✓      ✓(44k) ✗          ★★★★☆
SpeechTokenizer     35M    ✓✓✓✓       ✓✓      ✗      ✗          ★★★☆☆
EnCodec             23M    ✗          ✗       ✗      ✓          ★★★★☆
DAC                 74M    ✗          ✗       ✓✓✓✓✓  ✗          ★★★★☆
X-Codec 2.0         80M    ✓✓✓✓       ✓✓✓     ✗      ✗          ★★★☆☆
Fish ModifiedDAC   446M    ✓✓✓✓✓      ✓✓✓✓✓   ✓✓✓✓✓  ✓✓✓✓✓      ★★☆☆☆
```

---

## Decisión: Mimi como nuestro mini modelo

```
┌─────────────────────────────────────────────────────────┐
│                                                          │
│  ELEGIMOS: Mimi (~96M, <450MB VRAM)                     │
│  ═══════════════════════════════════                     │
│                                                          │
│  ¿Por qué?                                              │
│                                                          │
│  1. Hace exactamente lo que Fish hace:                  │
│     Split-RVQ con destilación semántica                 │
│     (WavLM en vez de w2v-BERT, mismo principio)         │
│                                                          │
│  2. 96M vs 446M de Fish → 4.6x más pequeño             │
│                                                          │
│  3. 12.5 Hz frame rate → secuencias ultra cortas        │
│     10s audio = 125 tokens (vs 210 en Fish, 860 en DAC) │
│                                                          │
│  4. Streaming causal con 80ms latencia                  │
│                                                          │
│  5. Probado en producción: Moshi reconoce 70+ emociones │
│     directamente desde tokens Mimi                      │
│                                                          │
│  6. CC-BY 4.0 (uso comercial OK)                        │
│                                                          │
│  Trade-off: 24kHz (no 44.1kHz)                          │
│  → Aceptable para V1. Podemos subir después.            │
│                                                          │
└─────────────────────────────────────────────────────────┘

Alternativa para V2: WavTokenizer
  → 1 solo codebook = no necesitamos Fast AR
  → Simplificación radical de la arquitectura
  → Evaluar cuando tengamos el pipeline funcionando
```

---

## Cómo encaja Mimi en nuestra arquitectura

```
                    Audio de entrada
                          │
                          ▼
                ┌───────────────────┐
                │  Audio Analyzer    │     ← doc 07
                │  SNR, VAD, ruido   │
                └─────────┬─────────┘
                          │
               ┌──────────┴──────────┐
               │                     │
               ▼                     ▼
      RAMA ANOTACIÓN          RAMA TTS
      (doc 08)                (doc 07)
               │                     │
               │                     ▼
               │           ┌───────────────────┐
               │           │  Mimi Encoder      │  ← ESTE DOC
               │           │  (96M, <450MB)     │
               │           │                    │
               │           │  Audio → 8 codebooks│
               │           │  @ 12.5 Hz         │
               │           │                    │
               │           │  CB1: semántico    │
               │           │  CB2-8: acústico   │
               │           └─────────┬─────────┘
               │                     │
               │                     ▼
               │           ┌───────────────────┐
               │           │  Slow AR (LLM)     │
               │           │  Predice CB1       │
               │           │  en el eje temporal│
               │           └─────────┬─────────┘
               │                     │
               │                     ▼
               │           ┌───────────────────┐
               │           │  Fast AR           │
               │           │  Predice CB2-8     │
               │           │  por timestep      │
               │           └─────────┬─────────┘
               │                     │
               │                     ▼
               │           ┌───────────────────┐
               │           │  Mimi Decoder      │
               │           │  Tokens → 24kHz    │
               │           └─────────┬─────────┘
               │                     │
               │                     ▼
               │                Audio WAV
               │
               ▼
      SenseVoice + Emotion2Vec
      + GLM-4-Voice (escalación)
      → Anotaciones ricas
      → Datos de entrenamiento
```

---

## Actualización de la arquitectura del sistema

Con Mimi en vez de DAC, cambia:

| Aspecto | Antes (doc 07, DAC) | Ahora (Mimi) |
|---------|---------------------|--------------|
| Codec params | 74M (DAC orig.) | 96M (Mimi) |
| Sample rate | 44.1 kHz | 24 kHz |
| Frame rate | ~86 Hz | 12.5 Hz |
| Tokens por 10s | ~860 | ~125 |
| Codebooks | 9 | 8 |
| Semántico | No | Sí (WavLM distill) |
| Streaming | No | Sí (80ms, causal) |
| VRAM codec | ~300MB | ~450MB |
| Contexto LLM para 30s | 2580 tokens | 375 tokens |

**El cambio más grande**: 12.5 Hz vs 86 Hz = **6.9x menos tokens**. Nuestro LLM backbone puede procesar audio mucho más largo sin agotar el contexto.

---

## Roadmap de codec

### V1: Mimi como codec base
- [ ] Integrar Mimi encoder/decoder
- [ ] Validar calidad a 24kHz para nuestros idiomas (ES, EN)
- [ ] Probar que el codebook 1 captura prosodia/emoción
- [ ] Benchmark: tokens/segundo en RTX 4090

### V2: Evaluar alternativas
- [ ] Comparar Mimi vs WavTokenizer (1 codebook, sin Fast AR)
- [ ] Comparar Mimi vs SNAC 44kHz (si necesitamos 44.1kHz)
- [ ] Considerar entrenar codec custom con destilación semántica
      (como hizo Fish: DAC base + w2v-BERT distill + EVA-GAN decoder)

### V3: Codec custom (si es necesario)
- [ ] Fine-tune Mimi en nuestros datos
- [ ] O construir mini-codec custom (~50-100M) con:
      - DAC encoder (causal convolutions)
      - Destilación de w2v-BERT 2.0 o WavLM en CB1
      - EVA-GAN decoder
      - Entrenamiento multi-discriminador

---

## Referencias

- [Fish Speech paper (v1.4)](https://arxiv.org/abs/2411.01156)
- [Fish Audio S2 Pro report](https://arxiv.org/abs/2603.08823)
- [Mimi / Moshi paper](https://arxiv.org/abs/2410.00037)
- [WavTokenizer paper (ICLR 2025)](https://arxiv.org/abs/2408.16532)
- [SNAC paper (NeurIPS 2024)](https://arxiv.org/abs/2410.02264)
- [SpeechTokenizer paper (ICLR 2024)](https://arxiv.org/abs/2308.16692)
- [EnCodec paper](https://arxiv.org/abs/2210.13438)
- [DAC GitHub](https://github.com/descriptinc/descript-audio-codec)
- [X-Codec GitHub](https://github.com/microsoft/X-Codec)
- [BigCodec paper](https://arxiv.org/abs/2409.16663)
