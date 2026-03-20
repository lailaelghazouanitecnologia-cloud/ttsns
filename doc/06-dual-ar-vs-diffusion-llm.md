# Dual-AR (Fish Audio) vs LLM + Difusión — Comparativa de arquitecturas

> Ambos enfoques usan un LLM como backbone para entender texto.
> La diferencia está en **cómo generan el audio** a partir de esa comprensión.

---

## Visión general: dos caminos desde el mismo punto de partida

```
                         Texto + Audio referencia
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │      LLM Backbone    │
                        │   (ej. Qwen3-4B)     │
                        │                       │
                        │   Entiende el texto,  │
                        │   planifica prosodia,  │
                        │   codifica hablante    │
                        └──────────┬────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
        ┌─────────────────────┐     ┌──────────────────────────┐
        │   CAMINO A:         │     │   CAMINO B:              │
        │   DUAL-AR           │     │   LLM + DIFUSIÓN         │
        │   (Fish Audio)      │     │                          │
        │                     │     │                          │
        │   LLM genera tokens │     │   LLM genera condición   │
        │   de audio uno a    │     │   latente, luego difusión│
        │   uno (codec)       │     │   genera audio completo  │
        │                     │     │   desde ruido            │
        └─────────┬───────────┘     └────────────┬─────────────┘
                  │                              │
                  ▼                              ▼
              Audio WAV                      Audio WAV
```

---

## Arquitectura A: Dual-AR (Fish Audio S2)

```
Texto ─► [Qwen3-4B Tokenizer]
              │
              ▼
┌──────────────────────────────────────────┐
│          SLOW AR (4B params)              │
│          ════════════════════             │
│  Qwen3-4B autoregresivo en tiempo        │
│                                           │
│  t=0   t=1   t=2   t=3   t=4   ...      │
│  [s₀] → [s₁] → [s₂] → [s₃] → [s₄] →   │
│                                           │
│  Genera 1 token semántico por frame      │
│  (codebook 1 del DAC)                    │
│  Cada token condicionado en los previos  │
└──────────────────┬───────────────────────┘
                   │ hidden states
                   ▼
┌──────────────────────────────────────────┐
│          FAST AR (400M params)            │
│          ═════════════════════            │
│  Por cada timestep, genera codebooks 2-10│
│                                           │
│  t=0: s₀ → [c₂, c₃, c₄, ..., c₁₀]     │
│  t=1: s₁ → [c₂, c₃, c₄, ..., c₁₀]     │
│  t=2: s₂ → [c₂, c₃, c₄, ..., c₁₀]     │
│  ...                                      │
│                                           │
│  4 capas Transformer, embedding shared   │
└──────────────────┬───────────────────────┘
                   │ 10 codebooks × T frames
                   ▼
┌──────────────────────────────────────────┐
│          DAC DECODER                      │
│  10 codebooks → waveform 44.1 kHz        │
└──────────────────┬───────────────────────┘
                   │
                   ▼
               Audio WAV

FLUJO TEMPORAL:
═══════════════
  t=0 ────► t=1 ────► t=2 ────► ... ────► t=N ────► FIN
  │          │          │                    │
  genera     genera     genera               genera
  frame 0    frame 1    frame 2              frame N
  (streaming (streaming  ...                 último
   desde     continúa)                       frame)
   aquí)
```

---

## Arquitectura B: LLM + Difusión

```
Texto ─► [LLM Backbone (ej. Qwen3-4B)]
              │
              ▼
┌──────────────────────────────────────────┐
│     LLM ENCODER / PLANNER                │
│     ══════════════════════                │
│                                           │
│  Genera representación latente COMPLETA   │
│  de toda la frase de una vez:             │
│                                           │
│  "Hola, ¿cómo estás?" →                 │
│                                           │
│  [h₀, h₁, h₂, ..., hₘ]                 │
│  (secuencia de hidden states)            │
│                                           │
│  + Speaker embedding (del audio ref)      │
│  + Instrucciones de emoción/prosodia      │
└──────────────────┬───────────────────────┘
                   │ condición completa
                   ▼
┌──────────────────────────────────────────┐
│     DURATION PREDICTOR                    │
│     ══════════════════                    │
│                                           │
│  Predice cuántos frames dura cada token:  │
│                                           │
│  h₀: 12 frames                           │
│  h₁: 8 frames                            │
│  h₂: 15 frames                           │
│  ...                                      │
│  Total: T frames                          │
│                                           │
│  Expande la secuencia a longitud T        │
└──────────────────┬───────────────────────┘
                   │ condición expandida (T frames)
                   ▼
┌──────────────────────────────────────────┐
│     DIFFUSION / FLOW MATCHING            │
│     ═════════════════════════            │
│                                           │
│  Paso 0: Ruido gaussiano puro z_T        │
│          (T frames × dim acústica)        │
│          │                                │
│          ▼                                │
│  Paso 1: DiT(z_T, condición) → z_{T-1}  │
│          Un poco menos de ruido           │
│          │                                │
│          ▼                                │
│  Paso 2: DiT(z_{T-1}, cond) → z_{T-2}   │
│          Empieza a parecer audio          │
│          │                                │
│          ▼                                │
│          ...                              │
│          │                                │
│          ▼                                │
│  Paso N: DiT(z_1, cond) → z_0           │
│          Audio limpio                     │
│                                           │
│  ┌─────────────────────────────────┐     │
│  │  Diffusion Transformer (DiT)    │     │
│  │  ───────────────────────────    │     │
│  │  • Self-attention sobre frames  │     │
│  │  • Cross-attention con cond.    │     │
│  │  • Timestep embedding           │     │
│  │  • CFG (Classifier-Free         │     │
│  │    Guidance) para control       │     │
│  └─────────────────────────────────┘     │
│                                           │
│  N = 20-50 pasos (flow matching)          │
│  N = 50-100 pasos (difusión clásica)      │
└──────────────────┬───────────────────────┘
                   │ mel-espectrograma o latents limpios
                   ▼
┌──────────────────────────────────────────┐
│     VOCODER (HiFi-GAN / DAC Decoder)     │
│     Mel/latents → waveform 44.1 kHz      │
└──────────────────┬───────────────────────┘
                   │
                   ▼
               Audio WAV

FLUJO TEMPORAL:
═══════════════
  Toda la frase se procesa COMPLETA
  ┌─────────────────────────────────────────┐
  │ paso 1  paso 2  paso 3  ...  paso N     │
  │ ░░░░░░  ▒▒▒▒▒▒  ▓▓▓▓▓▓      ████████  │
  │ ruido   menos   forma        audio      │
  │ puro    ruido   emerge       limpio     │
  └─────────────────────────────────────────┘
  No hay streaming posible hasta completar paso N
```

---

## VS: Comparativa directa

### Generación de audio

```
DUAL-AR                              LLM + DIFUSIÓN
═══════                              ══════════════

Token a token en el tiempo:          Toda la secuencia a la vez:

t=0 → t=1 → t=2 → t=3 →...         [═══════════════════════]
 │     │     │     │                  paso 1: ░░░░░░░░░░░░░░░
 ▼     ▼     ▼     ▼                  paso 2: ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
🔊    🔊    🔊    🔊                  ...
(puedes escuchar                      paso N: █████████████████
 desde t=0)                                   🔊 (todo de golpe)
```

### Rendimiento

| Métrica | Dual-AR (Fish Audio) | LLM + Difusión |
|---------|---------------------|----------------|
| **TTFA** (tiempo al primer audio) | <100ms | 500ms–2s (mínimo) |
| **RTF** (Real-Time Factor) | 0.195 (5x tiempo real) | 0.3–1.5 (depende de pasos) |
| **Streaming** | Nativo (token a token) | No posible (genera todo o nada) |
| **Latencia total frase corta** | ~200ms | ~800ms–3s |
| **Latencia total frase larga** | Escala linealmente | Mejor relativo (paralelismo) |

### Calidad de audio

| Aspecto | Dual-AR | LLM + Difusión |
|---------|---------|----------------|
| **Naturalidad general** | Excelente (0.515 Turing) | Excelente (comparable) |
| **Frecuencias altas** | Limitado por codec (DAC) | Superior (difusión modela el continuo) |
| **Suavidad / continuidad** | Puede tener micro-artefactos entre frames | Más suave (genera todo junto) |
| **Consistencia de largo** | Puede degradar en textos muy largos | Más consistente (ve toda la secuencia) |
| **Diversidad entre tomas** | Baja (más determinista) | Alta (ruido inicial aleatorio) |

### Robustez

| Problema | Dual-AR | LLM + Difusión |
|----------|---------|----------------|
| **Repetición de palabras** | Posible (error acumulativo AR) | No ocurre (genera todo de una) |
| **Saltar palabras** | Posible (atención pierde alineación) | No ocurre (duration predictor) |
| **Alucinaciones** | Posible (como cualquier LLM) | Muy raro |
| **Estabilidad en frases largas** | Degradación gradual | Estable (pero necesita más memoria) |

### Infraestructura y escalado

| Aspecto | Dual-AR | LLM + Difusión |
|---------|---------|----------------|
| **Serving** | SGLang, vLLM, TensorRT-LLM (todo el stack LLM) | Custom (no hay estándar) |
| **KV Cache** | Sí (reduce cómputo redundante) | No aplica |
| **Prefix caching** | Sí (86%+ hit rate para misma voz) | Parcial (solo el encoder) |
| **Continuous batching** | Sí (nativo) | Difícil (secuencias de distinta longitud) |
| **GPU memory** | Proporcional a longitud actual | Proporcional a longitud TOTAL × pasos |
| **Multi-GPU** | Tensor parallelism estándar | Más complejo |

### Control y flexibilidad

| Aspecto | Dual-AR | LLM + Difusión |
|---------|---------|----------------|
| **Control de emoción** | Via tokens inline en el LLM | Via CFG (Classifier-Free Guidance) |
| **Intensidad del control** | Binario (token presente o no) | Continuo (escala de CFG: 1.0–15.0) |
| **Voice cloning** | Speaker tokens + referencia en contexto | Speaker embedding + cross-attention |
| **Duración** | Implícita (el AR para cuando termina) | Explícita (duration predictor la decide) |
| **Pitch/energía** | Implícito en los codebooks | Se puede condicionar explícitamente |

---

## Cuándo elegir cada uno

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  DUAL-AR (Fish Audio style)                              │
│  ══════════════════════════                               │
│                                                          │
│  ✓ Asistentes de voz en tiempo real                      │
│  ✓ Chatbots con respuesta instantánea                    │
│  ✓ Streaming de audio (podcast en vivo, radio)           │
│  ✓ Aplicaciones con latencia crítica (<100ms TTFA)       │
│  ✓ Producción a escala (infra LLM madura)                │
│  ✓ Cuando ya tienes infra de LLMs (SGLang, vLLM)        │
│                                                          │
│  ✗ Textos muy largos sin streaming                       │
│  ✗ Cuando necesitas variación natural entre tomas        │
│                                                          │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                                                          │
│  LLM + DIFUSIÓN                                          │
│  ══════════════                                          │
│                                                          │
│  ✓ Audiobooks (calidad máxima, sin prisa)                │
│  ✓ Doblaje de películas/series                           │
│  ✓ Producción musical / canto                            │
│  ✓ Cuando necesitas máxima calidad en freq. altas        │
│  ✓ Generación con variaciones (varias tomas)             │
│  ✓ Control fino de intensidad (CFG escalable)            │
│  ✓ Robustez total (sin repeticiones/saltos)              │
│                                                          │
│  ✗ Streaming en tiempo real                              │
│  ✗ Latencia crítica                                      │
│  ✗ Batching masivo en producción                         │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Arquitectura híbrida: lo mejor de ambos mundos

El enfoque más potente combina ambos. De hecho, Tortoise TTS y XTTS ya hacen esto parcialmente:

```
Texto + Audio referencia
         │
         ▼
┌────────────────────────────┐
│  LLM Backbone (Qwen3-4B)  │
│  Genera tokens semánticos  │
│  con Slow AR (streaming)   │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  Buffer: acumula N frames  │
│  de tokens semánticos      │
└────────────┬───────────────┘
             │ cada N frames
             ▼
┌────────────────────────────┐
│  Mini-Difusión (5-10 pasos)│
│  Flow matching sobre chunk │
│  Semántico → Acústico      │
│  (reemplaza al Fast AR)    │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  Vocoder → Audio chunk     │
│  (streaming por chunks)    │
└────────────┬───────────────┘
             │
             ▼
         🔊 Audio
         (semi-streaming,
          latencia ~200-300ms)

VENTAJAS:
  • Streaming (del LLM AR)
  • Calidad de difusión (en frecuencias altas)
  • Sin repeticiones (difusión por chunks)
  • Latencia aceptable (~200-300ms)
```

---

## Resumen ejecutivo

```
                    STREAMING    CALIDAD     ROBUSTEZ    INFRA
                    ═════════    ═══════     ════════    ═════

Dual-AR             ★★★★★        ★★★★☆       ★★★☆☆      ★★★★★
(Fish Audio)        nativo       muy alta    err. acum.  SGLang

LLM + Difusión      ★☆☆☆☆        ★★★★★       ★★★★★      ★★☆☆☆
                    imposible    la mejor    perfecta    custom

Híbrido             ★★★★☆        ★★★★★       ★★★★☆      ★★★☆☆
(AR + difusión)     por chunks   excelente   muy buena   medio
```

La elección depende del caso de uso. Para **nuestro proyecto** la pregunta clave es:
¿Priorizamos streaming/latencia o calidad offline máxima?
