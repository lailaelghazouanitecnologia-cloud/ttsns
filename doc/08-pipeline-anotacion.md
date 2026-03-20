# Pipeline de Anotación de Audio

> La anotación es el componente más crítico del sistema.
> Sin datos bien anotados, ningún modelo TTS produce voz natural.
> Arquitectura: pipeline ligero para bulk + aLLM para casos complejos.

---

## Por qué la anotación es lo primero

```
Datos crudos → ANOTACIÓN → Datos de entrenamiento → Modelo TTS
                  ↑
            ESTO decide la
            calidad de TODO
            lo que viene después
```

Un modelo TTS necesita saber **más que el texto**:
- **Emoción** del hablante (alegría, tristeza, enfado, neutral...)
- **Eventos no-verbales** (risa, llanto, suspiros, respiración)
- **Prosodia** (énfasis, pausas, velocidad, entonación)
- **Calidad del audio** (ruido, reverberación, clipping)
- **Idioma/code-switching** (cambios de idioma mid-frase)

Si la anotación es pobre, el modelo aprende prosodia plana y genérica.
Si la anotación es rica, el modelo aprende a **actuar**.

---

## Arquitectura: Router aLLM

```
                        Audio de entrada
                              │
                              ▼
                    ┌───────────────────┐
                    │   Audio Analyzer   │
                    │   (doc 07, Fase 1) │
                    │   SNR, VAD, ruido  │
                    └─────────┬─────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │        PIPELINE LIGERO        │
               │        (Opción B)             │
               │                               │
               │  ┌─────────────────────────┐  │
               │  │  SenseVoice (234M)       │  │
               │  │  ─────────────────────   │  │
               │  │  • ASR (transcripción)   │  │
               │  │  • Emoción básica        │  │
               │  │    (happy/sad/angry/     │  │
               │  │     neutral)             │  │
               │  │  • Detección de eventos  │  │
               │  │    (risa, llanto, tos,   │  │
               │  │     aplausos, música)    │  │
               │  │  • Idioma               │  │
               │  │                          │  │
               │  │  VRAM: <0.5GB            │  │
               │  │  Latencia: 70ms/10s      │  │
               │  └────────────┬────────────┘  │
               │               │               │
               │  ┌────────────▼────────────┐  │
               │  │  Emotion2Vec (90-300M)   │  │
               │  │  ─────────────────────   │  │
               │  │  • 9 emociones finas     │  │
               │  │    (angry, disgusted,    │  │
               │  │     fearful, happy,      │  │
               │  │     neutral, sad,        │  │
               │  │     surprised, other,    │  │
               │  │     unknown)             │  │
               │  │  • Embeddings de emoción │  │
               │  │  • Confianza por clase   │  │
               │  │                          │  │
               │  │  VRAM: <0.3GB            │  │
               │  │  SOTA en IEMOCAP         │  │
               │  └────────────┬────────────┘  │
               │               │               │
               └───────────────┼───────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   CONFIDENCE CHECK   │
                    │   ─────────────────  │
                    │                      │
                    │   ¿SenseVoice y      │
                    │   Emotion2Vec        │
                    │   están de acuerdo?  │
                    │                      │
                    │   ¿Confianza > θ?    │
                    │                      │
                    │   ¿Caso simple?      │
                    │                      │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                SÍ: aceptar          NO: escalar
                anotación             al aLLM
                    │                     │
                    ▼                     ▼
          ┌─────────────────┐   ┌─────────────────────┐
          │  Anotación       │   │  GLM-4-Voice (9B)    │
          │  directa         │   │  ─────────────────   │
          │                  │   │                      │
          │  → dataset       │   │  • Comprensión       │
          │                  │   │    profunda de tono   │
          │  (90% de los     │   │  • Detecta sarcasmo, │
          │   audios)        │   │    ironía, duda       │
          │                  │   │  • Entiende contexto  │
          └─────────────────┘   │    conversacional     │
                                │  • Emoción compuesta  │
                                │    (triste pero       │
                                │    intentando sonar   │
                                │    alegre)            │
                                │                      │
                                │  VRAM: ~12GB INT4     │
                                │                      │
                                │  (10% de los audios)  │
                                └──────────┬──────────┘
                                           │
                                           ▼
                                 ┌─────────────────┐
                                 │  Anotación       │
                                 │  enriquecida     │
                                 │                  │
                                 │  → dataset       │
                                 └─────────────────┘
```

---

## Cuándo escalar al aLLM (GLM-4-Voice)

El router decide escalar cuando:

| Señal | Ejemplo | Por qué |
|-------|---------|---------|
| **Baja confianza de Emotion2Vec** | Confianza < 0.6 en todas las clases | El modelo no está seguro |
| **Conflicto SenseVoice ↔ Emotion2Vec** | SenseVoice dice "neutral" pero Emotion2Vec dice "sad" | Desacuerdo entre modelos |
| **Emoción compuesta** | Scores similares en 2+ emociones (ej: happy=0.4, sad=0.35) | No es una emoción simple |
| **Eventos no-verbales ambiguos** | Risa que podría ser nerviosa o genuina | SenseVoice no distingue tipo de risa |
| **Code-switching detectado** | Cambio de idioma mid-frase | Prosodia cambia con el idioma |
| **Audio largo (>30s)** | Monólogo con cambios emocionales | La emoción puede cambiar durante el audio |
| **Prosodia inusual** | Pitch variance alta + emoción "neutral" | Podría ser sarcasmo, pregunta retórica |

---

## Formato de anotación

Cada audio produce un registro con esta estructura:

```json
{
  "id": "audio_00142",
  "file": "data/raw/audio_00142.wav",
  "duration_s": 4.7,
  "sample_rate": 44100,

  "transcription": {
    "text": "No, está bien, de verdad",
    "language": "es",
    "confidence": 0.97,
    "source": "sensevoice"
  },

  "emotion": {
    "primary": "sad",
    "primary_confidence": 0.72,
    "secondary": "neutral",
    "secondary_confidence": 0.18,
    "source": "emotion2vec",
    "escalated": false
  },

  "events": [
    {"type": "sigh", "start_s": 0.1, "end_s": 0.4},
    {"type": "pause", "start_s": 2.1, "end_s": 2.6}
  ],

  "prosody": {
    "speaking_rate": "slow",
    "pitch_mean": 142.3,
    "pitch_std": 28.7,
    "energy_profile": "declining"
  },

  "quality": {
    "snr_db": 32.1,
    "noise_class": "clean",
    "clipping": false,
    "reverb": "low"
  },

  "annotation_tier": "pipeline",
  "annotation_version": "v1"
}
```

Cuando el aLLM interviene, se añade:

```json
{
  "emotion": {
    "primary": "resigned_sadness",
    "primary_confidence": 0.85,
    "secondary": "forced_cheerfulness",
    "secondary_confidence": 0.71,
    "source": "glm4voice",
    "escalated": true,
    "escalation_reason": "emotion_conflict",
    "allm_note": "El hablante intenta sonar bien pero su tono descendente y la pausa larga revelan tristeza contenida"
  },
  "annotation_tier": "allm",
  "annotation_version": "v1"
}
```

---

## Modelos: detalle técnico

### SenseVoice-Small (Alibaba / FunAudioLLM)

```
Tipo:           ASR + SER + Event Detection (todo en uno)
Params:         234M
VRAM:           <0.5GB FP16
Inferencia:     70ms por 10s de audio (no-autoregresivo)
Velocidad:      15x más rápido que Whisper-Large
Emociones:      Happy, Sad, Angry, Neutral
Eventos:        Risa, llanto, aplausos, tos, estornudo, respiración, BGM
Idiomas:        50+ (incluyendo ES, EN, ZH)
Licencia:       Open source
HuggingFace:    FunAudioLLM/SenseVoiceSmall
GitHub:         github.com/FunAudioLLM/SenseVoice
```

### Emotion2Vec (standalone)

```
Tipo:           Clasificador de emoción puro (no ASR)
Params:         90M (base) / 300M (large)
VRAM:           <1GB
Emociones:      9 clases: angry, disgusted, fearful, happy, neutral,
                sad, surprised, other, unknown
Precisión:      SOTA en IEMOCAP
Idiomas:        10+ (multilingüe)
Output:         Probabilidades por clase + embeddings
Licencia:       Open source
HuggingFace:    emotion2vec
GitHub:         github.com/ddlBoJack/emotion2vec
```

### GLM-4-Voice (Zhipu AI / THUDM)

```
Tipo:           Audio LLM end-to-end (comprensión + generación de voz)
Params:         9B (LLM) + tokenizer + decoder
VRAM:           ~12GB INT4 / ~20-30GB FP16
Bilingüe:       Chino + Inglés (español vía instrucciones en inglés)
Emoción:        Comprensión profunda: sarcasmo, ironía, emociones
                compuestas, intención vs expresión
Tono:           Control de entonación, velocidad, dialecto
Arquitectura:
  • Voice-Tokenizer: basado en Whisper, 175 bps (ultra compacto)
  • Voice-9B: LLM basado en GLM-4
  • Voice-Decoder: basado en CosyVoice
Licencia:       Open source
HuggingFace:    THUDM/glm-4-voice-9b
GitHub:         github.com/THUDM/GLM-4-Voice
Paper:          arxiv.org/abs/2412.02612
```

---

## Uso de VRAM (RTX 4090, 24GB)

```
Pipeline ligero solo:
  SenseVoice       0.5 GB
  Emotion2Vec      0.3 GB
  ────────────────────────
  Total:           0.8 GB  ← cabe en CUALQUIER GPU
  Libre:          23.2 GB

Pipeline + GLM-4-Voice (cuando escala):
  SenseVoice       0.5 GB
  Emotion2Vec      0.3 GB
  GLM-4-Voice     12.0 GB  (INT4)
  ────────────────────────
  Total:          12.8 GB
  Libre:          11.2 GB  ← sobra para batching

Nota: GLM-4-Voice se carga bajo demanda.
  → Carga lazy: solo se instancia cuando hay escalación.
  → Se puede descargar de VRAM cuando no se usa
    (offload a CPU o liberar).
  → En bulk processing, se procesa primero TODO con
    el pipeline ligero, luego se carga GLM-4-Voice
    para los casos escalados.
```

---

## Pipeline de procesamiento en bulk

```
PASO 1: Pipeline ligero (TODO el dataset)
══════════════════════════════════════════
  • Carga SenseVoice + Emotion2Vec (~0.8GB)
  • Procesa TODOS los audios
  • Genera anotaciones base
  • Marca los que necesitan escalación
  • Velocidad: ~15x tiempo real (miles de horas/día)

PASO 2: Filtrado
══════════════════
  • Separa: anotación_ok vs necesita_escalación
  • Típicamente 85-95% pasan directo
  • 5-15% necesitan el aLLM

PASO 3: aLLM (solo los escalados)
══════════════════════════════════
  • Descarga pipeline ligero (opcional)
  • Carga GLM-4-Voice INT4 (~12GB)
  • Procesa solo los audios escalados
  • Enriquece las anotaciones
  • Más lento pero solo es el 5-15%

PASO 4: Merge y validación
══════════════════════════
  • Une anotaciones de ambos tiers
  • Validación de consistencia
  • Genera estadísticas del dataset
  • Export al formato de entrenamiento
```

---

## Alternativas evaluadas

Además de los modelos elegidos, evaluamos:

| Modelo | Params | Por qué NO lo elegimos (para anotación) |
|--------|--------|----------------------------------------|
| MiniCPM-o 4.5 | 9B | Excelente, pero más orientado a conversación que a anotación batch. Candidato para V2 |
| Moshi | 7B | 70+ emociones pero diseñado para diálogo full-duplex, no batch |
| Qwen2-Audio | 7B | Bueno para SER pero SenseVoice + Emotion2Vec son más rápidos para bulk |
| Phi-4-Multimodal | 5.6B | Mejor ASR pero sin detección de emoción explícita |
| SALMONN | 13B | Emoción + ASR pero más pesado que nuestra combinación |
| Qwen3-Omni | 30B | Demasiado grande para anotación bulk eficiente |
| Gemma 3n | 5-8B | Sin detección de emoción |
| Ultravox | 8B | Sin detección de emoción |

**Decisión**: SenseVoice + Emotion2Vec para velocidad, GLM-4-Voice para profundidad.
El routing inteligente nos da lo mejor de ambos mundos.

---

## Roadmap de anotación

### V1: Pipeline básico
- [ ] SenseVoice: transcripción + emoción básica + eventos
- [ ] Emotion2Vec: clasificación fina de emoción
- [ ] Merge de resultados
- [ ] Formato de salida JSON
- [ ] Script de procesamiento batch

### V2: Router aLLM
- [ ] Lógica de confidence check
- [ ] Integración GLM-4-Voice INT4
- [ ] Escalación automática
- [ ] Merge pipeline + aLLM
- [ ] Métricas de acuerdo inter-modelo

### V3: Prosodia y refinamiento
- [ ] Extracción de pitch/energy/speaking rate
- [ ] Detección de pausas significativas
- [ ] Timeline de emociones para audios largos
- [ ] Considerar MiniCPM-o 4.5 como alternativa/complemento a GLM-4-Voice
- [ ] Human-in-the-loop para validación de casos difíciles

---

## Siguiente paso

Empezar por V1: integrar SenseVoice-Small para transcripción + emoción + eventos.
Es el componente que más datos genera con menos recursos.
