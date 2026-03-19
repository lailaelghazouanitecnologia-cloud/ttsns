# Arquitecturas Clásicas de TTS Neuronal

## 1. WaveNet (DeepMind, 2016)

### Arquitectura

```
              ┌─────────────────────────┐
              │   Condicionamiento      │
              │   (texto, speaker id)   │
              └───────────┬─────────────┘
                          │
              ┌───────────▼─────────────┐
              │  Capas causales         │
              │  dilatadas (dilated     │
              │  causal convolutions)   │
              │                         │
              │  x1, x2, x4, x8, ...   │
              │  (campo receptivo       │
              │   exponencialmente      │
              │   creciente)            │
              └───────────┬─────────────┘
                          │
              ┌───────────▼─────────────┐
              │  Conexiones residuales  │
              │  + skip connections     │
              └───────────┬─────────────┘
                          │
              ┌───────────▼─────────────┐
              │  Softmax sobre μ-law    │
              │  (256 valores)          │
              └───────────┬─────────────┘
                          │
                    Muestra de audio
```

- **Tipo**: Modelo autoregresivo a nivel de muestra de audio.
- **Innovación**: Convoluciones causales dilatadas para capturar dependencias de largo alcance en audio raw.
- **Limitación**: Extremadamente lento en inferencia (genera muestra a muestra a 16 kHz+).
- **Uso actual**: Principalmente como vocoder condicionado, no como sistema TTS completo.

---

## 2. Tacotron (Google, 2017)

### Arquitectura

```
Texto (caracteres)
       │
       ▼
┌──────────────┐
│  Embedding   │
│  de caracteres│
└──────┬───────┘
       ▼
┌──────────────┐
│  Encoder     │
│  (CBHG:      │
│   Conv Bank + │
│   Highway +   │
│   Bi-GRU)     │
└──────┬───────┘
       ▼
┌──────────────┐
│  Attention   │
│  (Bahdanau)  │
└──────┬───────┘
       ▼
┌──────────────┐
│  Decoder     │
│  (GRU +      │
│   reducción) │
└──────┬───────┘
       ▼
Mel-espectrograma
       │
       ▼
┌──────────────┐
│  Griffin-Lim │
│  (vocoder)   │
└──────────────┘
       │
       ▼
   Audio WAV
```

- **Tipo**: Seq2seq con atención, genera mel-espectrogramas.
- **Innovación**: Primera arquitectura neuronal end-to-end que logra TTS de calidad desde caracteres.
- **Limitación**: Griffin-Lim produce audio de calidad limitada. Problemas de alineación (skipping, repeticiones).

---

## 3. Tacotron 2 (Google, 2018)

### Arquitectura

```
Texto (caracteres)
       │
       ▼
┌──────────────────┐
│  Embedding       │
└──────┬───────────┘
       ▼
┌──────────────────┐
│  Encoder         │
│  (3x Conv1D +    │
│   Bi-LSTM)       │
└──────┬───────────┘
       ▼
┌──────────────────┐
│  Location-       │
│  Sensitive       │
│  Attention       │
└──────┬───────────┘
       ▼
┌──────────────────┐
│  Decoder         │
│  (2x LSTM +      │
│   Pre-Net +      │
│   Post-Net)      │
└──────┬───────────┘
       ▼
  Mel-espectrograma
       │
       ▼
┌──────────────────┐
│  WaveNet vocoder │
│  (modificado)    │
└──────────────────┘
       │
       ▼
   Audio WAV
```

- **Mejoras sobre Tacotron**: Location-sensitive attention (más robusta), arquitectura simplificada, uso de WaveNet como vocoder.
- **Calidad**: MOS cercano a habla humana (~4.5/5).
- **Limitación**: Autoregresivo → lento. Vocoder WaveNet también lento.

---

## 4. FastSpeech (Microsoft, 2019) y FastSpeech 2 (2021)

### Arquitectura de FastSpeech 2

```
Texto (fonemas)
       │
       ▼
┌──────────────────────┐
│  Encoder             │
│  (Transformer FFT    │
│   blocks)            │
└──────┬───────────────┘
       │
       ├──► Duration Predictor ──► Regulador de longitud
       ├──► Pitch Predictor
       └──► Energy Predictor
                │
                ▼
┌──────────────────────┐
│  Variance Adaptor    │
│  (expande secuencia  │
│   según duración)    │
└──────┬───────────────┘
       ▼
┌──────────────────────┐
│  Decoder             │
│  (Transformer FFT    │
│   blocks)            │
└──────┬───────────────┘
       ▼
  Mel-espectrograma
       │
       ▼
┌──────────────────────┐
│  Vocoder paralelo    │
│  (HiFi-GAN, etc.)   │
└──────────────────────┘
       │
       ▼
   Audio WAV
```

- **Tipo**: No autoregresivo (paralelo).
- **Innovación**: Predictor de duración explícito permite generación paralela. FastSpeech 2 añade predictores de pitch y energía.
- **Ventaja**: ~100x más rápido que Tacotron 2 en inferencia.
- **Limitación**: Requiere alineaciones de duración precomputadas para entrenamiento (knowledge distillation o MFA).

---

## 5. Vocoders neuronales relevantes

| Vocoder | Tipo | Velocidad | Calidad | Año |
|---------|------|-----------|---------|-----|
| WaveNet | AR convolucional | Muy lento | Excelente | 2016 |
| WaveRNN | AR recurrente | Lento | Muy buena | 2018 |
| WaveGlow | Flow-based (paralelo) | Rápido | Muy buena | 2019 |
| MelGAN | GAN (paralelo) | Muy rápido | Buena | 2019 |
| HiFi-GAN | GAN (paralelo) | Muy rápido | Excelente | 2020 |
| UnivNet | GAN (paralelo) | Muy rápido | Excelente | 2022 |

HiFi-GAN es actualmente el vocoder más usado por su equilibrio entre velocidad y calidad.
