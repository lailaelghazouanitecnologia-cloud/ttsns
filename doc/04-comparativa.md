# Comparativa de Modelos TTS

## Tabla comparativa

| Modelo | Año | Tipo | Velocidad | Calidad (MOS) | Zero-shot Cloning | Multilingüe | Open Source |
|--------|-----|------|-----------|---------------|-------------------|-------------|-------------|
| Tacotron 2 | 2018 | AR seq2seq + vocoder | Lento | ~4.5 | No | No | Sí |
| FastSpeech 2 | 2021 | NAR Transformer + vocoder | Muy rápido | ~4.0 | No | No | Sí |
| VITS | 2021 | VAE + Flow + GAN | Rápido | ~4.4 | Con fine-tune | Sí (con datos) | Sí |
| Tortoise TTS | 2022 | AR GPT + Difusión | Muy lento | ~4.5 | Sí | Limitado | Sí |
| Bark | 2023 | 3x AR Transformer + EnCodec | Medio | ~4.2 | Sí (prompts) | Sí (13+ idiomas) | Sí |
| VALL-E | 2023 | AR + NAR + EnCodec | Medio | ~4.5 | Sí (3 seg) | No | No |
| XTTS v2 | 2023 | GPT + Difusión + HiFi-GAN | Medio | ~4.3 | Sí (6 seg) | Sí (17+ idiomas) | Sí |
| F5-TTS | 2024 | Flow Matching + DiT | Rápido | ~4.4 | Sí | Sí | Sí |

## Criterios de selección según caso de uso

### Producción en tiempo real (streaming, asistentes)
- **Recomendado**: VITS, FastSpeech 2 + HiFi-GAN
- **Por qué**: Baja latencia, generación paralela

### Máxima calidad (audiobooks, doblaje)
- **Recomendado**: Tortoise TTS, XTTS v2
- **Por qué**: Alta naturalidad, clonación de voz precisa

### Clonación de voz zero-shot
- **Recomendado**: XTTS v2, F5-TTS, VALL-E
- **Por qué**: Requieren solo segundos de audio de referencia

### Expresividad (emociones, risas, efectos)
- **Recomendado**: Bark
- **Por qué**: Genera expresiones no-verbales de forma nativa

### Multilingüe
- **Recomendado**: XTTS v2, Bark
- **Por qué**: Soporte amplio de idiomas, cross-lingual voice cloning

### Recursos limitados (edge, móvil)
- **Recomendado**: VITS (cuantizado), FastSpeech 2 + MB-MelGAN
- **Por qué**: Modelos pequeños, inferencia eficiente

## Trade-offs fundamentales

```
Calidad ◄────────────────────► Velocidad
   │                               │
   │  Tortoise, VALL-E             │  FastSpeech 2
   │  (lento, alta calidad)        │  (rápido, menos natural)
   │                               │
   │         VITS, F5-TTS          │
   │       (equilibrio)            │
   │                               │

Flexibilidad ◄──────────────► Simplicidad
   │                               │
   │  Bark (expresivo,             │  VITS (un solo modelo,
   │   impredecible)               │   predecible)
   │                               │

Control ◄───────────────────► Naturalidad
   │                               │
   │  FastSpeech 2 (pitch,         │  Bark (natural pero
   │   duración, energía)          │   poco controlable)
```

## Frameworks y herramientas

| Framework | Modelos soportados | Lenguaje |
|-----------|--------------------|----------|
| Coqui TTS | VITS, Tacotron 2, XTTS, FastSpeech 2, + más | Python |
| ESPnet | Tacotron 2, FastSpeech 2, VITS, JETS | Python |
| Fairseq | Modelos de Meta (research) | Python |
| PaddleSpeech | FastSpeech 2, SpeedySpeech | Python |
| TensorFlowTTS | Tacotron 2, FastSpeech 2, MelGAN | Python |
