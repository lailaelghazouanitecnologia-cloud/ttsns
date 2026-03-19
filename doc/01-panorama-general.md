# Panorama General: Modelos de IA para Generación de Voz

## Evolución histórica

| Etapa | Periodo | Enfoque |
|-------|---------|---------|
| Concatenativa | 1990s–2000s | Unión de fragmentos de audio pregrabados |
| Paramétrica (HMM) | 2000s–2015 | Modelos ocultos de Markov para generar parámetros acústicos |
| Neuronal (seq2seq) | 2016–2020 | Redes neuronales encoder-decoder (Tacotron, DeepVoice) |
| End-to-end | 2020–presente | Modelos que generan audio directamente desde texto (VITS, Bark) |

## Clasificación de modelos actuales

### Por tipo de salida

- **Modelos de espectrograma + vocoder**: Generan un mel-espectrograma intermedio y luego un vocoder lo convierte a forma de onda (ej. Tacotron 2 + WaveGlow).
- **Modelos end-to-end**: Generan la forma de onda de audio directamente desde texto (ej. VITS).
- **Modelos basados en códigos de audio (codec)**: Predicen tokens discretos de un codec neuronal de audio (ej. Bark, VALLE).

### Por paradigma de entrenamiento

- **Autoregresivos (AR)**: Generan la salida token a token, condicionados en los anteriores. Alta calidad, más lentos (ej. Tacotron, Tortoise TTS).
- **No autoregresivos (NAR)**: Generan toda la salida en paralelo. Más rápidos, requieren técnicas especiales para mantener calidad (ej. FastSpeech 2).
- **Híbridos (AR + NAR)**: Combinan etapas autoregresivas y paralelas (ej. VALLE, VITS).
- **Basados en difusión**: Usan procesos de difusión para generar audio de alta fidelidad (ej. Grad-TTS, DiffWave).

## Pipeline típico de un sistema TTS neuronal

```
Texto → [Preprocesamiento / Normalización]
     → [Encoder de texto]
     → [Modelo acústico (genera representación intermedia)]
     → [Vocoder / Decoder de audio]
     → Audio (waveform)
```

### Componentes clave

1. **Front-end de texto**: Normalización, tokenización, conversión grafema-a-fonema (G2P).
2. **Encoder**: Transforma la secuencia de tokens en representaciones latentes. Suele usar Transformers o RNNs.
3. **Modelo acústico**: Predice representaciones acústicas (mel-espectrogramas, tokens de codec). Es el núcleo del sistema.
4. **Vocoder**: Convierte la representación acústica en forma de onda audible. Ejemplos: WaveNet, WaveGlow, HiFi-GAN, UnivNet.

## Retos principales

- **Naturalidad y prosodia**: Generar habla que suene natural con entonación y ritmo correctos.
- **Clonación de voz (zero-shot / few-shot)**: Replicar una voz con pocos segundos de referencia.
- **Multilingüismo**: Soporte para múltiples idiomas y cambios de idioma dentro de una misma frase.
- **Velocidad de inferencia**: Lograr tiempo real o más rápido en dispositivos variados.
- **Control emocional y estilístico**: Permitir al usuario controlar el tono, emoción y estilo del habla.
