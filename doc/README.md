# Documentación: Modelos de IA para Generación de Voz

Esta carpeta contiene documentación técnica sobre los principales modelos de inteligencia artificial para síntesis de voz (Text-to-Speech / TTS) y su arquitectura.

## Índice

1. [Panorama General](01-panorama-general.md) — Evolución y clasificación de modelos TTS
2. [Arquitecturas Clásicas](02-arquitecturas-clasicas.md) — Tacotron, Tacotron 2, FastSpeech, WaveNet
3. [Arquitecturas Modernas](03-arquitecturas-modernas.md) — VITS, Bark, Tortoise TTS, XTTS, Valle
4. [Comparativa](04-comparativa.md) — Tabla comparativa de modelos, trade-offs y casos de uso
5. **[Fish Audio](05-fish-audio.md)** — Análisis completo: Dual-AR, GRPO, S2 Pro (SOTA)
6. [Dual-AR vs LLM+Difusión](06-dual-ar-vs-diffusion-llm.md) — VS completo: streaming, calidad, robustez, híbrido
7. **[Arquitectura del Sistema](07-arquitectura-sistema.md)** — Decisiones de diseño, componentes, stack, roadmap
8. **[Pipeline de Anotación](08-pipeline-anotacion.md)** — Router aLLM: SenseVoice + Emotion2Vec (bulk) → GLM-4-Voice (casos complejos)
9. **[Codecs Neurales de Audio](09-codecs-audio.md)** — Comparativa de codecs, Mimi como mini modelo para análisis de ondas
10. **[Plan V0.1](10-plan-v01.md)** — Plan completo de desarrollo: fases, arquitectura, modelo, entrenamiento, inferencia
