"""Audio Analyzer — Classifies audio quality and noise before processing.

This is the FIRST step in the pipeline. Before tokenizing or doing anything
with audio, we need to know what we're dealing with.

Classification categories:
- clean: SNR > 30dB, ready to use
- mild_noise: SNR 20-30dB, light denoising recommended
- moderate_noise: SNR 10-20dB, aggressive denoising needed
- heavy_noise: SNR < 10dB, may not be usable
- music_background: music detected, source separation needed
- multiple_speakers: more than one speaker, diarization needed
- non_speech: no speech detected, reject
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import torch
import torchaudio


class NoiseCategory(str, Enum):
    CLEAN = "clean"
    MILD_NOISE = "mild_noise"
    MODERATE_NOISE = "moderate_noise"
    HEAVY_NOISE = "heavy_noise"
    NON_SPEECH = "non_speech"


class AudioIssue(str, Enum):
    CLIPPING = "clipping"
    LOW_VOLUME = "low_volume"
    DC_OFFSET = "dc_offset"
    SHORT_DURATION = "short_duration"


@dataclass
class AnalysisResult:
    """Complete analysis of an audio file."""

    # Basic info
    sample_rate: int
    duration_seconds: float
    num_channels: int

    # Noise classification
    noise_category: NoiseCategory
    estimated_snr_db: float

    # Voice Activity Detection
    speech_ratio: float  # 0.0 - 1.0, proportion of audio that is speech
    speech_segments: list[tuple[float, float]] = field(default_factory=list)  # (start, end) in seconds

    # Issues detected
    issues: list[AudioIssue] = field(default_factory=list)

    # Recommendations
    needs_denoising: bool = False
    usable: bool = True
    rejection_reason: str | None = None

    def summary(self) -> str:
        status = "USABLE" if self.usable else f"REJECTED ({self.rejection_reason})"
        lines = [
            f"[{status}] {self.noise_category.value}",
            f"  Duration: {self.duration_seconds:.1f}s | SR: {self.sample_rate} Hz | SNR: {self.estimated_snr_db:.1f} dB",
            f"  Speech: {self.speech_ratio:.0%} of audio",
        ]
        if self.issues:
            lines.append(f"  Issues: {', '.join(i.value for i in self.issues)}")
        if self.needs_denoising:
            lines.append(f"  → Denoising recommended")
        return "\n".join(lines)


class AudioAnalyzer:
    """Analyzes audio quality and classifies noise level.

    This runs BEFORE any preprocessing. It tells us what we're dealing with
    so we can decide how to process it (or reject it).
    """

    def __init__(
        self,
        snr_clean_threshold: float = 30.0,
        snr_mild_threshold: float = 20.0,
        snr_moderate_threshold: float = 10.0,
        vad_energy_threshold: float = 0.01,
        min_duration: float = 0.5,
        max_clipping_ratio: float = 0.01,
        min_speech_ratio: float = 0.1,
    ):
        self.snr_clean = snr_clean_threshold
        self.snr_mild = snr_mild_threshold
        self.snr_moderate = snr_moderate_threshold
        self.vad_energy_threshold = vad_energy_threshold
        self.min_duration = min_duration
        self.max_clipping_ratio = max_clipping_ratio
        self.min_speech_ratio = min_speech_ratio

    def analyze(self, audio_path: str | Path) -> AnalysisResult:
        """Analyze an audio file and return classification results."""
        audio_path = Path(audio_path)
        waveform, sr = torchaudio.load(str(audio_path))
        return self.analyze_waveform(waveform, sr)

    def analyze_waveform(self, waveform: torch.Tensor, sample_rate: int) -> AnalysisResult:
        """Analyze a waveform tensor directly.

        Args:
            waveform: (channels, samples) tensor
            sample_rate: sample rate in Hz
        """
        # Convert to mono for analysis
        if waveform.shape[0] > 1:
            mono = waveform.mean(dim=0)
        else:
            mono = waveform.squeeze(0)

        duration = mono.shape[0] / sample_rate
        issues: list[AudioIssue] = []

        # Check duration
        if duration < self.min_duration:
            issues.append(AudioIssue.SHORT_DURATION)

        # Check for clipping
        clipping_ratio = self._detect_clipping(mono)
        if clipping_ratio > self.max_clipping_ratio:
            issues.append(AudioIssue.CLIPPING)

        # Check for DC offset
        dc_offset = mono.mean().abs().item()
        if dc_offset > 0.05:
            issues.append(AudioIssue.DC_OFFSET)

        # Check volume
        rms = torch.sqrt(torch.mean(mono**2)).item()
        if rms < 0.001:
            issues.append(AudioIssue.LOW_VOLUME)

        # Energy-based VAD
        speech_segments, speech_ratio = self._energy_vad(mono, sample_rate)

        # Estimate SNR
        snr_db = self._estimate_snr(mono, sample_rate, speech_segments)

        # Classify noise
        noise_category = self._classify_noise(snr_db, speech_ratio)

        # Determine usability
        usable = True
        rejection_reason = None

        if speech_ratio < self.min_speech_ratio:
            usable = False
            rejection_reason = f"Too little speech ({speech_ratio:.0%})"
            noise_category = NoiseCategory.NON_SPEECH

        if duration < self.min_duration:
            usable = False
            rejection_reason = f"Too short ({duration:.1f}s)"

        if AudioIssue.LOW_VOLUME in issues and rms < 0.0001:
            usable = False
            rejection_reason = "Nearly silent"

        needs_denoising = noise_category in (
            NoiseCategory.MILD_NOISE,
            NoiseCategory.MODERATE_NOISE,
            NoiseCategory.HEAVY_NOISE,
        )

        return AnalysisResult(
            sample_rate=sample_rate,
            duration_seconds=duration,
            num_channels=waveform.shape[0],
            noise_category=noise_category,
            estimated_snr_db=snr_db,
            speech_ratio=speech_ratio,
            speech_segments=speech_segments,
            issues=issues,
            needs_denoising=needs_denoising,
            usable=usable,
            rejection_reason=rejection_reason,
        )

    def _detect_clipping(self, mono: torch.Tensor) -> float:
        """Detect proportion of samples at max amplitude (clipping)."""
        threshold = 0.99
        clipped = (mono.abs() > threshold).sum().item()
        return clipped / mono.shape[0]

    def _energy_vad(
        self,
        mono: torch.Tensor,
        sample_rate: int,
        frame_ms: int = 30,
    ) -> tuple[list[tuple[float, float]], float]:
        """Simple energy-based Voice Activity Detection.

        Returns speech segments and speech ratio.
        Uses short-time energy in frames.
        """
        frame_size = int(sample_rate * frame_ms / 1000)
        num_frames = mono.shape[0] // frame_size

        if num_frames == 0:
            return [], 0.0

        # Compute energy per frame
        frames = mono[: num_frames * frame_size].reshape(num_frames, frame_size)
        energies = torch.mean(frames**2, dim=1)

        # Adaptive threshold: use a fraction of the mean energy of top frames
        sorted_energies = torch.sort(energies, descending=True).values
        top_k = max(1, num_frames // 10)  # top 10%
        adaptive_threshold = sorted_energies[:top_k].mean().item() * 0.1
        threshold = max(adaptive_threshold, self.vad_energy_threshold**2)

        is_speech = energies > threshold

        # Convert frame indices to time segments
        segments: list[tuple[float, float]] = []
        in_segment = False
        seg_start = 0.0

        for i, speech in enumerate(is_speech):
            t = i * frame_ms / 1000
            if speech and not in_segment:
                seg_start = t
                in_segment = True
            elif not speech and in_segment:
                segments.append((seg_start, t))
                in_segment = False

        if in_segment:
            segments.append((seg_start, num_frames * frame_ms / 1000))

        # Merge close segments (gap < 300ms)
        merged = self._merge_segments(segments, max_gap=0.3)

        speech_frames = is_speech.sum().item()
        speech_ratio = speech_frames / num_frames if num_frames > 0 else 0.0

        return merged, speech_ratio

    def _merge_segments(
        self, segments: list[tuple[float, float]], max_gap: float
    ) -> list[tuple[float, float]]:
        """Merge speech segments that are close together."""
        if not segments:
            return []

        merged = [segments[0]]
        for start, end in segments[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= max_gap:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
        return merged

    def _estimate_snr(
        self,
        mono: torch.Tensor,
        sample_rate: int,
        speech_segments: list[tuple[float, float]],
    ) -> float:
        """Estimate Signal-to-Noise Ratio in dB.

        Uses speech segments as signal and non-speech as noise.
        If no speech segments, estimates from overall statistics.
        """
        if not speech_segments:
            # No speech detected — estimate from signal statistics
            rms = torch.sqrt(torch.mean(mono**2)).item()
            if rms < 1e-8:
                return 0.0
            # Use the bottom 10% of frame energies as noise estimate
            frame_size = int(sample_rate * 0.03)
            num_frames = mono.shape[0] // frame_size
            if num_frames < 2:
                return 0.0
            frames = mono[: num_frames * frame_size].reshape(num_frames, frame_size)
            energies = torch.mean(frames**2, dim=1)
            sorted_e = torch.sort(energies).values
            noise_energy = sorted_e[: max(1, num_frames // 10)].mean().item()
            signal_energy = energies.mean().item()
            if noise_energy < 1e-10:
                return 60.0  # effectively no noise
            return float(10 * np.log10(signal_energy / noise_energy))

        # Use speech segments for signal, non-speech for noise
        speech_samples = []
        noise_samples = []

        total_samples = mono.shape[0]
        speech_mask = torch.zeros(total_samples, dtype=torch.bool)

        for start, end in speech_segments:
            s = int(start * sample_rate)
            e = min(int(end * sample_rate), total_samples)
            speech_mask[s:e] = True

        speech_audio = mono[speech_mask]
        noise_audio = mono[~speech_mask]

        if speech_audio.numel() < sample_rate * 0.1:  # less than 100ms of speech
            return 0.0

        signal_power = torch.mean(speech_audio**2).item()

        if noise_audio.numel() < sample_rate * 0.05:  # less than 50ms of noise
            return 60.0  # very little noise

        noise_power = torch.mean(noise_audio**2).item()

        if noise_power < 1e-10:
            return 60.0

        return float(10 * np.log10(signal_power / noise_power))

    def _classify_noise(self, snr_db: float, speech_ratio: float) -> NoiseCategory:
        """Classify noise level based on SNR and speech ratio."""
        if speech_ratio < self.min_speech_ratio:
            return NoiseCategory.NON_SPEECH

        if snr_db >= self.snr_clean:
            return NoiseCategory.CLEAN
        elif snr_db >= self.snr_mild:
            return NoiseCategory.MILD_NOISE
        elif snr_db >= self.snr_moderate:
            return NoiseCategory.MODERATE_NOISE
        else:
            return NoiseCategory.HEAVY_NOISE
