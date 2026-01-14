from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AlignAttConfig:
    eval_data_path: str = "tmp"
    segment_length: float = field(default=1.0, metadata={"help": "in second"})
    frame_threshold: int = 4
    rewind_threshold: int = 200
    audio_max_len: float = 20.0
    cif_ckpt_path: str = ""
    never_fire: bool = False
    language: str = field(default="zh")
    nonspeech_prob: float = 0.5
    audio_min_len: float = 1.0
    decoder_type: Literal["greedy","beam"] = "greedy"
    beam_size: int = 5
    task: Literal["transcribe","translate"] = "transcribe"
    tokenizer_is_multilingual: bool = False
    init_prompt: str = field(default=None)
    static_init_prompt: str = field(default=None)
    max_context_tokens: int = field(default=None)
    # Language identification parameters
    lang_id_confidence_threshold: float = 0.5
    lang_id_ensemble_chunks: int = 3
    lang_id_dynamic_threshold: bool = True
    lang_id_fallback_lang: str = "en"
    lang_id_min_consensus_ratio: float = 0.5
    # Gap analysis parameters (new soft voting algorithm)
    lang_id_min_gap_threshold: float = 0.05      # Minimum 5% gap required
    lang_id_ambiguity_threshold: float = 0.10   # Mark as ambiguous if gap < 10%
    lang_id_use_soft_voting: bool = True        # Use new soft voting algorithm
    # Progressive Detection parameters
    lang_id_progressive_enabled: bool = True    # Enable progressive detection with checkpoints
    lang_id_chunk_duration: float = 2.5         # Duration of each detection chunk in seconds
    lang_id_stage1_time: float = 5.0            # Time for Stage 1 checkpoint (seconds)
    lang_id_stage1_confidence: float = 0.9      # Confidence threshold for Stage 1 early exit
    lang_id_stage2_time: float = 10.0           # Time for Stage 2 checkpoint (seconds)
    lang_id_stage2_gap: float = 0.15            # Gap threshold for Stage 2 acceptance
    lang_id_stage3_time: float = 15.0           # Time for Stage 3 forced decision (seconds)
