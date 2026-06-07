# A Parameter Efficient Domain Adaptation Framework For Speech Recognition using Synthetic Data 

A fully open-source, end-to-end pipeline for adapting OpenAI's Whisper ASR model to domain-specific speech using only synthetic data — no real audio, no manual annotations, no proprietary tools.

Fine-tuned with LoRA adapters across three domains (Music, Weather, Sports), the pipeline outperforms Meta AI's state-of-the-art DAS framework on 2 of 3 domains, evaluated on 5,640 real speech samples from Mozilla Common Voice 17.0.

---

## Overview

The pipeline generates domain-specific training data by injecting seed keywords into Llama3-8B prompts, converting the generated text to speech using Tacotron2-DDC, and fine-tuning only the Whisper decoder with lightweight LoRA adapters. Each domain gets its own modular adapter trained independently over a shared frozen encoder.

Core outcomes:

- 29.3% relative WER reduction on Weather domain (18.30% → 12.94%)
- 19.9% relative WER reduction on Music domain (14.63% → 11.72%)
- 4.4% relative WER reduction on Sports domain (16.61% → 15.88%)
- First fully open-source synthetic-only ASR domain adaptation pipeline
- Empirical finding: lexical alignment matters more than corpus size

---

## Problem

General-purpose ASR models like Whisper are trained on broad, diverse data. While they generalize well, they struggle with domain-specific vocabulary:

- Music: artist names, album titles, genre-specific terms
- Weather: temperature, precipitation, atmospheric vocabulary
- Sports: player names, team names, event-specific stats

Collecting and annotating real domain speech data is expensive and hard to scale. Existing domain adaptation methods either require real audio or rely on proprietary tools unavailable to the broader research community.

---

## Solution

This pipeline adapts Whisper to specific domains using only synthetically generated data:

1. **Keyword Selection** — Extract seed vocabulary from frequently searched domain-specific terms
2. **Text Generation** — Inject keywords into structured Llama3-8B prompts to generate chatbot-style conversational text
3. **Speech Synthesis** — Convert generated text to audio using Tacotron2-DDC
4. **LoRA Fine-tuning** — Train lightweight adapters on the Whisper decoder independently per domain
5. **Evaluation** — Test on real, unseen human speech from Common Voice 17.0

---

## Results

| Domain  | Base WER (%) | LoRA WER (%) | Relative Gain (%) |
|---------|-------------|--------------|-------------------|
| Weather | 18.30       | 12.94        | **+29.3%**        |
| Music   | 14.63       | 11.72        | **+19.9%**        |
| Sports  | 16.61       | 15.88        | **+4.4%**         |

The DAS framework (Meta AI, 2025) reports 10–17% improvement across domains. This pipeline exceeds that range on 2 of 3 domains using entirely open-source tools.

Key empirical finding: vocabulary structural regularity predicts adaptation success better than corpus size. The smallest corpus (Weather, 31K samples) produced the highest WER gain; the largest (Sports, 46K samples) produced the least.

---

## Architecture

```
Domain Keywords (seed vocabulary)
        |
        v
Llama3-8B (local, two-phase keyword-guided prompting)
        |
        v
Domain-specific Text Corpus (chatbot-style Q&A)
        |
        v
Tacotron2-DDC (TTS synthesis, single speaker)
        |
        v
Synthetic Audio-Transcript Pairs
        |
        +------------------+------------------+
        |                  |                  |
   Music (44K)       Weather (31K)      Sports (46K)
        |                  |                  |
        v                  v                  v
   LoRA Adapter      LoRA Adapter       LoRA Adapter
   (music)           (weather)          (sports)
        |                  |                  |
        +------------------+------------------+
                           |
                    Frozen Whisper Encoder
                    (shared, not fine-tuned)
                           |
                    Whisper Decoder
                    (base weights frozen)
                           |
                    Domain-specific Adapter
                    (swapped at inference time)
                           |
                    Transcript Output
                    (evaluated on Common Voice 17.0)
```

---

## Dataset Statistics

| Domain  | Synthetic Train | Total Duration  | Eval Samples (Common Voice 17.0) |
|---------|-----------------|-----------------|----------------------------------|
| Music   | 44,000          | 31h 42m 53s     | 2,470                            |
| Weather | 31,000          | 45h 14m 45s     | 1,670                            |
| Sports  | 46,000          | 37h 36m 06s     | 1,500                            |
| **Total** | **121,000**   | **>114h**       | **5,640**                        |

---

## Key Design Decisions

**Decoder-only fine-tuning**
The Whisper encoder handles acoustic features robustly across domains and is kept fully frozen. Only the decoder — which functions as a conditional language model — is adapted, since domain-specific vocabulary priors live there.

**LoRA over full fine-tuning**
Full decoder fine-tuning causes catastrophic cross-domain regression (e.g., fine-tuning on Music caused a 28.8% WER spike on Sports in the original DAS paper). LoRA adapters (r=32, α=64, ~1.2% extra parameters) avoid this by keeping base weights frozen.

**Keyword-guided prompting over generic metadata**
Instead of generic prompts like "generate music-related sentences", seed vocabulary is explicitly injected into structured prompt fields (use-case, skills, persona, instruction). This grounds the LLM output in real domain vocabulary frequency, particularly for underrepresented terms.

**Modular per-domain adapters**
Each adapter is trained independently with no cross-domain data mixing, making them individually swappable at inference time over the shared encoder.

---

## Tech Stack

| Component         | Tool / Library                         |
|-------------------|----------------------------------------|
| Base ASR Model    | Whisper-base (OpenAI, 74M params)      |
| Text Generation   | Llama3-8B (run locally)                |
| TTS Synthesis     | Tacotron2-DDC                          |
| Fine-tuning       | LoRA via HuggingFace PEFT              |
| Model Management  | HuggingFace Transformers               |
| Optimizer         | AdamW                                  |
| Evaluation Data   | Mozilla Common Voice 17.0              |
| Hardware          | NVIDIA RTX 3050 (consumer GPU)         |
| Language          | Python, PyTorch                        |

---

## LoRA Configuration

| Hyperparameter    | Value                  |
|-------------------|------------------------|
| Rank (r)          | 32                     |
| Alpha (α)         | 64                     |
| Target Modules    | Query and Value attention heads (decoder) |
| Initialization    | PiSSA (Principal Singular Value Decomposition) |
| Scaling           | Rank-Stable Scaling    |
| Extra Parameters  | ~1.2% of base model    |
| Optimizer         | AdamW, lr = 3e-6       |
| Epochs            | 10                     |
| Batch Size        | 16                     |
| Warmup            | Linear, first 10% of steps |

---

## Getting Started

### Prerequisites

- Python 3.9+
- PyTorch (CUDA recommended)
- HuggingFace Transformers and PEFT
- Llama3-8B (local weights)
- Tacotron2-DDC
- Mozilla Common Voice 17.0 (for evaluation)

### Installation

```bash
git clone https://github.com/Janjirala-Srikar/Lora_Whisper.git
cd Lora_Whisper
pip install -r requirements.txt
```

### Pipeline Steps

**Step 1 — Generate synthetic text**
```bash
python generate_text.py --domain music --keywords keywords/music.txt --output data/music/text/
```

**Step 2 — Synthesize speech**
```bash
python synthesize_speech.py --input data/music/text/ --output data/music/audio/
```

**Step 3 — Fine-tune LoRA adapter**
```bash
python train_lora.py --domain music --data data/music/ --output adapters/music/
```

**Step 4 — Evaluate**
```bash
python evaluate.py --domain music --adapter adapters/music/ --eval_data data/eval/music/
```

---

## Project Structure

```
Lora_Whisper/
├── speech_demo/          # Demo scripts and audio samples
├── lora/                 # LoRA adapter weights and configs
├── generate_text.py      # Keyword-guided text generation via Llama3-8B
├── synthesize_speech.py  # TTS synthesis using Tacotron2-DDC
├── train_lora.py         # LoRA fine-tuning on Whisper decoder
├── evaluate.py           # WER evaluation on Common Voice
├── keywords/             # Domain-specific seed vocabulary files
│   ├── music.txt
│   ├── weather.txt
│   └── sports.txt
├── data/                 # Generated synthetic datasets (not tracked)
├── adapters/             # Trained LoRA adapters (not tracked)
└── .gitignore
```

---

## Comparison with DAS (Meta AI, 2025)

| Aspect                  | DAS (Tran et al., 2025)         | This Work                          |
|-------------------------|---------------------------------|------------------------------------|
| LLM for text generation | Llama3-70B (proprietary access) | Llama3-8B (local, open-source)     |
| Prompting strategy      | Generic metadata-driven         | Keyword-injection (two-phase)      |
| TTS system              | Internal Meta TTS               | Tacotron2-DDC (open-source)        |
| Evaluation data         | Meta Ray-Ban Glasses (internal) | Common Voice 17.0 (public)         |
| WER improvement range   | 10–17% across domains           | Up to 29.3% (exceeds on 2/3)       |
| Reproducibility         | Not reproducible (proprietary)  | Fully reproducible, open-source    |
| Hardware                | V100 GPU                        | NVIDIA RTX 3050 (consumer)         |

---

## Limitations and Future Work

**Sports domain regression** — High proper-noun density and temporally volatile vocabulary (player names, event stats) are hard to cover with static synthetic corpora. The Sports adapter showed the least improvement (+4.4%).

**Single speaker TTS** — All synthetic audio uses a single speaker identity to isolate language modeling from acoustic variation. Multi-speaker TTS could improve acoustic robustness.

**Proposed extension: LoJA** — Low-Rank Jacobian Adapter modifies the decoder's Jacobian matrix relative to token predictions, enabling input-conditioned sensitivity at each decoding step — particularly useful for proper-noun-heavy domains where global weight perturbation is insufficient.

Other future directions:
- Extend to medical and legal speech domains
- Experiment with larger Whisper variants
- Develop automated domain learnability estimation before adaptation

---

## Reference

This work builds on and extends:

> Tran et al., "A Domain Adaptation Framework for Speech Recognition Systems with Only Synthetic Data", Meta AI, arXiv:2501.12501, January 2025.
