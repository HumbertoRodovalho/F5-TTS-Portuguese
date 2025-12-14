This repository provides a working training pipeline for Portuguese using the new Hydra-based version of F5-TTS.
The goal is to make the training process reproducible, modular, and adaptable for Portuguese speech synthesis research.

⚠️ Status: Training in progress — early checkpoints and logs available.

📌 Motivation

The recent versions of F5-TTS migrated to a Hydra-based configuration system, which introduced flexibility but also significantly raised the entry barrier for new languages.

As of now, there is no public, documented pipeline for training F5-TTS in Portuguese.

This project aims to:

    Adapt the new F5-TTS training pipeline to Portuguese

    Provide a fully functional Hydra configuration

    Support custom tokenizers and datasets

    Enable reproducible training in environments such as Kaggle

🧠 Key Features

    ✅ Hydra-based configuration (no legacy .toml)

    ✅ Custom Portuguese tokenizer

    ✅ Support for custom Portuguese datasets

    ✅ Integrated vocoder (Vocos)

    ✅ TensorBoard logging with audio samples

    ✅ Tested on Kaggle (single-GPU)

OBS: This repository is configured by default for maximum compatibility with GPUs of 16GB or less. If you have access to superior hardware (e.g., RTX 3090/4090,   A100, H100), you can drastically speed up training by modifying the `configs/braille_t4_optimized.yaml` file.

⚙️ How to Use

1. Prepare the Dataset

Place your audio files (.wav) and the texts.txt file (path|transcription format) in a folder. The script below normalizes the audio to 24kHz, cleans the text, and generates the Arrow structure:

python scripts/prepare_dataset.py --dataset_path /caminho/do/dataset

2. Training

The training uses the accelerate library and the optimized configuration file from this repository:

accelerate launch --num_processes=1 \
    F5-TTS/src/f5_tts/train/train.py \
    --config-dir ./configs \
    --config-name braille_t4_optimized

3. Inference (Generate Voice)

To test the trained model:

python scripts/inference.py --ckpt_path ckpts/model_last.pt --ref_audio ref.wav --text "Olá mundo!"