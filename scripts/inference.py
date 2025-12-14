import os
import torch
import soundfile as sf
import argparse
from f5_tts.model import DiT, CFM
from f5_tts.infer.utils_infer import load_vocoder

# Argument configuration (for running via terminal)
parser = argparse.ArgumentParser(description="Gerar áudio com F5-TTS (Inferência)")
parser.add_argument("--ckpt_path", type=str, required=True, help="Caminho para o arquivo .pt (checkpoint)")
parser.add_argument("--ref_audio", type=str, required=True, help="Caminho para um áudio .wav de referência (5-10s)")
parser.add_argument("--text", type=str, required=True, help="Texto para ser falado")
parser.add_argument("--output_file", type=str, default="generated.wav", help="Nome do arquivo de saída")
parser.add_argument("--vocab_file", type=str, default="./F5-TTS/data/braille_arrow_custom/vocab.txt", help="Caminho do vocab.txt")

args = parser.parse_args()

def main():
    print(f"🧪 Iniciando Inferência...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint não encontrado: {args.ckpt_path}")
    if not os.path.exists(args.ref_audio):
        raise FileNotFoundError(f"Áudio de referência não encontrado: {args.ref_audio}")

    # 1. Load Vocabulary
    # If it can't find the file, use a basic pattern (risky, but it works for testing).
    if os.path.exists(args.vocab_file):
        with open(args.vocab_file, "r", encoding="utf-8") as f:
            vocab_char_map = {c.strip(): i for i, c in enumerate(f.readlines())}
    else:
        print("⚠️ vocab.txt não encontrado! Usando mapa padrão (pode dar erro de caracteres).")
        vocab_char_map = None

    # 2. Define the Architecture (It must be IDENTICAL to the training)
    model_cfg = dict(
        vocab_char_map=vocab_char_map,
        vocab_size=len(vocab_char_map) + 1 if vocab_char_map else 100,
        name="F5-TTS",
        backbone=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
        mel_spec_kwargs=dict(target_sample_rate=24000, n_mel_channels=100, hop_length=256, win_length=1024, n_fft=1024)
    )

    # 3. Instantiate Model
    model = CFM(
        backbone=DiT(**model_cfg["backbone"]),
        mel_spec_kwargs=model_cfg["mel_spec_kwargs"],
        vocab_char_map=vocab_char_map
    ).to(device)

    # 4. Carrying Weights
    print(f"📂 Carregando checkpoint: {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    
    # Treatment for different rescue formats
    state_dict = checkpoint.get("ema_model_state_dict", checkpoint.get("model_state_dict", checkpoint))
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(clean_state_dict)
    model.eval()

    # 5. Load Vocoder
    vocoder = load_vocoder(is_local=False)

    # 6. Generate Audio
    print(f"🗣️  Falando: '{args.text}'")
    with torch.inference_mode():
        generated_audio, _ = model.infer(
            ref_file=args.ref_audio,
            text=args.text,
            torchaudio_backend="soundfile"
        )

    # 7. Save Audio
    sf.write(args.output_file, generated_audio, 24000)
    print(f"✅ Áudio salvo com sucesso em: {args.output_file}")

if __name__ == "__main__":
    main()