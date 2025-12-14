import os
import glob
import re
import unicodedata
import soundfile as sf
import librosa
import num2words
import json
import argparse
from tqdm import tqdm
from datasets import Dataset

# Setup of arguments to run via terminal.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, required=True, help="Pasta raiz contendo 'wavs/' e 'texts.txt'")
parser.add_argument("--output_path", type=str, default="./F5-TTS/data/braille_custom", help="Onde salvar os dados processados")
args = parser.parse_args()

def clean_text(text):
    text = unicodedata.normalize('NFC', text)
    try: text = re.sub(r'\d+', lambda x: num2words.num2words(int(x.group()), lang='pt_BR'), text)
    except: pass
    return " ".join(text.split())

def main():
    raw_dir = os.path.join(args.output_path, "raw")
    wavs_out = os.path.join(args.output_path, "wavs_processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(wavs_out, exist_ok=True)

    # Search for texts.txt
    txt_path = os.path.join(args.dataset_path, "texts.txt")
    if not os.path.exists(txt_path):
        # Try to find it recursively if it's not at the root.
        candidates = glob.glob(os.path.join(args.dataset_path, "**/texts.txt"), recursive=True)
        if not candidates: raise FileNotFoundError(f"texts.txt não encontrado em {args.dataset_path}")
        txt_path = candidates[0]

    print(f"📍 Usando índice: {txt_path}")
    
    audio_paths, texts, durations, duration_map = [], [], [], {}

    with open(txt_path, 'r', encoding='utf-8') as f: lines = f.readlines()

    for line in tqdm(lines, desc="Processando"):
        if not line.strip(): continue
        parts = line.strip().split('|') if '|' in line else line.strip().split(',')
        if len(parts) < 2: continue
        
        # Flatten logic
        fname = os.path.basename(parts[0].strip().replace('"', '').replace("'", ""))
        if fname.lower().endswith(".wav"): fname = fname[:-4]
        
        # Flexible Search for audio file
        possible_paths = [
            os.path.join(os.path.dirname(txt_path), fname + ".wav"),
            os.path.join(os.path.dirname(txt_path), "wavs", fname + ".wav"),
            os.path.join(args.dataset_path, fname + ".wav")
        ]
        
        src = next((p for p in possible_paths if os.path.exists(p)), None)
        
        if src:
            try:
                y, sr = librosa.load(src, sr=24000)
                dur = librosa.get_duration(y=y, sr=sr)
                if 0.5 <= dur <= 15.0:
                    dst_name = f"f5_{fname}.wav"
                    dst_path = os.path.join(wavs_out, dst_name)
                    sf.write(dst_path, y, 24000)
                    
                    # Creates absolute paths for Arrow.
                    audio_paths.append(os.path.abspath(dst_path))
                    texts.append(clean_text(parts[1].strip()))
                    durations.append(dur)
                    duration_map[dst_name] = dur
            except: pass

    # Save Arrow
    dataset = Dataset.from_dict({"audio_path": audio_paths, "text": texts, "duration": durations})
    dataset.save_to_disk(raw_dir)

    # Metadata
    all_text = "".join(texts)
    unique_chars = sorted(list(set(all_text)))
    with open(os.path.join(args.output_path, "vocab.txt"), "w", encoding="utf-8") as f:
        for c in unique_chars: f.write(f"{c}\n")

    with open(os.path.join(args.output_path, "duration.json"), "w", encoding="utf-8") as f:
        json.dump({"duration": duration_map}, f, indent=4)

    print(f"✅ Processamento concluído! {len(audio_paths)} áudios prontos.")

if __name__ == "__main__":
    main()