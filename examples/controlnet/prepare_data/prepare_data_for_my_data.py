from pathlib import Path
import numpy as np
from PIL import Image
import shutil
import matplotlib.pyplot as plt
import os
import tqdm
import json
import torchaudio
from speechbrain.lobes.features import Fbank
import torch

# Empty the folder if it exists
shutil.rmtree("prepared_data/data", ignore_errors=True)

# Define the folder paths
base_dir = Path.cwd() / "prepared_data/data"
source_dir = base_dir / "conditioning_images"
target_dir = base_dir / "images"

# Empty if exists if not create the folders
shutil.rmtree(base_dir, ignore_errors=True)
os.makedirs(source_dir)
os.makedirs(target_dir)

print(f"Created directories:\n{base_dir}\n{source_dir}\n{target_dir}")

data_origin_parent_folder = "images"
images_sub_folders = [x for x in Path(data_origin_parent_folder).iterdir() if x.is_dir()]
metadata = {}

from pathlib import Path
import numpy as np
from PIL import Image
import shutil
import os
import json
import torchaudio
from speechbrain.lobes.features import Fbank
import torch

# (Previous setup code remains the same)

# Initialize the Fbank feature extractor
fbank = Fbank(
    n_mels=80,  # number of mel filters
    left_frames=0,
    right_frames=0,
    deltas=False,
)

# Define fixed duration in seconds
FIXED_DURATION = 5  # adjust this value as needed

for sub_folder in images_sub_folders[:10]:
    images = [x for x in sub_folder.iterdir() if x.is_file() and x.suffix in [".jpg", ".jpeg", ".png"]]
    audio_files = [x for x in Path(str(sub_folder).replace("images", "audios")).iterdir() if x.is_file() and x.suffix in [".wav", ".mp3"]]

    for image in images:
        image_name = image.stem
        print(f"Processing {image_name}...")
        print(f"image {image} .")
        img_counter = 0
        for audio_file in audio_files:
            audio_name = audio_file.stem
            
            image_target_name = f"{image_name}_{audio_name}"
            shutil.copy(image, target_dir / f"{image_target_name}.png")

            # Load the audio file
            signal, sample_rate = torchaudio.load(str(audio_file))
            
            # Ensure mono audio
            if signal.shape[0] > 1:
                signal = torch.mean(signal, dim=0, keepdim=True)
            
            # Calculate target length
            target_length = int(FIXED_DURATION * sample_rate)
            
            # Pad or trim the signal
            if signal.shape[1] < target_length:
                # Pad with zeros
                signal = torch.nn.functional.pad(signal, (0, target_length - signal.shape[1]))
            else:
                # Trim
                signal = signal[:, :target_length]
            
            # Generate the mel spectrogram
            with torch.no_grad():
                mel_spectrogram = fbank(signal)
            mel_spectrogram = mel_spectrogram.squeeze().cpu().numpy()

            # Normalize the spectrogram
            mel_spectrogram = (mel_spectrogram - np.min(mel_spectrogram)) / (np.max(mel_spectrogram) - np.min(mel_spectrogram))
            
            # Convert to uint8
            mel_spectrogram = (mel_spectrogram * 255).astype(np.uint8)

            # Create and save the image
            img = Image.fromarray(mel_spectrogram.T)  # Transpose for correct orientation
            img = img.convert("L")  # Convert to grayscale
            img.save(source_dir / f"{audio_name}.png")


            metadata_entry = {
                "image": f"images/{image_target_name}.png",
                "conditioning_image": f"conditioning_images/{audio_name}.png",
                "text": ""
            }


            # print spectogram shape


            # Save each metadata entry to the file
            with open(base_dir / "metadata.json", "a") as f:
                f.write(json.dumps(metadata_entry) + "\n")

print("Processing completed.")