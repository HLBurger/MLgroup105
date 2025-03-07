# We will define all the configurations here
import torch
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Bepaal de juiste project root
BASE_DIR = Path(__file__).resolve().parent.parent  # Gaat 2 niveaus omhoog naar MLGroup105
CSV_PATH = BASE_DIR / "data" / "ageutk_data.csv"
# IMAGE_DIR = BASE_DIR / "data" / "organised_images"

