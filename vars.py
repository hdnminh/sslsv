import os
from dotenv import load_dotenv
load_dotenv()

CACHED_DIR = os.getenv('CACHED_DIR')


tts_cache_dir = os.path.join(CACHED_DIR, "tts_outputs")
detector_cache_dir = os.path.join(CACHED_DIR, "detector_outputs")
os.makedirs(tts_cache_dir, exist_ok=True)
os.makedirs(detector_cache_dir, exist_ok=True)
aasist2_path = os.path.join(CACHED_DIR, "models/AASIST2.pth")
rawnet2_path = os.path.join(CACHED_DIR, "models/rawnet2.pth")
aasist_path = os.path.join(CACHED_DIR, "models/AASIST.pth")
clad_path = os.path.join(CACHED_DIR, "models/CLAD_150_10_2310.pth.tar")
f5tts_path = CACHED_DIR
wav2net2_path = os.path.join(CACHED_DIR, "models/xlsr2_300m.pt")