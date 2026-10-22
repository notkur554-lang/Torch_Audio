from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
import uvicorn
import torch
import torch.nn as nn
from torchaudio import transforms
import io
import torch.nn.functional as F
import soundfile as sf
from pathlib import Path

Audio2_router = APIRouter(prefix="/audio2", tags=["Audio 2"])

class CheckAudio(nn.Module):
  def __init__(self):
    super().__init__()
    self.first = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((8, 8))
    )
    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
  def forward(self, x):
    x = x.unsqueeze(1)
    x = self.first(x)
    x = self.second(x)
    return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.MelSpectrogram(
    sample_rate=22050,
    n_mels=64
)
max_len = 500

base_dir = Path(__file__).resolve().parents[3]
path_labels = base_dir / 'al_models' / 'labels.pth'
path_model = base_dir / 'al_models' / 'model.pth'

labels = torch.load(path_labels, map_location='cpu', weights_only=False)
index_to_label = {ind: lab for ind, lab in enumerate(labels)}
model = CheckAudio()
model.load_state_dict(torch.load(path_model, map_location=device))
model.to(device)
model.eval()

def change_audio(waveform, sr):
    if sr != 22050:
        resample = transforms.Resample(orig_freq=sr, new_freq=22050)
        waveform = resample(waveform)

    spec = transform(waveform).squeeze(0)

    if spec.shape[1] > max_len:
        spec = spec[:, :max_len]

    if spec.shape[1] < max_len:
        count_len = max_len - spec.shape[1]
        spec = F.pad(spec, (0, count_len))

    return spec




@Audio2_router.post('/predict/')
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail='Empty File')

        wf, sr = sf.read(io.BytesIO(data), dtype='float32')
        #wf = torch.tensor(wf).T
        wf = torch.tensor(wf)
        if wf.ndim == 2:
            wf = wf.T

        spec = change_audio(wf, sr).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(spec)
            pred_ind = torch.argmax(y_pred, dim=1).item()
            pred_class = index_to_label[pred_ind]

        return {f'Индекс: {pred_ind}, Жанр: {pred_class}'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




