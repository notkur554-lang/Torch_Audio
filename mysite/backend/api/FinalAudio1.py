from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from pathlib import Path
import torch
import torch.nn as nn
from torchaudio import transforms
import io
import torch.nn.functional as F
import soundfile as sf

Audio1_router = APIRouter(prefix='/audio1', tags=['Audio 1'])

class CheckAudio(nn.Module):
  def __init__(self,):
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
        nn.Linear(128, 35)
    )

  def forward(self, x):
    x = x.unsqueeze(1)
    x = self.first(x)
    x = self.second(x)
    return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_dir = Path(__file__).resolve().parents[3]
path_labels = base_dir / 'al_models' / 'labels (1).pth'
path_model = base_dir / 'al_models' / 'audio_model.pth'

labels = torch.load(path_labels, map_location='cpu', weights_only=False)
index_to_labels = {ind: lab for ind, lab in enumerate(labels)}
model = CheckAudio()
model.load_state_dict(torch.load(path_model, map_location=device))
model.to(device)
model.eval()

transforms = transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=64
)
max_len = 100

def change_audio(waveform, sample_rate):
    if sample_rate != 16000:
        new_sr = transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = new_sr(torch.tensor(waveform))

    spec = transforms(waveform).squeeze(0)

    if spec.shape[1] > max_len:
        spec = spec[:, :max_len]

    if spec.shape[1] < max_len:
        count_len = max_len - spec.shape[1]
        spec = F.pad(spec, (0, count_len))

    return spec


#Audio1_router = APIRouter()

@Audio1_router.post('/predict/')
async def predict_audio(file: UploadFile = File(..., )):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail='Файл Бош')
        wf, sr = sf.read(io.BytesIO(data), dtype='float32')
        wf = torch.tensor(wf).T

        spec = change_audio(wf, sr).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(spec)
            pred_index = torch.argmax(y_pred, dim=1).item()
            pred_class = index_to_labels[pred_index]
            return {'Индекс':pred_index, 'Класс':pred_class}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

