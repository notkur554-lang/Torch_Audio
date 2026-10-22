from fastapi import FastAPI
import uvicorn

from mysite.backend.api import FinalAudio1
from mysite.backend.api import FinalAudio2


app = FastAPI()
app.include_router(FinalAudio1.Audio1_router)
app.include_router(FinalAudio2.Audio2_router)



if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)

