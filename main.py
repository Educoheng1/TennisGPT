from fastapi import FastAPI
from tennis_gpt_logic import router as tennis_gpt_router

app = FastAPI()

app.include_router(tennis_gpt_router)
