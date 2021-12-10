from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.routers.eeg_data import eeg_router
import uvicorn

app = FastAPI()

# 註冊
app.include_router(eeg_router, prefix='/api/v1')


@app.get("/")
async def read_root():
    return {"Hello": "world"}


# 跨域的資源共享
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# if __name__ =="__main__":
#     uvicorn.run(app='main:app', host="127.0.0.1", port=8080, reload=True, debug=True)
