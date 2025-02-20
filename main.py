from typing import Union
from fastapi import FastAPI
import uvicorn
from model import AndModel

model = AndModel()

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome to AND Model API"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/predict/{x}/{y}")
def predict(x: int, y: int):
    result = model.predict([x, y])
    return {"prediction": int(result)}

@app.post("/train")
def train():
    model.train()
    return {"result": "OK"}

# FastAPI 서버 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)