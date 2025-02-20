from typing import Union
from fastapi import FastAPI
import uvicorn

import model # model.py 불러오기

model = model.AndModel() # AndModel 클래스 인스턴스 생성

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/predict/left/{left}/right/{right}")
def predict(left:int, right:int):
    result = model.predict([left, right])
    return {"result" : result}

@app.post("/train")
def train():
    model.train()
    return {"result": "OK"}

# FastAPI 서버 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)