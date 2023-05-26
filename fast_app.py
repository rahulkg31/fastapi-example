import pickle
import asyncio
import uvicorn

from fastapi import FastAPI
from aiohttp import ClientSession

from schemas import (
    RequestBody,
    ResponseBody
)


app = FastAPI(
    title="simple-model-server",
    description="a simple model serve in FastAPI",
    version="0.1",
)

with open("models/model.pkl", "rb") as rf:
    clf = pickle.load(rf)

with open("models/vector.pkl", "rb") as rf:
    vectorizer = pickle.load(rf)

client_session = ClientSession()

@app.get("/healthcheck")
async def healthcheck():
    """
        Check if API is running
    """
    return "The api is working fine"


@app.post("/predict", response_model=ResponseBody)
async def predict(body: RequestBody):
    """
        Predict given list of texts
    """
    vectors = vectorizer.transform(body.samples)
    predictions = clf.predict(vectors)

    # add 1 sec sleep for testing async
    await asyncio.sleep(1)

    return {
        "samples": body.samples,    
        "predictions": predictions.tolist()
    }


@app.post("/predict/{text}", response_model=ResponseBody)
async def predict_label(text: str):
    """
        Predict given a text
    """
    vectors = vectorizer.transform([text])
    predictions = clf.predict(vectors)

    # add 1 sec sleep for testing async
    await asyncio.sleep(1)

    return {
        "samples": [text],    
        "predictions": predictions.tolist()
    }


@app.on_event("shutdown")
async def cleanup():
    """
        Clean up when shutdown
    """
    await client_session.close()

if __name__ == "__main__":
    uvicorn.run("fast_app:app", host="0.0.0.0", port=5000, log_level="info")