from typing import List
from pydantic import BaseModel

class RequestBody(BaseModel):
    samples: List[str]

class ResponseBody(BaseModel):
    samples: List[str]
    predictions: List[str]