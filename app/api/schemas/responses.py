from pydantic import BaseModel


class PredictionResponse(BaseModel):

    accepted: bool

    message: str

    gatekeeper_confidence: float

    diagnosis: str | None = None

    diagnosis_confidence: float | None = None

    probabilities: dict[str, float] | None = None