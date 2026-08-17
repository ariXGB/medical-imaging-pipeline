from io import BytesIO

from fastapi import APIRouter,UploadFile,File,HTTPException

from PIL import Image

from predict.prediction_pipeline import predict_pipeline


router = APIRouter()

@router.post("/")
async def predict(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):

        raise HTTPException(
            status_code=400,
            detail="File must be an image."
        )
    contents = await file.read()

    image = Image.open(BytesIO(contents))

    return predict_pipeline(image)