from pydantic import BaseModel
from fastapi import Depends, APIRouter
from fastapi.responses import JSONResponse
from app.pipelines import TextToImagePipeline
from app.dependencies import get_pipeline
from app.routes.util import image_to_data_url, ImageResponse, HTTPError, http_error
import logging
from typing import List
import random

router = APIRouter()

logger = logging.getLogger(__name__)


class TextToImageParams(BaseModel):
    # TODO: Make model_id optional once Go codegen tool supports OAPI 3.1
    # https://github.com/deepmap/oapi-codegen/issues/373
    model_id: str = ""
    prompt: str
    height: int = None
    width: int = None
    guidance_scale: float = 7.5
    negative_prompt: str = ""
    seed: int = None
    num_images_per_prompt: int = 1


responses = {400: {"model": HTTPError}, 500: {"model": HTTPError}}


@router.post("/text-to-image", response_model=ImageResponse, responses=responses)
@router.post("/text-to-image/", response_model=ImageResponse, include_in_schema=False)
async def text_to_image(
    params: TextToImageParams, pipeline: TextToImagePipeline = Depends(get_pipeline)
):
    if params.model_id != "" and params.model_id != pipeline.model_id:
        return JSONResponse(
            status_code=400,
            content=http_error(
                f"pipeline configured with {pipeline.model_id} but called with {params.model_id}"
            ),
        )

    if params.seed is None:
        init_seed = random.randint(0, 2**32 - 1)
        if params.num_images_per_prompt > 1:
            params.seed = [
                i for i in range(init_seed, init_seed + params.num_images_per_prompt)
            ]
        else:
            params.seed = init_seed

    try:
        images = pipeline(**params.model_dump())
    except Exception as e:
        logger.error(f"TextToImagePipeline error: {e}")
        logger.exception(e)
        return JSONResponse(
            status_code=500, content=http_error("TextToImagePipeline error")
        )

    seeds = params.seed
    if not isinstance(seeds, list):
        seeds = [seeds]

    output_images = []
    for img, sd in zip(images, seeds):
        output_images.append({"url": image_to_data_url(img), "seed": sd})

    return {"images": output_images}
