from fastapi import Depends, APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.pipelines.base import Pipeline
from app.dependencies import get_pipeline
from app.routes.util import image_to_data_url, ImageResponse, HTTPError, http_error
from PIL import Image
from typing import Annotated
import logging
import random
import os

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

router = APIRouter()

logger = logging.getLogger(__name__)

responses = {400: {"model": HTTPError}, 500: {"model": HTTPError}}


# TODO: Make model_id optional once Go codegen tool supports OAPI 3.1
# https://github.com/deepmap/oapi-codegen/issues/373
@router.post("/image-to-image", response_model=ImageResponse, responses=responses)
@router.post(
    "/image-to-image/",
    response_model=ImageResponse,
    responses=responses,
    include_in_schema=False,
)
async def image_to_image(
    prompt: Annotated[str, Form()],
    image: Annotated[UploadFile, File()],
    model_id: Annotated[str, Form()] = "",
    base_model_id: Annotated[str, Form()] = "",
    strength: Annotated[float, Form()] = 0.8,
    guidance_scale: Annotated[float, Form()] = 7.5,
    image_guidance_scale: Annotated[float, Form()] = 0,
    negative_prompt: Annotated[str, Form()] = "",
    seed: Annotated[int, Form()] = None,
    num_images_per_prompt: Annotated[int, Form()] = 1,
    pipeline: Pipeline = Depends(get_pipeline),
    token: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
    speedup_module: Annotated[str, Form()] = "",
    animate_module: Annotated[str, Form()] = "",
):
    auth_token = os.environ.get("AUTH_TOKEN")
    if auth_token:
        if not token or token.credentials != auth_token:
            return JSONResponse(
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
                content=http_error("Invalid bearer token"),
            )

    if model_id != "" and model_id != pipeline.model_id:
        return JSONResponse(
            status_code=400,
            content=http_error(
                f"pipeline configured with {pipeline.model_id} but called with "
                f"{model_id}"
            ),
        )

    if seed is None:
        init_seed = random.randint(0, 2**32 - 1)
        if num_images_per_prompt > 1:
            seed = [i for i in range(init_seed, init_seed + num_images_per_prompt)]
        else:
            seed = init_seed

    img = Image.open(image.file).convert("RGB")
    # If a list of seeds/generators is passed, diffusers wants a list of images
    # https://github.com/huggingface/diffusers/blob/17808a091e2d5615c2ed8a63d7ae6f2baea11e1e/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py#L715
    if isinstance(seed, list):
        image = [img] * num_images_per_prompt
    else:
        image = img

    try:
        # TODO: hotswap base_model_id
        images = pipeline(
            prompt=prompt,
            image=image,
            strength=strength,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            negative_prompt=negative_prompt,
            seed=seed,
            num_images_per_prompt=num_images_per_prompt,
            speedup_module=speedup_module,
            animate_module=animate_module,
        )
    except Exception as e:
        logger.error(f"ImageToImagePipeline error: {e}")
        logger.exception(e)
        return JSONResponse(
            status_code=500, content=http_error("ImageToImagePipeline error")
        )

    seeds = seed
    if not isinstance(seeds, list):
        seeds = [seeds]

    output_images = []
    for img, s in zip(images, seeds):
        output_images.append({"url": image_to_data_url(img), "seed": s})

    return {"images": output_images}
