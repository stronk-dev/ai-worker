from fastapi import Depends, APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.pipelines.base import Pipeline
from app.dependencies import get_pipeline
from app.routes.util import image_to_data_url, VideoResponse, HTTPError, http_error
from PIL import Image
from typing import Annotated
import imageio
from io import BytesIO
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
@router.post("/video-to-video", response_model=VideoResponse, responses=responses)
@router.post(
    "/video-to-video/",
    response_model=VideoResponse,
    responses=responses,
    include_in_schema=False,
)
async def video_to_video(
    video: Annotated[UploadFile, File()],
    prompt: Annotated[str, Form()] = "",
    negative_prompt: Annotated[str, Form()] = "",
    model_id: Annotated[str, Form()] = "",
    base_model_id: Annotated[str, Form()] = "",
    height: Annotated[int, Form()] = 576,
    width: Annotated[int, Form()] = 1024,
    fps: Annotated[int, Form()] = 6,
    motion_bucket_id: Annotated[int, Form()] = 127,
    noise_aug_strength: Annotated[float, Form()] = 0.02,
    seed: Annotated[int, Form()] = None,
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

    if height % 8 != 0 or width % 8 != 0:
        return JSONResponse(
            status_code=400,
            content=http_error(
                f"`height` and `width` have to be divisible by 8 but are {height} and "
                f"{width}."
            ),
        )

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    images = []
    content = BytesIO(video.file)
    vid = imageio.get_reader(content)
    for frame in vid:
        pil_image = Image.fromarray(frame)
        images.append(pil_image)
    if len(images < 1):
        logger.error(f"VideoToVideoPipeline error: unable to decode video file")
        logger.exception(e)
        return JSONResponse(
            status_code=500, content=http_error("VideoToVideoPipeline error")
        )

    try:
        batch_frames = pipeline(
            video=images,
            height=height,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            seed=seed,
            base_model_id=base_model_id,
            speedup_module=speedup_module,
            animate_module=animate_module,
        )
    except Exception as e:
        logger.error(f"VideoToVideoPipeline error: {e}")
        logger.exception(e)
        return JSONResponse(
            status_code=500, content=http_error("VideoToVideoPipeline error")
        )

    output_frames = []
    for frames in batch_frames:
        output_frames.append(
            [{"url": image_to_data_url(frame), "seed": seed} for frame in frames]
        )

    return {"frames": output_frames}
