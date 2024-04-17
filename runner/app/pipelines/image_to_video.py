from app.pipelines.base import Pipeline
from app.pipelines.util import get_torch_device, get_model_dir

from diffusers import StableVideoDiffusionPipeline, I2VGenXLPipeline, DPMSolverMultistepScheduler
from huggingface_hub import file_download, hf_hub_download
from safetensors.torch import load_file
from app.lcm_scheduler import AnimateLCMSVDStochasticIterativeScheduler
import torch
import PIL
from typing import List
import logging
import os
import time

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

SFAST_WARMUP_ITERATIONS = 2  # Model warm-up iterations when SFAST is enabled.
I2VGEN_LIGHTNING_MODEL_ID = "ali-vilab/i2vgen-xl"
SVD_LCM_MODEL_ID = "wangfuyun/AnimateLCM-SVD-xt"
SVD_MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"

class ImageToVideoPipeline(Pipeline):
    def __init__(self, model_id: str):
        kwargs = {"cache_dir": get_model_dir()}

        torch_device = get_torch_device()
        folder_name = file_download.repo_folder_name(
            repo_id=model_id, repo_type="model"
        )
        folder_path = os.path.join(get_model_dir(), folder_name)
        has_fp16_variant = any(
            ".fp16.safetensors" in fname
            for _, _, files in os.walk(folder_path)
            for fname in files
        )
        if torch_device != "cpu" and has_fp16_variant:
            logger.info("ImageToVideoPipeline loading fp16 variant for %s", model_id)

            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"

        if I2VGEN_LIGHTNING_MODEL_ID in model_id:
            self.preferredWidth = 768
            self.preferredHeight = 768
            self.ldm = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16")
        elif SVD_LCM_MODEL_ID in model_id:
            self.preferredWidth = 1024
            self.preferredHeight = 576
            noise_scheduler = AnimateLCMSVDStochasticIterativeScheduler(
                num_train_timesteps=40,
                sigma_min=0.002,
                sigma_max=700.0,
                sigma_data=1.0,
                s_noise=1.0,
                rho=7,
                clip_denoised=False,
            )
            self.ldm = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt-1-1", 
                torch_dtype=torch.float16, scheduler=noise_scheduler,
                variant="fp16")
            self.ldm.unet.load_state_dict(load_file(hf_hub_download("wangfuyun/AnimateLCM-SVD-xt", "AnimateLCM-SVD-xt-1.1.safetensors"), device="cuda"), strict=False)
        elif SVD_MODEL_ID in model_id:
            self.preferredWidth = 1024
            self.preferredHeight = 576
            self.ldm = StableVideoDiffusionPipeline.from_pretrained(model_id, **kwargs)
        self.model_id = model_id
        self.ldm.to(get_torch_device())

        if os.getenv("SFAST", "").strip().lower() == "true":
            logger.info(
                "ImageToVideoPipeline will be dynamically compiled with stable-fast "
                "for %s",
                model_id,
            )
            from app.pipelines.sfast import compile_model

            self.ldm = compile_model(self.ldm)

            # Warm-up the pipeline.
            # NOTE: Initial calls may be slow due to compilation. Subsequent calls will
            # be faster.
            if os.getenv("SFAST_WARMUP", "true").lower() == "true":
                # Retrieve default model params.
                # TODO: Retrieve defaults from Pydantic class in route.
                warmup_kwargs = {
                    "image": PIL.Image.new("RGB", (576, 1024)),
                    "height": 576,
                    "width": 1024,
                    "fps": 6,
                    "motion_bucket_id": 127,
                    "noise_aug_strength": 0.02,
                    "decode_chunk_size": 25,
                }

                logger.info("Warming up ImageToVideoPipeline pipeline...")
                total_time = 0
                for ii in range(SFAST_WARMUP_ITERATIONS):
                    t = time.time()
                    try:
                        self.ldm(**warmup_kwargs).frames
                    except Exception as e:
                        # FIXME: When out of memory, pipeline is corrupted.
                        logger.error(f"ImageToVideoPipeline warmup error: {e}")
                        raise e
                    iteration_time = time.time() - t
                    total_time += iteration_time
                    logger.info(
                        "Warmup iteration %s took %s seconds", ii + 1, iteration_time
                    )
                logger.info("Total warmup time: %s seconds", total_time)

    def __call__(self, image: PIL.Image, **kwargs) -> List[List[PIL.Image]]:
        seed = kwargs.pop("seed", None)
        if seed is not None:
            if isinstance(seed, int):
                kwargs["generator"] = torch.Generator(get_torch_device()).manual_seed(
                    seed
                )
            elif isinstance(seed, list):
                kwargs["generator"] = [
                    torch.Generator(get_torch_device()).manual_seed(s) for s in seed
                ]

        if "width" not in kwargs or kwargs["width"] == None:
            kwargs["width"] = self.preferredWidth
        if "height" not in kwargs or kwargs["height"] == None:
            kwargs["height"] = self.preferredHeight

        if "speedup_module" in kwargs:
            del kwargs["speedup_module"]
        if "animate_module" in kwargs:
            del kwargs["animate_module"]

        # Scale to requested resolution
        if image.size[1] != kwargs["height"] or image.size[0] != kwargs["width"]:
            hpercent = (kwargs["height"] / float(image.size[1]))
            wsize = int((float(image.size[0]) * float(hpercent)))
            hsize = kwargs["height"]
            # If mismatching aspect ratio, adjust down again
            if wsize > kwargs["width"]:
                wpercent = (kwargs["width"] / float(image.size[0]))
                hsize = int((float(image.size[1]) * float(wpercent)))
                wsize = kwargs["width"]
            image = image.resize((wsize, hsize), PIL.Image.Resampling.LANCZOS)
            kwargs["height"] = hsize
            kwargs["width"] = wsize

        if SVD_LCM_MODEL_ID in self.model_id:
            kwargs["num_inference_steps"] = 6
            kwargs["min_guidance_scale"] = 1.0
            kwargs["max_guidance_scale"] = 1.2
            if "num_frames" not in kwargs:
                kwargs["num_frames"] = 25
            if "decode_chunk_size" not in kwargs:
                kwargs["decode_chunk_size"] = 8
            if "prompt" in kwargs:
                del kwargs["prompt"]
            if "negative_prompt" in kwargs:
                del kwargs["negative_prompt"]
        elif I2VGEN_LIGHTNING_MODEL_ID in self.model_id:
            kwargs["num_inference_steps"] = 25
            if "num_frames" not in kwargs:
                kwargs["num_frames"] = 25
            if "decode_chunk_size" not in kwargs:
                kwargs["decode_chunk_size"] = 8
            if "num_frames" not in kwargs:
                kwargs["num_frames"] = 25
            if "fps" in kwargs:
                del kwargs["fps"]
            if "motion_bucket_id" in kwargs:
                del kwargs["motion_bucket_id"]
            if "noise_aug_strength" in kwargs:
                del kwargs["noise_aug_strength"]
            if "negative_prompt" in kwargs:
                del kwargs["negative_prompt"]
            prompt = ""
            if "prompt" in kwargs:
                prompt = kwargs["prompt"]
                del kwargs["prompt"]
            return self.ldm(prompt, image, **kwargs).frames
        elif SVD_MODEL_ID in self.model_id:
            if "prompt" in kwargs:
                del kwargs["prompt"]
            if "decode_chunk_size" not in kwargs:
                kwargs["decode_chunk_size"] = 8
            if "num_frames" not in kwargs:
                kwargs["num_frames"] = 25
            if "negative_prompt" in kwargs:
                del kwargs["negative_prompt"]

        return self.ldm(image, **kwargs).frames

    def __str__(self) -> str:
        return f"ImageToVideoPipeline model_id={self.model_id}"
