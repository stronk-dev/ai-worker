from app.pipelines.base import Pipeline
from app.pipelines.util import get_torch_device, get_model_dir

from diffusers import AnimateDiffVideoToVideoPipeline, MotionAdapter, DPMSolverMultistepScheduler
from huggingface_hub import file_download
import torch
import PIL
from typing import List
import logging
import os
import time

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# See https://huggingface.co/ByteDance/AnimateDiff-Lightning
# Compatible with any SD1.5 stylized base model
bases = {
    "AbsoluteReality": "digiplay/AbsoluteReality_v1.8.1",
    "epiCRealism": "emilianJR/epiCRealism",
    "DreamShaper": "Lykon/DreamShaper",
    "RealisticVision": "SG161222/Realistic_Vision_V6.0_B1_noVAE"
}
base_loaded = "epiCRealism"

motionChoices = [
    ("Default", ""),
    ("Zoom in", "guoyww/animatediff-motion-lora-zoom-in"),
    ("Zoom out", "guoyww/animatediff-motion-lora-zoom-out"),
    ("Tilt up", "guoyww/animatediff-motion-lora-tilt-up"),
    ("Tilt down", "guoyww/animatediff-motion-lora-tilt-down"),
    ("Pan left", "guoyww/animatediff-motion-lora-pan-left"),
    ("Pan right", "guoyww/animatediff-motion-lora-pan-right"),
    ("Roll left", "guoyww/animatediff-motion-lora-rolling-anticlockwise"),
    ("Roll right", "guoyww/animatediff-motion-lora-rolling-clockwise"),
],

logger = logging.getLogger(__name__)

class VideoToVideoPipeline(Pipeline):
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
            logger.info("VideoToVideoPipeline loading fp16 variant for %s", model_id)

            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"
            kwargs["use_safetensors"] = True

        self.model_id = model_id

        kwargs["torch_dtype"] = torch.float16
        # Load base
        adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
        self.ldm = AnimateDiffVideoToVideoPipeline.from_pretrained(bases[base_loaded], motion_adapter=adapter, torch_dtype=torch.float16).to("cuda")
        # Load noise scheduler
        self.ldm.scheduler = DPMSolverMultistepScheduler.from_config(self.ldm.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++", clip_sample=False, timestep_spacing="trailing", beta_schedule="linear", steps_offset=1)
        self.ldm.to(get_torch_device())

    def __call__(self, video: List[PIL.Image], **kwargs) -> List[List[PIL.Image]]:
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

        if "speedup_module" in kwargs:
            del kwargs["speedup_module"]
        if "animate_module" in kwargs:
            del kwargs["animate_module"]

        kwargs["strength"] = 0.5
        kwargs["num_inference_steps"] = 25
        if "fps" in kwargs:
            kwargs["frame_rate"] = kwargs["fps"]
            del kwargs["fps"]
        return self.ldm(video, **kwargs).frames

    def __str__(self) -> str:
        return f"VideoToVideoPipeline model_id={self.model_id}"
