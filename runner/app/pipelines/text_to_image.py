from app.pipelines.base import Pipeline
from app.pipelines.util import get_torch_device, get_model_dir

from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler, UNet2DConditionModel, LCMScheduler,
    StableDiffusionPipeline
)
from safetensors.torch import load_file
from huggingface_hub import file_download, hf_hub_download
import torch
import PIL
from typing import List
import logging
import os

logger = logging.getLogger(__name__)

SDXL_LIGHTNING_MODEL_ID = "ByteDance/SDXL-Lightning"
SDXL_BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SD15_BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
ABSOLUTE_REALITY_BASE_MODEL_ID = "digiplay/AbsoluteReality_v1.8.1"
EPIC_REALISM_BASE_MODEL_ID = "emilianJR/epiCRealism"
DREAMSHAPER_BASE_MODEL_ID = "Lykon/DreamShaper"
REALISTIC_VISION_MODEL_ID = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
DREAMSHAPER_XL_BASE_MODEL_ID = "Lykon/dreamshaper-xl-1-0"
DREAMSHAPER_LIGHTNING_MODEL_ID = "Lykon/dreamshaper-xl-lightning"
DREAMSHAPER_TURBO_MODEL_ID = "Lykon/dreamshaper-xl-v2-turbo"


class TextToImagePipeline(Pipeline):
    def __init__(self, model_id: str):
        kwargs = {"cache_dir": get_model_dir()}

        torch_device = get_torch_device()
        folder_name = file_download.repo_folder_name(
            repo_id=model_id, repo_type="model"
        )
        folder_path = os.path.join(get_model_dir(), folder_name)
        # Load fp16 variant if fp16 safetensors files are found in cache
        # Special case SDXL-Lightning because the safetensors files are fp16 but are not
        # named properly right now
        has_fp16_variant = (
            any(
                ".fp16.safetensors" in fname
                for _, _, files in os.walk(folder_path)
                for fname in files
            )
            or SDXL_LIGHTNING_MODEL_ID in model_id
        )
        if torch_device != "cpu" and has_fp16_variant:
            logger.info("TextToImagePipeline loading fp16 variant for %s", model_id)

            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"

        if os.environ.get("BFLOAT16"):
            logger.info("TextToImagePipeline using bfloat16 precision for %s", model_id)
            kwargs["torch_dtype"] = torch.bfloat16

        self.model_id = model_id
        self.loaded_model_id = model_id
        self.base_model_id = ""
        self.speedup_module = ""
        self.animate_module = ""

        if SDXL_LIGHTNING_MODEL_ID in model_id:
            self.preferredWidth = 1024
            self.preferredHeight = 1024
            # Special case SDXL-Lightning because the unet for SDXL needs to be swapped
            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"
            kwargs["use_safetensors"] = True
            base = "stabilityai/stable-diffusion-xl-base-1.0"

            # ByteDance/SDXL-Lightning-2step
            if "2step" in model_id:
                unet_id = "sdxl_lightning_2step_unet"
            # ByteDance/SDXL-Lightning-4step
            elif "4step" in model_id:
                unet_id = "sdxl_lightning_4step_unet"
            # ByteDance/SDXL-Lightning-8step
            elif "8step" in model_id:
                unet_id = "sdxl_lightning_8step_unet"
            else:
                # Default to 2step
                unet_id = "sdxl_lightning_8step_unet"

            unet = UNet2DConditionModel.from_config(
                base, subfolder="unet", cache_dir=kwargs["cache_dir"]
            ).to(torch_device, kwargs["torch_dtype"])
            unet.load_state_dict(
                load_file(
                    hf_hub_download(
                        SDXL_LIGHTNING_MODEL_ID,
                        f"{unet_id}.safetensors",
                        cache_dir=kwargs["cache_dir"],
                    ),
                    device=str(torch_device),
                )
            )

            self.ldm = StableDiffusionXLPipeline.from_pretrained(
                base, unet=unet, **kwargs
            ).to(torch_device)

            self.ldm.scheduler = EulerDiscreteScheduler.from_config(
                self.ldm.scheduler.config, timestep_spacing="trailing"
            )
        elif SDXL_BASE_MODEL_ID in self.model_id or DREAMSHAPER_LIGHTNING_MODEL_ID in self.model_id or DREAMSHAPER_TURBO_MODEL_ID in self.model_id:
            self.preferredWidth = 1024
            self.preferredHeight = 1024
            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"
            kwargs["use_safetensors"] = True
            self.ldm = StableDiffusionXLPipeline.from_pretrained(model_id, **kwargs).to("cuda")
            self.ldm.scheduler = DPMSolverMultistepScheduler.from_config(self.ldm.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
        elif SD15_BASE_MODEL_ID in self.model_id or ABSOLUTE_REALITY_BASE_MODEL_ID in self.model_id or EPIC_REALISM_BASE_MODEL_ID in self.model_id or DREAMSHAPER_BASE_MODEL_ID in self.model_id or REALISTIC_VISION_MODEL_ID in self.model_id:
            self.preferredWidth = 768
            self.preferredHeight = 768
            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"
            kwargs["use_safetensors"] = True
            self.ldm = StableDiffusionPipeline.from_pretrained(model_id, **kwargs).to("cuda")
            self.ldm.scheduler = DPMSolverMultistepScheduler.from_config(self.ldm.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
        else:
            self.preferredWidth = 512
            self.preferredHeight = 512
            self.ldm = AutoPipelineForText2Image.from_pretrained(model_id, **kwargs).to(
                torch_device
            )

        if os.environ.get("TORCH_COMPILE"):
            torch._inductor.config.conv_1x1_as_mm = True
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.epilogue_fusion = False
            torch._inductor.config.coordinate_descent_check_all_directions = True

            self.ldm.unet.to(memory_format=torch.channels_last)
            self.ldm.vae.to(memory_format=torch.channels_last)

            self.ldm.unet = torch.compile(
                self.ldm.unet, mode="max-autotune", fullgraph=True
            )
            self.ldm.vae.decode = torch.compile(
                self.ldm.vae.decode, mode="max-autotune", fullgraph=True
            )

        if os.getenv("SFAST", "").strip().lower() == "true":
            logger.info(
                "TextToImagePipeline will be dynamically compiled with stable-fast for "
                "%s",
                model_id,
            )
            from app.pipelines.sfast import compile_model

            self.ldm = compile_model(self.ldm)

            # Warm-up the pipeline.
            # TODO: Not yet supported for ImageToImagePipeline.
            if os.getenv("SFAST_WARMUP", "true").lower() == "true":
                logger.warning(
                    "The 'SFAST_WARMUP' flag is not yet supported for the "
                    "TextToImagePipeline and will be ignored. As a result the first "
                    "call may be slow if 'SFAST' is enabled."
                )

    def __call__(self, prompt: str, **kwargs) -> List[PIL.Image]:
        self.base_model_id = kwargs["base_model_id"]
        del kwargs["base_model_id"]

        if self.base_model_id == "" or self.base_model_id == None:
            self.base_model_id = self.model_id

        if self.loaded_model_id != self.base_model_id:
            self.ldm.unet.load_state_dict(torch.load(hf_hub_download(self.base_model_id, "unet/diffusion_pytorch_model.bin"), map_location="cuda"), strict=False)
            self.loaded_model_id = self.base_model_id

        if "width" not in kwargs or kwargs["width"] == None:
            kwargs["width"] = self.preferredWidth
        if "height" not in kwargs or kwargs["height"] == None:
            kwargs["height"] = self.preferredHeight

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

        if (
            self.model_id == "stabilityai/sdxl-turbo"
            or self.model_id == "stabilityai/sd-turbo"
        ):
            # SD turbo models were trained without guidance_scale so
            # it should be set to 0
            kwargs["guidance_scale"] = 0.0

            if "num_inference_steps" not in kwargs:
                kwargs["num_inference_steps"] = 1
        elif SDXL_LIGHTNING_MODEL_ID in self.model_id:
            # SDXL-Lightning models should have guidance_scale = 0 and use
            # the correct number of inference steps for the unet checkpoint loaded
            kwargs["guidance_scale"] = 0.0

            if "2step" in self.model_id:
                kwargs["num_inference_steps"] = 2
            elif "4step" in self.model_id:
                kwargs["num_inference_steps"] = 4
            elif "8step" in self.model_id:
                kwargs["num_inference_steps"] = 8
            else:
                # Default to 2step
                kwargs["num_inference_steps"] = 2
        elif DREAMSHAPER_LIGHTNING_MODEL_ID in self.model_id:
            kwargs["guidance_scale"] = 2.0
            kwargs["num_inference_steps"] = 4
        elif DREAMSHAPER_TURBO_MODEL_ID in self.model_id:
            kwargs["guidance_scale"] = 2.0
            kwargs["num_inference_steps"] = 6
        elif SDXL_BASE_MODEL_ID in self.model_id or DREAMSHAPER_XL_BASE_MODEL_ID in self.model_id:
            # (un)load speedup module
            if kwargs["speedup_module"] == "LCM" and self.speedup_module != "LCM":
                self.speedup_module = "LCM"
                # Unload LCM LoRa weights
                self.ldm.unload_lora_weights()
                # Switch to LCM scheduler
                self.ldm.load_lora_weights("latent-consistency/lcm-lora-sdxl")
                self.ldm.scheduler = LCMScheduler.from_config(self.ldm.scheduler.config)
                self.ldm.fuse_lora()
            elif kwargs["speedup_module"] == "Lightning" and self.speedup_module != "Lightning":
                self.speedup_module = "Lightning"
                # Unload LCM LoRa weights
                self.ldm.unload_lora_weights()
                # Switch to Euler scheduler
                self.ldm.load_lora_weights(hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_8step_lora.safetensors"))
                self.ldm.scheduler = EulerDiscreteScheduler.from_config(self.ldm.scheduler.config, timestep_spacing="trailing")
                self.ldm.fuse_lora()
            elif kwargs["speedup_module"] == "" and self.speedup_module != "":
                self.speedup_module = ""
                # Unload LCM LoRa weights
                self.ldm.unload_lora_weights()
                # Switch scheduler back to default scheduler
                self.ldm.scheduler = DPMSolverMultistepScheduler.from_config(self.ldm.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
            # Set appropriate inference step for speedup module used
            if self.speedup_module == "LCM":
                kwargs["guidance_scale"] = 1.0
                kwargs["num_inference_steps"] = 8
            elif self.speedup_module == "Lightning":
                kwargs["num_inference_steps"] = 8
            elif self.speedup_module == "":
                kwargs["num_inference_steps"] = 25
        elif SD15_BASE_MODEL_ID in self.model_id or ABSOLUTE_REALITY_BASE_MODEL_ID in self.model_id or EPIC_REALISM_BASE_MODEL_ID in self.model_id or DREAMSHAPER_BASE_MODEL_ID in self.model_id or REALISTIC_VISION_MODEL_ID in self.model_id:
            # (un)load speedup module
            if kwargs["speedup_module"] == "LCM" and self.speedup_module != "LCM":
                self.speedup_module = "LCM"
                # Unload LCM LoRa weights
                self.ldm.unload_lora_weights()
                # Switch to LCM scheduler
                self.ldm.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
                self.ldm.scheduler = LCMScheduler.from_config(self.ldm.scheduler.config)
                self.ldm.fuse_lora()
            elif kwargs["speedup_module"] == "" and self.speedup_module != "":
                self.speedup_module = ""
                # Unload LCM LoRa weights
                self.ldm.unload_lora_weights()
                # Switch scheduler back to default scheduler
                self.ldm.scheduler = DPMSolverMultistepScheduler.from_config(self.ldm.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
            # Set appropriate inference step for speedup module used
            if self.speedup_module == "LCM":
                kwargs["guidance_scale"] = 1.0
                kwargs["num_inference_steps"] = 8
            elif self.speedup_module == "":
                kwargs["num_inference_steps"] = 25
        else:
            kwargs["num_inference_steps"] = 50

        if "speedup_module" in kwargs:
            del kwargs["speedup_module"]
        if "animate_module" in kwargs:
            del kwargs["animate_module"]


        return self.ldm(prompt, **kwargs).images

    def __str__(self) -> str:
        return f"TextToImagePipeline model_id={self.model_id}"
