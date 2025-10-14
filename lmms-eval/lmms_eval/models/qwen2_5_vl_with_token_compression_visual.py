import base64
import re
from io import BytesIO
from typing import List, Optional, Tuple, Union
import textwrap

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer
from qwen25vl import Qwen2_5_VLForConditionalGeneration
from token_compression.visionzip_official import Qwen2_5_VLForConditionalGeneration_VisionZip
from token_compression.selector_model import Qwen2_5_VLForConditionalGeneration_Selector
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64
import os

# Added imports for visualization
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")
import argparse

@register_model("qwen2_5_vl_with_token_compression_visual")
class Qwen2_5_VL_with_token_compression_visual(lmms):
    """
    Qwen2.5_VL Model
    "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        method: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs != {}:
            defined_params = {
            "pretrained", "device", "batch_size", "attn_implementation",
            "use_flash_attention_2", "device_map", "max_pixels", "use_cache",
            "min_pixels", "max_num_frames", "method"
            }
            extra_kwargs = {k: v for k, v in kwargs.items() if k not in defined_params}
            self.args = argparse.Namespace(**extra_kwargs)
        
        self.visualization_dir = kwargs.get("visualization_dir", 'visualizations')
        self.budgets = kwargs['budgets']
        
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {"torch_dtype": "auto", "device_map": self.device_map}
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self.method = method
        if self.method == 'visionzip_official':
            print('using visionzip official')
            self._model = Qwen2_5_VLForConditionalGeneration_VisionZip.from_pretrained(pretrained, **model_kwargs).eval() 
            self._model.budget = kwargs['budgets']
        elif self.method == 'selector':
            print('using selector')
            self._model = Qwen2_5_VLForConditionalGeneration_Selector.from_pretrained(pretrained, **model_kwargs).eval()
            self._model.visual.budgets = kwargs['budgets']
        else:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, **model_kwargs).eval() 
        
        if self.method is not None:
            from token_compression.monkeypatch import replace_qwen25vl
            replace_qwen25vl(self.args,self._model,self.method.lower())

        self.max_pixels, self.min_pixels, self.max_num_frames = max_pixels, min_pixels, max_num_frames
        self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n") if reasoning_prompt else None
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt, self.interleave_visuals = system_prompt, interleave_visuals
        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu, self.use_cache = int(batch_size), use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU], "Unsupported distributed type"
            self._model = accelerator.prepare_model(self.model, evaluation_mode=True) if accelerator.distributed_type != DistributedType.FSDP else accelerator.prepare(self.model)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank, self._world_size = self.accelerator.local_process_index, self.accelerator.num_processes
        else:
            self._rank, self._world_size = 0, 1

    @property
    def config(self): return self._config
    @property
    def tokenizer(self): return self._tokenizer
    @property
    def model(self): return self.accelerator.unwrap_model(self._model) if hasattr(self, "accelerator") else self._model
    @property
    def eot_token_id(self): return self.tokenizer.eos_token_id
    @property
    def max_length(self): return self._max_length
    @property
    def batch_size(self): return self.batch_size_per_gpu
    @property
    def device(self): return self._device
    @property
    def rank(self): return self._rank
    @property
    def world_size(self): return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i: new_list.append(j)
        return new_list


    # def _plot_divprune_visualization(self, image, output_filename, question, model_answer):
    #     required_attrs = ['last_divprune_scores', 'last_grid_thw', 'last_selected_indices']
    #     if not all(hasattr(self.model.visual, attr) and getattr(self.model.visual, attr) is not None for attr in required_attrs):
    #         eval_logger.warning(f"Could not find all required info for DivPrune visualization. Skipping for {os.path.basename(output_filename)}.")
    #         return

    #     eval_logger.info(f"  -> Generating DivPrune visualization for {os.path.basename(output_filename)}...")
    #     scores, grid_thw, selected_indices = self.model.visual.last_divprune_scores.float(), self.model.visual.last_grid_thw, self.model.visual.last_selected_indices
        
    #     grid_h, grid_w = grid_thw[0, 1].item(), grid_thw[0, 2].item()
    #     heatmap_shape = (grid_h // 2, grid_w // 2)
    #     heatmap = scores.reshape(heatmap_shape).cpu().numpy()

    #     # Get the NumPy array of the original image
    #     original_np = np.array(image.convert("RGB"))
    #     image_size = (original_np.shape[1], original_np.shape[0])

    #     # Normalize the heatmap for coloring
    #     norm = Normalize(vmin=np.min(heatmap), vmax=np.max(heatmap))
    #     cmap = plt.get_cmap('jet')
    #     alpha = 0.6

    #     # --- Full attention heatmap overlay ---
    #     heat_norm = norm(heatmap)
    #     colored_heatmap_uint8 = (cmap(heat_norm)[:, :, :3] * 255).astype(np.uint8)
    #     heat_pil = Image.fromarray(colored_heatmap_uint8).resize(image_size, Image.NEAREST)
    #     full_overlay_np = (original_np * (1 - alpha) + np.array(heat_pil) * alpha).astype(np.uint8)

    #     # --- Mask image: selected patches are RGB, unselected are light gray
    #     unselected_mask_lowres = np.ones(heatmap_shape, dtype=bool)
    #     rows, cols = np.unravel_index(selected_indices.cpu().numpy(), heatmap_shape)
    #     unselected_mask_lowres[rows, cols] = False
    #     mask_pil = Image.fromarray(unselected_mask_lowres).resize(image_size, Image.NEAREST)
    #     unselected_mask_hires = np.expand_dims(np.array(mask_pil), axis=2)
    #     light_gray_color = np.array([211, 211, 211], dtype=np.uint8)
    #     gray_background_np = np.full_like(original_np, light_gray_color)
    #     masked_image_np = np.where(unselected_mask_hires, gray_background_np, original_np)

    #     # Generate the base filename
    #     output_base = os.path.splitext(output_filename)[0]

    #     # --- Save the original RGB image ---
    #     plt.figure(figsize=(original_np.shape[1]/100, original_np.shape[0]/100), dpi=100)
    #     plt.imshow(original_np)
    #     plt.axis('off')
    #     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    #     plt.savefig(f"{output_base}_RGB.png", bbox_inches='tight', pad_inches=0)
    #     plt.close()

    #     # --- Save the attention heatmap overlay image ---
    #     norm = Normalize(vmin=np.min(heatmap), vmax=np.max(heatmap))
    #     cmap = plt.get_cmap('jet')
    #     heat_norm = norm(heatmap)
    #     colored = (cmap(heat_norm)[:, :, :3] * 255).astype(np.uint8)
    #     heat_pil = Image.fromarray(colored).resize(image_size, Image.NEAREST)
    #     full_overlay_np = (original_np * 0.4 + np.array(heat_pil) * 0.6).astype(np.uint8)
    #     plt.figure(figsize=(full_overlay_np.shape[1]/100, full_overlay_np.shape[0]/100), dpi=100)
    #     plt.imshow(full_overlay_np)
    #     plt.axis('off')
    #     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    #     plt.savefig(f"{output_base}_attention.png", bbox_inches='tight', pad_inches=0)
    #     plt.close()

    #     # --- Save the Mask image ---
    #     plt.figure(figsize=(masked_image_np.shape[1]/100, masked_image_np.shape[0]/100), dpi=100)
    #     plt.imshow(masked_image_np)
    #     plt.axis('off')
    #     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    #     plt.savefig(f"{output_base}_mask.png", bbox_inches='tight', pad_inches=0)
    #     plt.close()

    #     eval_logger.info(f"  --- DivPrune mask visualization saved as: {output_base}_mask.png ---")


    # def _plot_visionzip_visualization(self, image, output_filename, question, model_answer):
    #     """
    #     Generates 3 separate visualizations for the VisionZip method:
    #     1. Original RGB Image -> XXX_RGB.png
    #     2. Full attention overlay -> XXX_attention.png
    #     3. Masked overlay (only on retained patches) -> XXX_mask.png
    #     """
    #     model = self.model
    #     required_attrs = ['last_attention_sum', 'last_grid_thw', 'last_selected_indices']
    #     if not all(hasattr(model.visual, attr) and getattr(model.visual, attr) is not None for attr in required_attrs):
    #         eval_logger.warning(f"Could not find all required info for VisionZip visualization. Skipping for {os.path.basename(output_filename)}.")
    #         return

    #     eval_logger.info(f"  -> Generating VisionZip visualization for {os.path.basename(output_filename)}...")
        
    #     # 1. Retrieve data
    #     attention_sum = model.visual.last_attention_sum.float()
    #     grid_thw = model.visual.last_grid_thw
    #     selected_indices = model.visual.last_selected_indices
        
    #     # 2. Calculate grid shape
    #     grid_h_orig, grid_w_orig = grid_thw[0, 1].item(), grid_thw[0, 2].item()
    #     num_patches = len(attention_sum)
    #     merged_w = grid_w_orig // 2
    #     merged_h = num_patches // merged_w
    #     if merged_h * merged_w != num_patches:
    #         eval_logger.error(f"Cannot recover grid shape for VisionZip. Original: ({grid_h_orig}, {grid_w_orig}), Num Patches: {num_patches}. Skipping visualization.")
    #         return
            
    #     heatmap_shape = (merged_h, merged_w)
    #     heatmap = attention_sum.reshape(heatmap_shape).cpu().numpy()

    #     # Get the NumPy array of the original image
    #     original_np = np.array(image.convert("RGB"))
    #     image_size = (original_np.shape[1], original_np.shape[0])

    #     # Normalize the heatmap for coloring
    #     norm = Normalize(vmin=np.min(heatmap), vmax=np.max(heatmap))
    #     cmap = plt.get_cmap('jet')
    #     alpha = 0.6

    #     # --- Full attention heatmap overlay ---
    #     heat_norm = norm(heatmap)
    #     colored_heatmap_uint8 = (cmap(heat_norm)[:, :, :3] * 255).astype(np.uint8)
    #     heat_pil = Image.fromarray(colored_heatmap_uint8).resize(image_size, Image.NEAREST)
    #     full_overlay_np = (original_np * (1 - alpha) + np.array(heat_pil) * alpha).astype(np.uint8)

    #     # --- Mask image: selected patches are RGB, unselected are light gray
    #     unselected_mask_lowres = np.ones(heatmap_shape, dtype=bool)
    #     rows, cols = np.unravel_index(selected_indices.cpu().numpy(), heatmap_shape)
    #     unselected_mask_lowres[rows, cols] = False # Mark the selected patch as False
    #     mask_pil = Image.fromarray(unselected_mask_lowres).resize(image_size, Image.NEAREST)
    #     unselected_mask_hires = np.expand_dims(np.array(mask_pil), axis=2)
    #     light_gray_color = np.array([211, 211, 211], dtype=np.uint8)
    #     gray_background_np = np.full_like(original_np, light_gray_color)
    #     masked_image_np = np.where(unselected_mask_hires, gray_background_np, original_np)
        

    #     # Generate the base filename
    #     output_base = os.path.splitext(output_filename)[0]

    #     # --- Save the original RGB image ---
    #     plt.figure(figsize=(original_np.shape[1]/100, original_np.shape[0]/100), dpi=100)
    #     plt.imshow(original_np)
    #     plt.axis('off')
    #     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    #     plt.savefig(f"{output_base}_RGB.png", bbox_inches='tight', pad_inches=0)
    #     plt.close()

    #     # --- Save the full attention heatmap overlay image ---
    #     plt.figure(figsize=(full_overlay_np.shape[1]/100, full_overlay_np.shape[0]/100), dpi=100)
    #     plt.imshow(full_overlay_np)
    #     plt.axis('off')
    #     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    #     plt.savefig(f"{output_base}_attention.png", bbox_inches='tight', pad_inches=0)
    #     plt.close()

    #     # --- Save the Mask image ---
    #     plt.figure(figsize=(masked_image_np.shape[1]/100, masked_image_np.shape[0]/100), dpi=100)
    #     plt.imshow(masked_image_np)
    #     plt.axis('off')
    #     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    #     plt.savefig(f"{output_base}_mask.png", bbox_inches='tight', pad_inches=0)
    #     plt.close()

    #     eval_logger.info(f"  --- VisionZip visualizations saved as:\n"
    #                     f"      {output_base}_RGB.png\n"
    #                     f"      {output_base}_attention.png\n"
    #                     f"      {output_base}_mask.png ---")


    # def _plot_selector_visualization(self, image, output_filename, question, model_answer):
    #     """
    #     Generates 3 separate visualizations for the Selector method:
    #     1. Original RGB Image -> XXX_RGB.png
    #     2. Full attention overlay -> XXX_attention.png
    #     3. Masked overlay (only on retained patches) -> XXX_mask.png
    #     """
    #     model = self.model
    #     required_attrs = ['last_combined_scores', 'last_grid_thw', 'last_selected_indices']
    #     if not all(hasattr(model.visual, attr) and getattr(model.visual, attr) is not None for attr in required_attrs):
    #         eval_logger.warning(f"Could not find all required info for Selector visualization. Skipping for {os.path.basename(output_filename)}.")
    #         return

    #     eval_logger.info(f"  -> Generating Selector visualization for {os.path.basename(output_filename)}...")
        
    #     # 1. Retrieve data
    #     attention_sum = model.visual.last_combined_scores.float()
    #     grid_thw = model.visual.last_grid_thw
    #     selected_indices = model.visual.last_selected_indices
        
    #     # 2. Calculate grid shape
    #     grid_h_orig, grid_w_orig = grid_thw[0, 1].item(), grid_thw[0, 2].item()
    #     num_patches = len(attention_sum)
    #     merged_w = grid_w_orig // 2
    #     merged_h = num_patches // merged_w
    #     if merged_h * merged_w != num_patches:
    #         eval_logger.error(f"Cannot recover grid shape for Selector. Original: ({grid_h_orig}, {grid_w_orig}), Num Patches: {num_patches}. Skipping visualization.")
    #         return
            
    #     heatmap_shape = (merged_h, merged_w)
    #     heatmap = attention_sum.reshape(heatmap_shape).cpu().numpy()

    #     # Get the NumPy array of the original image
    #     original_np = np.array(image.convert("RGB"))
    #     image_size = (original_np.shape[1], original_np.shape[0])

    #     # Normalize the heatmap for coloring
    #     norm = Normalize(vmin=np.min(heatmap), vmax=np.max(heatmap))
    #     cmap = plt.get_cmap('jet')
    #     alpha = 0.6

    #     # --- Full attention heatmap overlay ---
    #     heat_norm = norm(heatmap)
    #     colored_heatmap_uint8 = (cmap(heat_norm)[:, :, :3] * 255).astype(np.uint8)
    #     heat_pil = Image.fromarray(colored_heatmap_uint8).resize(image_size, Image.NEAREST)
    #     full_overlay_np = (original_np * (1 - alpha) + np.array(heat_pil) * alpha).astype(np.uint8)

    #     # --- Mask image: selected patches are RGB, unselected are light gray
    #     unselected_mask_lowres = np.ones(heatmap_shape, dtype=bool)
    #     rows, cols = np.unravel_index(selected_indices.cpu().numpy(), heatmap_shape)
    #     unselected_mask_lowres[rows, cols] = False # Mark the selected patch as False
    #     mask_pil = Image.fromarray(unselected_mask_lowres).resize(image_size, Image.NEAREST)
    #     unselected_mask_hires = np.expand_dims(np.array(mask_pil), axis=2)
    #     light_gray_color = np.array([211, 211, 211], dtype=np.uint8)
    #     gray_background_np = np.full_like(original_np, light_gray_color)
    #     masked_image_np = np.where(unselected_mask_hires, gray_background_np, original_np)

    #     # Generate the base filename
    #     output_base = os.path.splitext(output_filename)[0]

    #     # --- Save the original RGB image ---
    #     plt.figure(figsize=(original_np.shape[1]/100, original_np.shape[0]/100), dpi=100)
    #     plt.imshow(original_np)
    #     plt.axis('off')
    #     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    #     plt.savefig(f"{output_base}_RGB.png", bbox_inches='tight', pad_inches=0)
    #     plt.close()

    #     # --- Save Full Attention Overlay ---
    #     plt.figure(figsize=(full_overlay_np.shape[1]/100, full_overlay_np.shape[0]/100), dpi=100)
    #     plt.imshow(full_overlay_np)
    #     plt.axis('off')
    #     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    #     plt.savefig(f"{output_base}_attention.png", bbox_inches='tight', pad_inches=0)
    #     plt.close()

    #     # --- Save Masked (Selected Patches) Overlay ---
    #     plt.figure(figsize=(masked_image_np.shape[1]/100, masked_image_np.shape[0]/100), dpi=100)
    #     plt.imshow(masked_image_np)
    #     plt.axis('off')
    #     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    #     plt.savefig(f"{output_base}_mask.png", bbox_inches='tight', pad_inches=0)
    #     plt.close()

    #     eval_logger.info(f"  --- Selector visualizations saved as:\n"
    #                     f"      {output_base}_RGB.png\n"
    #                     f"      {output_base}_attention.png\n"
    #                     f"      {output_base}_mask.png ---")



    def _plot_divprune_visualization(self, image, output_filename, question, model_answer):
        required_attrs = ['last_divprune_scores', 'last_grid_thw', 'last_selected_indices']
        if not all(hasattr(self.model.visual, attr) and getattr(self.model.visual, attr) is not None for attr in required_attrs):
            eval_logger.warning(f"Could not find all required info for DivPrune visualization. Skipping for {os.path.basename(output_filename)}.")
            return

        eval_logger.info(f"  -> Generating DivPrune visualization for {os.path.basename(output_filename)}...")
        scores, grid_thw, selected_indices = self.model.visual.last_divprune_scores.float(), self.model.visual.last_grid_thw, self.model.visual.last_selected_indices
        
        grid_h, grid_w = grid_thw[0, 1].item(), grid_thw[0, 2].item()
        heatmap_shape = (grid_h // 2, grid_w // 2)
        heatmap = scores.reshape(heatmap_shape).cpu().numpy()

        # Get original image as numpy array
        original_np = np.array(image.convert("RGB"))
        image_size = (original_np.shape[1], original_np.shape[0])

        # Normalize heatmap for coloring
        norm = Normalize(vmin=np.min(heatmap), vmax=np.max(heatmap))
        cmap = plt.get_cmap('jet')
        heat_norm = norm(heatmap)
        colored = (cmap(heat_norm)[:, :, :3] * 255).astype(np.uint8)
        heat_pil = Image.fromarray(colored).resize(image_size, Image.NEAREST)
        full_overlay_np = (original_np * 0.4 + np.array(heat_pil) * 0.6).astype(np.uint8)

        # Create mask for selected patches
        selected_mask = np.ones(heatmap_shape, dtype=bool)
        rows, cols = np.unravel_index(selected_indices.cpu().numpy(), heatmap_shape)
        selected_mask[rows, cols] = False
        mask_pil = Image.fromarray(selected_mask).resize(image_size, Image.NEAREST)
        mask_upscaled = np.expand_dims(np.array(mask_pil), axis=2)
        selected_overlay_np = np.where(mask_upscaled, original_np, full_overlay_np)

        # Generate base filename (remove extension if any)
        output_base = os.path.splitext(output_filename)[0]

        # --- Save Original RGB ---
        plt.figure(figsize=(original_np.shape[1]/100, original_np.shape[0]/100), dpi=100)
        plt.imshow(original_np)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f"{output_base}_RGB.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        # --- Save Full Attention Overlay ---
        plt.figure(figsize=(full_overlay_np.shape[1]/100, full_overlay_np.shape[0]/100), dpi=100)
        plt.imshow(full_overlay_np)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f"{output_base}_attention.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        # --- Save Masked (Selected Patches) Overlay ---
        plt.figure(figsize=(selected_overlay_np.shape[1]/100, selected_overlay_np.shape[0]/100), dpi=100)
        plt.imshow(selected_overlay_np)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f"{output_base}_mask.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        eval_logger.info(f"  --- DivPrune visualizations saved as:\n"
                        f"      {output_base}_RGB.png\n"
                        f"      {output_base}_attention.png\n"
                        f"      {output_base}_mask.png ---")


    def _plot_visionzip_visualization(self, image, output_filename, question, model_answer):
        """
        Generates 3 separate visualizations for the VisionZip method:
        1. Original RGB Image -> XXX_RGB.png
        2. Full attention overlay -> XXX_attention.png
        3. Masked overlay (only on retained patches) -> XXX_mask.png
        """
        model = self.model
        required_attrs = ['last_attention_sum', 'last_grid_thw', 'last_selected_indices']
        if not all(hasattr(model.visual, attr) and getattr(model.visual, attr) is not None for attr in required_attrs):
            eval_logger.warning(f"Could not find all required info for VisionZip visualization. Skipping for {os.path.basename(output_filename)}.")
            return

        eval_logger.info(f"  -> Generating VisionZip visualization for {os.path.basename(output_filename)}...")
        
        # 1. Retrieve data
        attention_sum = model.visual.last_attention_sum.float()
        grid_thw = model.visual.last_grid_thw
        selected_indices = model.visual.last_selected_indices
        
        # 2. Calculate grid shape
        grid_h_orig, grid_w_orig = grid_thw[0, 1].item(), grid_thw[0, 2].item()
        num_patches = len(attention_sum)
        merged_w = grid_w_orig // 2
        merged_h = num_patches // merged_w
        if merged_h * merged_w != num_patches:
            eval_logger.error(f"Cannot recover grid shape for VisionZip. Original: ({grid_h_orig}, {grid_w_orig}), Num Patches: {num_patches}. Skipping visualization.")
            return
            
        heatmap_shape = (merged_h, merged_w)
        heatmap = attention_sum.reshape(heatmap_shape).cpu().numpy()

        # Get original image as numpy array
        original_np = np.array(image.convert("RGB"))
        image_size = (original_np.shape[1], original_np.shape[0])

        # Normalize heatmap for coloring
        norm = Normalize(vmin=np.min(heatmap), vmax=np.max(heatmap))
        cmap = plt.get_cmap('jet')
        alpha = 0.6

        # --- Full attention overlay ---
        heat_norm = norm(heatmap)
        colored_heatmap_uint8 = (cmap(heat_norm)[:, :, :3] * 255).astype(np.uint8)
        heat_pil = Image.fromarray(colored_heatmap_uint8).resize(image_size, Image.NEAREST)
        full_overlay_np = (original_np * (1 - alpha) + np.array(heat_pil) * alpha).astype(np.uint8)

        # --- Masked overlay: only on selected patches ---
        background_mask = np.ones(heatmap_shape, dtype=bool)
        rows, cols = np.unravel_index(selected_indices.cpu().numpy(), heatmap_shape)
        background_mask[rows, cols] = False
        mask_pil = Image.fromarray(background_mask).resize(image_size, Image.NEAREST)
        mask_upscaled = np.expand_dims(np.array(mask_pil), axis=2)
        retained_overlay_np = np.where(mask_upscaled, original_np, full_overlay_np)

        # Generate base filename
        output_base = os.path.splitext(output_filename)[0]

        # --- Save Original RGB ---
        plt.figure(figsize=(original_np.shape[1]/100, original_np.shape[0]/100), dpi=100)
        plt.imshow(original_np)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f"{output_base}_RGB.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        # --- Save Full Attention Overlay ---
        plt.figure(figsize=(full_overlay_np.shape[1]/100, full_overlay_np.shape[0]/100), dpi=100)
        plt.imshow(full_overlay_np)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f"{output_base}_attention.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        # --- Save Masked (Selected Patches) Overlay ---
        plt.figure(figsize=(retained_overlay_np.shape[1]/100, retained_overlay_np.shape[0]/100), dpi=100)
        plt.imshow(retained_overlay_np)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f"{output_base}_mask.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        eval_logger.info(f"  --- VisionZip visualizations saved as:\n"
                        f"      {output_base}_RGB.png\n"
                        f"      {output_base}_attention.png\n"
                        f"      {output_base}_mask.png ---")



    def _plot_selector_visualization(self, image, output_filename, question, model_answer):
        """
        Generates 3 separate visualizations for the Selector method:
        1. Original RGB Image -> XXX_RGB.png
        2. Full attention overlay -> XXX_attention.png
        3. Masked overlay (only on retained patches) -> XXX_mask.png
        """
        model = self.model
        required_attrs = ['last_combined_scores', 'last_grid_thw', 'last_selected_indices']
        if not all(hasattr(model.visual, attr) and getattr(model.visual, attr) is not None for attr in required_attrs):
            eval_logger.warning(f"Could not find all required info for Selector visualization. Skipping for {os.path.basename(output_filename)}.")
            return

        eval_logger.info(f"  -> Generating Selector visualization for {os.path.basename(output_filename)}...")
        
        # 1. Retrieve data
        attention_sum = model.visual.last_combined_scores.float()
        grid_thw = model.visual.last_grid_thw
        selected_indices = model.visual.last_selected_indices
        
        # 2. Calculate grid shape
        grid_h_orig, grid_w_orig = grid_thw[0, 1].item(), grid_thw[0, 2].item()
        num_patches = len(attention_sum)
        merged_w = grid_w_orig // 2
        merged_h = num_patches // merged_w
        if merged_h * merged_w != num_patches:
            eval_logger.error(f"Cannot recover grid shape for Selector. Original: ({grid_h_orig}, {grid_w_orig}), Num Patches: {num_patches}. Skipping visualization.")
            return
            
        heatmap_shape = (merged_h, merged_w)
        heatmap = attention_sum.reshape(heatmap_shape).cpu().numpy()

        # Get original image as numpy array
        original_np = np.array(image.convert("RGB"))
        image_size = (original_np.shape[1], original_np.shape[0])

        # Normalize heatmap for coloring
        norm = Normalize(vmin=np.min(heatmap), vmax=np.max(heatmap))
        cmap = plt.get_cmap('jet')
        alpha = 0.6

        # --- Full attention overlay ---
        heat_norm = norm(heatmap)
        colored_heatmap_uint8 = (cmap(heat_norm)[:, :, :3] * 255).astype(np.uint8)
        heat_pil = Image.fromarray(colored_heatmap_uint8).resize(image_size, Image.NEAREST)
        full_overlay_np = (original_np * (1 - alpha) + np.array(heat_pil) * alpha).astype(np.uint8)

        # --- Masked overlay: only on selected patches ---
        background_mask = np.ones(heatmap_shape, dtype=bool)
        rows, cols = np.unravel_index(selected_indices.cpu().numpy(), heatmap_shape)
        background_mask[rows, cols] = False
        mask_pil = Image.fromarray(background_mask).resize(image_size, Image.NEAREST)
        mask_upscaled = np.expand_dims(np.array(mask_pil), axis=2)
        retained_overlay_np = np.where(mask_upscaled, original_np, full_overlay_np)

        # Generate base filename
        output_base = os.path.splitext(output_filename)[0]

        # --- Save Original RGB ---
        plt.figure(figsize=(original_np.shape[1]/100, original_np.shape[0]/100), dpi=100)
        plt.imshow(original_np)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f"{output_base}_RGB.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        # --- Save Full Attention Overlay ---
        plt.figure(figsize=(full_overlay_np.shape[1]/100, full_overlay_np.shape[0]/100), dpi=100)
        plt.imshow(full_overlay_np)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f"{output_base}_attention.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        # --- Save Masked (Selected Patches) Overlay ---
        plt.figure(figsize=(retained_overlay_np.shape[1]/100, retained_overlay_np.shape[0]/100), dpi=100)
        plt.imshow(retained_overlay_np)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f"{output_base}_mask.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        eval_logger.info(f"  --- Selector visualizations saved as:\n"
                        f"      {output_base}_RGB.png\n"
                        f"      {output_base}_attention.png\n"
                        f"      {output_base}_mask.png ---")




    def generate_until(self, requests: List[Instance]) -> List[str]:
        res, cal_num, input_len = [], 0, 0
        def _collate(x): return -len(self.tokenizer.encode(x[0])), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task, split = task[0], split[0]
            # CORRECTED: Safer way to build visual_list in multi-GPU
            visual_list = [doc_to_visual[i](self.task_dict[task][split][doc_id[i]]) for i in range(len(doc_id))]
            gen_kwargs = all_gen_kwargs[0]
            
            until = gen_kwargs.pop("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str): until = [until]
            until = [item for item in until if item != "\n\n"]
            contexts = list(contexts)

            for i in range(len(contexts)): contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            for i, context in enumerate(contexts):
                message = [{"role": "system", "content": self.system_prompt}]
                if self.reasoning_prompt: contexts[i] = context.strip() + self.reasoning_prompt
                
                processed_visuals = []
                for visual in visual_list[i]:
                    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
                        processed_visuals.append({"type": "video", "video": visual, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})
                    elif isinstance(visual, Image.Image):
                        buffer = BytesIO()
                        visual.convert("RGB").save(buffer, format="JPEG")
                        base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
                        processed_visuals.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}", "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})
                
                message.append({"role": "user", "content": processed_visuals + [{"type": "text", "text": contexts[i]}]})
                batched_messages.append(message)

            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
            image_inputs, video_inputs = process_vision_info(batched_messages)
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                if total_frames - 1 not in indices: indices = np.append(indices, total_frames - 1)
                video_inputs[0] = video_inputs[0][indices]
            
            inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            
            # --- FIX: Manually cache grid_thw for visualization ---
            if self.method in ['selector', 'divprune', "visionzip"] and 'image_grid_thw' in inputs:
                self.model.visual.last_grid_thw = inputs['image_grid_thw']
            # ----------------------------------------------------

            inputs = inputs.to(self.device)

            default_gen_kwargs = {"max_new_tokens": 128, "temperature": 0.0, "top_p": None, "num_beams": 1}
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            if current_gen_kwargs["temperature"] > 0: current_gen_kwargs["do_sample"] = True
            else: current_gen_kwargs["do_sample"], current_gen_kwargs["temperature"], current_gen_kwargs["top_p"] = False, None, None
            
            cont = self.model.generate(**inputs, eos_token_id=self.eot_token_id, pad_token_id=self.tokenizer.pad_token_id, use_cache=self.use_cache, **current_gen_kwargs)
            input_len += inputs.input_ids.shape[1]

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            for gen_ids in generated_ids_trimmed: cal_num += len(gen_ids)
            answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            if self.visualization_dir: 
                final_visualization_dir = self.visualization_dir
            elif hasattr(self.cache_hook, 'log_samples_path') and self.cache_hook.log_samples_path:
                final_visualization_dir = os.path.join(os.path.dirname(self.cache_hook.log_samples_path), "visualizations")
            else: 
                final_visualization_dir = "visualizations"

            if self.method == 'selector':
                for i, ans in enumerate(answers):
                    try:
                        # if self.rank == 0:
                        if True:
                            output_dir_for_task = os.path.join(final_visualization_dir, f"{task}_{self.method}_{self.budgets}")
                            os.makedirs(output_dir_for_task, exist_ok=True)
                            doc_id_val = doc_id[i][0] if isinstance(doc_id[i], tuple) else doc_id[i]
                            output_filename = os.path.join(output_dir_for_task, f"{task}_{doc_id_val}_selector.png")
                            self._plot_selector_visualization(image=visual_list[i][0], output_filename=output_filename, question=contexts[i], model_answer=ans)
                    except Exception as e:
                        eval_logger.error(f"Failed to generate visualization for doc_id {doc_id[i]}: {e}")

            if self.method == 'divprune':
                for i, ans in enumerate(answers):
                    try:
                        if True: # 确保只有主进程保存图片
                            output_dir_for_task = os.path.join(final_visualization_dir, f"{task}_{self.method}_{self.budgets}")
                            os.makedirs(output_dir_for_task, exist_ok=True)
                            doc_id_val = doc_id[i][0] if isinstance(doc_id[i], tuple) else doc_id[i]
                            output_filename = os.path.join(output_dir_for_task, f"{task}_{doc_id_val}_divprune.png")
                            # 调用 divprune 的可视化方法
                            self._plot_divprune_visualization(image=visual_list[i][0], output_filename=output_filename, question=contexts[i], model_answer=ans)
                    except Exception as e:
                        eval_logger.error(f"Failed to generate divprune visualization for doc_id {doc_id[i]}: {e}")
            
            if self.method == 'visionzip':
                for i, ans in enumerate(answers):
                    try:
                        if True:
                            output_dir_for_task = os.path.join(final_visualization_dir, f"{task}_{self.method}_{self.budgets}")
                            os.makedirs(output_dir_for_task, exist_ok=True)
                            doc_id_val = doc_id[i][0] if isinstance(doc_id[i], tuple) else doc_id[i]
                            output_filename = os.path.join(output_dir_for_task, f"{task}_{doc_id_val}_visionzip.png")
                            self._plot_visionzip_visualization(
                                image=visual_list[i][0], 
                                output_filename=output_filename, 
                                question=contexts[i], 
                                model_answer=ans
                            )
                    except Exception as e:
                        eval_logger.error(f"Failed to generate visionzip visualization for doc_id {doc_id[i]}: {e}")
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0: ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        avg_input_token = input_len / len(requests)
        print(f"Generated Input Average {avg_input_token} tokens ===========")
        avg_output_token = cal_num / len(requests)
        print(f"Generated Output Average {avg_output_token} tokens")
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")