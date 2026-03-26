"""Qwen2.5-VL model wrapper for multi-image VQA."""

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
from src.config import get_model_path
from src.inference.multi.models.base_model import MultiImageVQAModel
from src.common.utils import format_user_input, get_system_prompt, parse_answer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QwenVLModel(MultiImageVQAModel):
    """Qwen2.5-VL model for multi-image VQA."""

    def __init__(self, model_path: str = None, load_test: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.load_test = load_test
        self.model_path = model_path or get_model_path("qwenvl")
        self._set_clean_model_name()
        self.load_model()

    def load_model(self) -> None:
        quant_config = None
        if self.load_test:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=DEVICE,
            quantization_config=quant_config,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
            use_fast=True,
        )
        self.processor.tokenizer.padding_side = "left"

    def infer(self, question: str, images: list[str]) -> str:
        # images are already absolute paths resolved by the dataset
        content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in images]
        content.append({"type": "text", "text": format_user_input(question)})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": get_system_prompt()}]},
            {"role": "user",   "content": content},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad(), torch.autocast(
            device_type="cuda", enabled=torch.cuda.is_available(), dtype=torch.bfloat16
        ):
            generated_ids = self.model.generate(**inputs, max_new_tokens=100)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return parse_answer(output[0])