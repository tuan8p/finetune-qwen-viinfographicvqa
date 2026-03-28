"""Qwen2.5-VL model wrapper for multi-image VQA."""

import torch
from qwen_vl_utils import process_vision_info
from PIL import Image
from src.inference_core.model_utils import load_qwenvl_model_and_processor
from src.inference.multi.models.base_model import MultiImageVQAModel
from src.common.utils import format_user_input, get_system_prompt, parse_answer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QwenVLModel(MultiImageVQAModel):
    """Qwen2.5-VL model for multi-image VQA."""

    def __init__(
        self,
        model_path: str = None,
        load_test: bool = False,
        attn_implementation: str | None = "flash_attention_2",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.load_test = load_test
        self.attn_implementation = attn_implementation
        self.model_path = model_path
        if not self.model_path:
            raise ValueError("QwenVLModel requires an explicit model_path from inference config.")
        self._set_clean_model_name()
        self.load_model()

    def load_model(self) -> None:
        self.model, self.processor = load_qwenvl_model_and_processor(
            self.model_path,
            load_test=self.load_test,
            attn_implementation=self.attn_implementation,
        )

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
