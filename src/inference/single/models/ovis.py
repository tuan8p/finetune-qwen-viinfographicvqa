import torch
from PIL import Image
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from src.inference.single.models.base_model import VQAModel
from src.common.utils import get_system_prompt, parse_answer
from src.config import get_model_path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load full not quantization -> 33gb
class OvisModel(VQAModel):
    def __init__(self, model_path: str = None, load_test: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.load_test = load_test
        self.model_path = model_path or get_model_path("ovis")
        self._set_clean_model_name()
        self.load_model()
        
    def load_model(self):
        quannt_config = None 
        if self.load_test: # if true then 4 bytes load for testing 
            quannt_config = BitsAndBytesConfig(
            load_in_4bit=True,                     
            bnb_4bit_compute_dtype=torch.bfloat16,  
            bnb_4bit_use_double_quant=True,        
            bnb_4bit_quant_type="nf4",             
        )


        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16, multimodal_max_length=32768, trust_remote_code=True, 
            quantization_config = quannt_config, device_map = DEVICE
        ).eval()

    def infer(self, question: str, image_path: str) -> str:
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": [
                {"type": "image", "image": Image.open(image_path).convert('RGB')},
                {"type": "text", "text": f"Question: {question}\nAnswer:"}
            ]}
        ]
        
        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages, add_generation_prompt=True, enable_thinking=False
        )
        input_ids = input_ids.cuda()
        pixel_values = pixel_values.cuda() if pixel_values is not None else None
        grid_thws = grid_thws.cuda() if grid_thws is not None else None

        with torch.inference_mode():
            outputs = self.model.generate(
                inputs=input_ids, pixel_values=pixel_values, grid_thws=grid_thws,
                enable_thinking=False, enable_thinking_budget=False,
                max_new_tokens=100, thinking_budget=0, eos_token_id=self.model.text_tokenizer.eos_token_id
            )

        response = self.model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return parse_answer(response) 