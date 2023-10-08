from transformers import (
    AutoModelForCausalLM,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
)
import torch

AI_MODEL = "Open-Orca/Mistral-7B-OpenOrca"
CAPTION_MODEL = "nlpconnect/vit-gpt2-image-captioning"
PREFIX = "<|im_start|>"
PREFIX_USER = PREFIX + "user\n"
PREFIX_SYSTEM = PREFIX + "system\n"
PREFIX_ASSISTANT = PREFIX + "assistant\n"
SUFFIX = "<|im_end|>\n"
TEMPLATE = (
    PREFIX_SYSTEM
    + """
You are a helpful voice-activated assistant that exists to help the user achieve their goals in a safe manner.
The user may ask you questions about what they are seeing, in which case use the predictions of what an attached image conveys from computer vision models."""
    + SUFFIX
)


class ProofOfConceptLLM:
    def __init__(self):
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            AI_MODEL, load_in_8bit=True
        )
        self.llm_tokenizer = AutoTokenizer.from_pretrained(AI_MODEL)
        self.memory = []
        self.memory_size = len(TEMPLATE)
        self.max_new_tokens = 1024
        self.context_length = 8192
        self.max_memory_size = self.context_length - self.max_new_tokens

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.caption_model = VisionEncoderDecoderModel.from_pretrained(
            CAPTION_MODEL
        ).to(self.device)
        self.caption_feature_extractor = ViTImageProcessor.from_pretrained(
            CAPTION_MODEL
        )
        self.caption_tokenizer = AutoTokenizer.from_pretrained(CAPTION_MODEL)

    def generate_text(self, image, prompt):
        to_append = "".join(
            [
                PREFIX_USER,
                prompt,
                SUFFIX,
                PREFIX_SYSTEM,
                self.describe_image(image),
                SUFFIX,
                PREFIX_ASSISTANT,
            ]
        )

        if len(to_append) > self.max_memory_size:
            return "Please ask a shorter question.", "", ""

        self.memory.append(to_append)
        self.memory_size += len(to_append)

        while self.memory_size > self.max_memory_size:
            self.memory.pop(0)
            self.memory_size -= len(self.memory[0])

        contextualized_prompt = TEMPLATE + "".join(self.memory)

        inputs = self.llm_tokenizer(contextualized_prompt, return_tensors="pt").to(
            self.device
        )
        outputs = self.llm_model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )

        output = self.llm_tokenizer.batch_decode(outputs)[0]
        generated_prefix = " ".join([PREFIX, "assistant"])
        generated_text = output[
            output.rfind(generated_prefix) + len(generated_prefix) :
        ]

        self.memory.append(generated_text)
        self.memory_size += len(generated_text)

        return (
            generated_text[1 : generated_text.find("<|im_end|>")],
            contextualized_prompt,
            output,
        )

    def describe_image(self, image):
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        pixel_values = self.caption_feature_extractor(
            image, return_tensors="pt"
        ).pixel_values.to(self.device)
        output_ids = self.caption_model.generate(
            pixel_values, max_length=64, num_beams=4
        ).to(self.device)
        output = self.caption_tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]

        return (
            "Note: the following is a system message.\nHere are the computer vision predictions of what the attached image entails, do not mention the results unless the user's inquiry is explicitly related to the results:\n"
            + output
            + "."
        )
