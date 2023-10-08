import gradio as gr
from transformers import pipeline
import numpy as np
from poc_llm import ProofOfConceptLLM

speech_capture_started = False
transcription = " "
output = " "
debug_output = " "
debug_output_2 = " "

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-medium.en")

llm = ProofOfConceptLLM()


def transcribe(stream, stream_chunk, image):
    global transcription, enable_streaming, speech_capture_started, output, debug_output, debug_output_2

    if not stream_chunk:
        return None, "Missing audio", "", "", ""
    if not image:
        return None, "Missing image", "", "", ""

    sr, y = stream_chunk
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    transcription = transcriber({"sampling_rate": sr, "raw": y})["text"]
    llm_result = llm.generate_text(image, transcription)
    output, debug_output, debug_output_2 = llm_result

    return stream, transcription, output, debug_output, debug_output_2

demo = gr.Interface(
    transcribe,
    inputs=[
        "state",
        gr.Audio(source="microphone", label="Speech"),
        gr.Image(type="pil", label="Vision", value="./images/elephants.jpg"),
    ],
    outputs=["state", "text", "text", "text", "text"],
)

demo.launch()
