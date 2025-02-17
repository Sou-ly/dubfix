import gradio as gr
import requests
import tempfile
import os

DEMO_AUDIO_PATH = "./demo/84_121550_000074_000000.wav"

def get_transcript_server(audio, seed):
    """
    Calls the /transcribe endpoint on the API to run transcription on the input audio.
    Returns the transcript as a string.
    """
    url = "http://127.0.0.1:8000/transcribe"
    
    if audio is None:
        return ""
    
    payload = {
        "seed": str(seed)
    }
    try:
        with open(audio, "rb") as f:
            files = {"audio": ("audio.wav", f, "audio/wav")}
            response = requests.post(url, data=payload, files=files)
        
        if response.status_code == 200:
            json_data = response.json()
            transcript = json_data.get("transcript", "")
            return transcript
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

def edit_audio_server(
    audio,
    orig_transcript,
    target_transcript,
    edit_type,
    left_margin,
    right_margin,
    codec_audio_sr,
    codec_sr,
    top_k,
    top_p,
    temperature,
    kvcache,
    seed,
    silence_tokens,
    stop_repetition
):
    """
    Calls the /edit_audio endpoint on the API.
    Returns the path to the edited audio file.
    """
    url = "http://127.0.0.1:8000/edit_audio"
    
    if audio is None:
        return None

    # Prepare the payload
    payload = {
        "original_transcript": orig_transcript,
        "target_transcript": target_transcript,
        "edit_type": edit_type,
        "left_margin": str(left_margin),
        "right_margin": str(right_margin),
        "codec_audio_sr": str(codec_audio_sr),
        "codec_sr": str(codec_sr),
        "top_k": str(top_k),
        "top_p": str(top_p),
        "temperature": str(temperature),
        "kvcache": str(kvcache),
        "seed": str(seed),
        "silence_tokens": silence_tokens,
        "stop_repetition": str(stop_repetition)
    }

    # Send as multipart form data with audio file
    with open(audio, "rb") as f:
        files = {"audio": ("audio.wav", f, "audio/wav")}
        response = requests.post(url, data=payload, files=files)
    
    if response.status_code == 200:
        # Save the received audio data to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(response.content)
            return tmp.name
    else:
        return f"Error: {response.text}"

def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown(
            "### Speech Editing Demo\n"
            "Click **Edit Audio** to process the demo audio file with the default settings."
        )
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="Input Audio",
                    value=DEMO_AUDIO_PATH,
                    type="filepath"
                )
                orig_transcript = gr.Textbox(
                    label="Original Transcript",
                    value="But when I had approached so near to them The common object, which the sense deceives, Lost not by distance any of its marks,",
                    lines=4
                )
                target_transcript = gr.Textbox(
                    label="Target Transcript",
                    value="But when I saw the mirage of the lake in the distance, which the sense deceives, Lost not by distance any of its marks,",
                    lines=4
                )
                edit_type = gr.Dropdown(
                    label="Edit Type",
                    choices=["substitution", "insertion", "deletion"],
                    value="substitution"
                )
            
            with gr.Column():
                left_margin = gr.Number(label="Left Margin", value=0.08)
                right_margin = gr.Number(label="Right Margin", value=0.08)
                codec_audio_sr = gr.Number(label="Codec Audio SR", value=16000)
                codec_sr = gr.Number(label="Codec SR", value=50)
                top_k = gr.Number(label="Top k", value=0)
                top_p = gr.Number(label="Top p", value=0.8)
                temperature = gr.Number(label="Temperature", value=1.0)
                kvcache = gr.Number(label="Kvcache", value=0)
                silence_tokens = gr.Textbox(
                    label="Silence Tokens",
                    value="1388,1898,131"
                )
                stop_repetition = gr.Number(label="Stop Repetition", value=-1)
                seed = gr.Number(label="Seed", value=1)
        
        edit_button = gr.Button("Edit Audio")
        output_audio = gr.Audio(label="Edited Audio", type="filepath")
        
        edit_button.click(
            fn=edit_audio_server,
            inputs=[
                audio_input, orig_transcript, target_transcript, edit_type,
                left_margin, right_margin, codec_audio_sr, codec_sr,
                top_k, top_p, temperature, kvcache, seed, silence_tokens, stop_repetition
            ],
            outputs=output_audio
        )
        
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch() 