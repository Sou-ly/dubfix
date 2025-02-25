import gradio as gr
import requests
import tempfile
import os
import time

DEMO_AUDIO_PATH = "./demo/84_121550_000074_000000.wav"

def check_server_status():
    """Check if the server is running."""
    try:
        response = requests.get("http://127.0.0.1:8000/docs")
        return response.status_code == 200
    except:
        return False

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
    Returns a tuple of (status_message, audio_file_path).
    """
    url = "http://127.0.0.1:8000/edit_audio"
    
    if audio is None:
        return "Error: No audio input provided.", None
    
    # Check if server is running
    if not check_server_status():
        return "Error: Server is not running. Please start the server with 'python server.py'.", None

    # Prepare the payload
    payload = {
        "original_transcript": orig_transcript,
        "target_transcript": target_transcript,
        "edit_type": edit_type,
        "left_margin": float(left_margin),
        "right_margin": float(right_margin),
        "codec_audio_sr": int(codec_audio_sr),
        "codec_sr": int(codec_sr),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "temperature": float(temperature),
        "kvcache": int(kvcache),
        "seed": int(seed),
        "silence_tokens": silence_tokens,
        "stop_repetition": int(stop_repetition)
    }

    # Send as multipart form data with audio file
    try:
        with open(audio, "rb") as f:
            files = {"audio": ("audio.wav", f, "audio/wav")}
            response = requests.post(url, data=payload, files=files)
        
        if response.status_code == 200:
            # Save the received audio data to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(response.content)
                return "Audio processed successfully!", tmp.name
        else:
            error_msg = f"Error: {response.text}"
            print(f"Server error: {error_msg}")
            return error_msg, None
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"Exception: {error_msg}")
        return error_msg, None

def create_interface():
    server_running = check_server_status()
    
    with gr.Blocks() as demo:
        if not server_running:
            gr.Markdown(
                "### ⚠️ Warning: Server Not Running\n"
                "The VoiceCraft server is not running. Please start it with `python server.py` before using this interface."
            )
            
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
                codec_audio_sr = gr.Number(label="Codec Audio SR", value=16000, precision=0)
                codec_sr = gr.Number(label="Codec SR", value=50, precision=0)
                top_k = gr.Number(label="Top k", value=0, precision=0)
                top_p = gr.Number(label="Top p", value=0.8)
                temperature = gr.Number(label="Temperature", value=1.0)
                kvcache = gr.Number(label="Kvcache", value=0, precision=0)
                silence_tokens = gr.Textbox(
                    label="Silence Tokens",
                    value="1388,1898,131"
                )
                stop_repetition = gr.Number(label="Stop Repetition", value=-1, precision=0)
                seed = gr.Number(label="Seed", value=1, precision=0)
        
        status_msg = gr.Textbox(label="Status", interactive=False)
        edit_button = gr.Button("Edit Audio")
        output_audio = gr.Audio(label="Edited Audio", type="filepath")
        
        edit_button.click(
            fn=edit_audio_server,
            inputs=[
                audio_input, orig_transcript, target_transcript, edit_type,
                left_margin, right_margin, codec_audio_sr, codec_sr,
                top_k, top_p, temperature, kvcache, seed, silence_tokens, stop_repetition
            ],
            outputs=[status_msg, output_audio]
        )
        
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch() 