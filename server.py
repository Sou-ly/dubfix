import os
import torch
import torchaudio
import numpy as np
import random
from argparse import Namespace
from typing import List, Dict, Tuple, Optional
from data.tokenizer import AudioTokenizer, TextTokenizer
from models import voicecraft
from fastapi import FastAPI, HTTPException, Response, File, UploadFile, Form
from pydantic import BaseModel
import logging
from inference_speech_editing_scale import inference_one_sample
from inference_speech_editing_scale import get_mask_interval
from edit_utils import get_span
import uvicorn
import uuid

app = FastAPI()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TranscriptEdit(BaseModel):
    original_transcript: str
    target_transcript: str
    edit_type: str = "substitution"
    left_margin: float = 0.08
    right_margin: float = 0.08
    codec_audio_sr: int = 16000
    codec_sr: int = 50
    top_k: int = 0
    top_p: float = 0.8
    temperature: float = 1.0
    kvcache: int = 0
    seed: int = 1
    silence_tokens: str = "1388,1898,131"
    stop_repetition: int = -1

@app.on_event("startup")
async def startup_event():
    """Initialize model and tokenizers on startup"""
    global model, audio_tokenizer, text_tokenizer
    
    # Environment setup
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Setup random seeds
    seed_everything()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and tokenizers
    model = voicecraft.VoiceCraft.from_pretrained("pyp1/VoiceCraft_giga330M")
    model.to(device)
    audio_tokenizer, text_tokenizer = setup_tokenizers()

@app.post("/edit_audio")
async def edit_audio(
    audio: UploadFile = File(...),
    original_transcript: str = Form(...),
    target_transcript: str = Form(...),
    edit_type: str = Form("substitution"),
    left_margin: float = Form(0.08),
    right_margin: float = Form(0.08),
    codec_audio_sr: int = Form(16000),
    codec_sr: int = Form(50),
    top_k: int = Form(0),
    top_p: float = Form(0.8),
    temperature: float = Form(1.0),
    kvcache: int = Form(0),
    seed: int = Form(1),
    silence_tokens: str = Form("1388,1898,131"),
    stop_repetition: int = Form(-1)
):
    """Edit audio based on original and target transcripts"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        logger.info("Received edit request")
        logger.info(f"Edit type: {edit_type}")
        logger.info(f"Original transcript: {original_transcript}")
        logger.info(f"Target transcript: {target_transcript}")
        logger.info(f"Parameters: left_margin={left_margin}, right_margin={right_margin}, codec_audio_sr={codec_audio_sr}, codec_sr={codec_sr}")
        logger.info(f"Parameters: top_k={top_k}, top_p={top_p}, temperature={temperature}, kvcache={kvcache}, seed={seed}, stop_repetition={stop_repetition}")
        
        # Set random seed
        logger.info(f"Setting random seed: {seed}")
        seed_everything(seed)
        
        # Save uploaded audio to temporary file
        temp_dir = os.getenv("TMP_PATH", "./demo/temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_audio_path = os.path.join(temp_dir, f"input_{get_random_string()}.wav")
        
        audio_data = await audio.read()
        with open(temp_audio_path, "wb") as f:
            f.write(audio_data)
        
        # Prepare files
        logger.info("Preparing audio files...")
        audio_fn, transcript_fn, align_fn = prepare_audio_files(
            temp_audio_path, 
            original_transcript
        )
        logger.info(f"Files prepared: {audio_fn}, {transcript_fn}, {align_fn}")
        
        # Get edit spans
        logger.info("Getting edit spans...")
        orig_span, new_span = get_span(
            original_transcript, 
            target_transcript, 
            edit_type
        )
        logger.info(f"Spans: original={orig_span}, new={new_span}")
        
        # Process spans
        orig_span_save = [orig_span[0]] if orig_span[0] == orig_span[1] else orig_span
        new_span_save = [new_span[0]] if new_span[0] == new_span[1] else new_span
        
        orig_span_save = ",".join([str(item) for item in orig_span_save])
        new_span_save = ",".join([str(item) for item in new_span_save])
        logger.info(f"Processed spans: original={orig_span_save}, new={new_span_save}")
        
        # Get mask interval
        logger.info("Getting mask interval...")
        start, end = get_mask_interval(align_fn, orig_span_save, edit_type)
        logger.info(f"Mask interval: start={start}, end={end}")
        
        # Convert silence tokens
        logger.info(f"Converting silence tokens: {silence_tokens}")
        silence_tokens = [int(token) for token in silence_tokens.split(",")]
        
        # Setup decode config
        decode_config = {
            'top_k': top_k,
            'top_p': top_p,
            'temperature': temperature,
            'stop_repetition': stop_repetition,
            'kvcache': kvcache,
            'codec_audio_sr': codec_audio_sr,
            'codec_sr': codec_sr,
            'silence_tokens': silence_tokens
        }
        logger.info(f"Decode config: {decode_config}")
        
        # Get morphed span
        logger.info("Calculating morphed span...")
        info = torchaudio.info(audio_fn)
        audio_dur = info.num_frames / info.sample_rate
        morphed_span = (
            max(start - left_margin, 1 / codec_sr),
            min(end + right_margin, audio_dur)
        )
        logger.info(f"Morphed span: {morphed_span}")
        
        mask_interval = [[round(morphed_span[0] * codec_sr), 
                         round(morphed_span[1] * codec_sr)]]
        mask_interval = torch.LongTensor(mask_interval)
        logger.info(f"Mask interval tensor: {mask_interval}")
        
        # Run inference
        logger.info("Running inference...")
        _, gen_audio = inference_one_sample(
            model,
            model.args,  # Use model.args directly
            model.args.phn2num,  # Use model.args.phn2num directly
            text_tokenizer,
            audio_tokenizer,
            audio_fn,
            target_transcript,
            mask_interval,
            device,
            decode_config
        )
        
        # Save audio and return as bytes
        output_path = os.path.join(temp_dir, f"edited_{os.path.basename(audio_fn)}")
        logger.info(f"Saving edited audio to: {output_path}")
        torchaudio.save(output_path, gen_audio[0].cpu(), codec_audio_sr)
        
        with open(output_path, "rb") as f:
            audio_bytes = f.read()
        
        # Clean up temporary files
        for path in [temp_audio_path, audio_fn, transcript_fn, align_fn, output_path]:
            try:
                os.remove(path)
            except:
                pass
        
        return Response(content=audio_bytes, media_type="audio/wav")
        
    except Exception as e:
        logger.error(f"Error in edit_audio: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def setup_environment(cuda_device: str = "0", username: str = "souleyman") -> None:
    """Setup CUDA and environment variables."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    os.environ["USER"] = username

def seed_everything(seed: int = 1) -> None:
    """Set random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_model(model_name: str = "giga330M.pth", device: str = "cuda"):
    """Load the VoiceCraft model."""
    model = voicecraft.VoiceCraft.from_pretrained(f"pyp1/VoiceCraft_{model_name.replace('.pth', '')}")
    model.to(device)
    return model

def setup_tokenizers(encodec_path: str = "./pretrained_models/encodec_4cb2048_giga.th") -> Tuple[AudioTokenizer, TextTokenizer]:
    """Setup audio and text tokenizers."""
    if not os.path.exists(encodec_path):
        os.makedirs(os.path.dirname(encodec_path), exist_ok=True)
        os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th")
        os.system(f"mv encodec_4cb2048_giga.th {encodec_path}")
    
    audio_tokenizer = AudioTokenizer(signature=encodec_path)
    text_tokenizer = TextTokenizer(backend="espeak")
    return audio_tokenizer, text_tokenizer

def prepare_audio_files(
    orig_audio: str,
    orig_transcript: str,
    temp_folder: str = "./demo/temp"
) -> Tuple[str, str, str]:
    """Prepare audio files and transcripts for processing."""
    os.makedirs(temp_folder, exist_ok=True)
    
    # Copy audio file to temp folder
    filename = os.path.splitext(os.path.basename(orig_audio))[0]
    audio_path = f"{temp_folder}/{filename}.wav"
    os.system(f"cp {orig_audio} {audio_path}")
    
    # Create transcript file
    transcript_path = f"{temp_folder}/{filename}.txt"
    with open(transcript_path, "w") as f:
        f.write(orig_transcript)
        
    # Setup MFA alignment
    align_temp = f"{temp_folder}/mfa_alignments"
    os.makedirs(align_temp, exist_ok=True)
    
    # Run MFA alignment with error checking
    mfa_command = f"mfa align -j 1 --clean --overwrite --output_format csv {temp_folder} english_us_arpa english_us_arpa {align_temp}"
    result = os.system(mfa_command)
    
    align_path = f"{align_temp}/{filename}.csv"
    if result != 0 or not os.path.exists(align_path):
        raise Exception("Failed to generate MFA alignment. Please ensure MFA is installed and the audio/transcript are valid.")
    
    return (
        audio_path,
        transcript_path,
        align_path
    )

def get_decode_config(
    top_k: int = 0,
    top_p: float = 0.8,
    temperature: float = 1,
    kvcache: int = 0,
    silence_tokens: List[int] = [1388, 1898, 131],
    stop_repetition: int = -1,
    codec_audio_sr: int = 16000,
    codec_sr: int = 50
) -> Dict:
    """Create decoding configuration dictionary."""
    return {
        'top_k': top_k,
        'top_p': top_p,
        'temperature': temperature,
        'stop_repetition': stop_repetition,
        'kvcache': kvcache,
        'codec_audio_sr': codec_audio_sr,
        'codec_sr': codec_sr,
        'silence_tokens': silence_tokens
    }

def get_random_string():
    """Generate a random string for temporary files"""
    return str(uuid.uuid4())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)