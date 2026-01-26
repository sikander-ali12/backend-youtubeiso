import os
import sys
import uuid
import shutil
import subprocess
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
import yt_dlp

# Fix Windows encoding issues with special characters
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Directory to store temporary audio files
AUDIO_DIR = "audio_files"
SEPARATED_DIR = "separated_audio"


# Custom CORS middleware that definitely works
class CORSMiddlewareCustom(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Handle preflight OPTIONS request
        if request.method == "OPTIONS":
            response = Response(content="OK", status_code=200)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Access-Control-Max-Age"] = "86400"
            return response
        
        # Process the actual request
        response = await call_next(request)
        
        # Add CORS headers to all responses
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(SEPARATED_DIR, exist_ok=True)
    yield
    # Shutdown - cleanup audio files
    if os.path.exists(AUDIO_DIR):
        shutil.rmtree(AUDIO_DIR)
    if os.path.exists(SEPARATED_DIR):
        shutil.rmtree(SEPARATED_DIR)


app = FastAPI(
    title="YouTube to Audio Converter",
    description="API to convert YouTube videos to audio files with optional vocal isolation",
    version="1.0.0",
    lifespan=lifespan,
)

# Add custom CORS middleware (this one definitely works)
app.add_middleware(CORSMiddlewareCustom)


class YouTubeRequest(BaseModel):
    url: str
    extract_vocals: bool = False
    chunk_duration: int = 20  # Chunk duration in seconds (default 20s)

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "extract_vocals": True,
                "chunk_duration": 20
            }
        }


class ConversionResponse(BaseModel):
    message: str
    filename: str
    title: str


def download_youtube_audio(url: str, unique_id: str) -> tuple[str, str]:
    """Download YouTube video as audio and return the file path and title."""
    # Use restrictfilenames-compatible template to avoid encoding issues
    output_template = os.path.join(AUDIO_DIR, f"{unique_id}.%(ext)s")

    ydl_opts = {
        "format": "bestaudio*",  # More flexible format selection
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",  # Use WAV for better quality processing
                "preferredquality": "192",
            }
        ],
        "outtmpl": output_template,
        "quiet": False,  # Enable output to see what's happening
        "no_warnings": False,
        # Multiple strategies to bypass 403 errors
        "extractor_args": {
            "youtube": {
                "player_client": ["android"],  # Start with android only
                "player_skip": ["webpage"],  # Skip webpage parsing
            }
        },
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "referer": "https://www.youtube.com/",
        "nocheckcertificate": True,
        # Retry logic
        "retries": 3,
        "fragment_retries": 3,
        # Force UTF-8 encoding
        "encoding": "utf-8",
        "restrictfilenames": True,  # Use only ASCII characters in filenames to avoid encoding issues
    }

    # Try multiple strategies with different formats
    strategies = [
        {"format": "bestaudio*", "player_client": ["android"]},
        {"format": "bestaudio", "player_client": ["ios"]},
        {"format": "worst*", "player_client": ["android"]},  # Lower quality but more available
        {"format": "bestaudio/best", "player_client": []},  # Default with no special client
    ]
    
    last_error = None
    for i, strategy in enumerate(strategies):
        try:
            # Create a copy of options for this strategy
            current_opts = ydl_opts.copy()
            current_opts["format"] = strategy["format"]
            
            if strategy["player_client"]:
                current_opts["extractor_args"] = {
                    "youtube": {
                        "player_client": strategy["player_client"],
                        "player_skip": ["webpage"],
                    }
                }
            else:
                # Remove extractor_args for default strategy
                current_opts.pop("extractor_args", None)
            
            try:
                print(f"[DEBUG] Attempt {i+1}/{len(strategies)}: format={strategy['format']}, client={strategy['player_client']}")
            except Exception:
                pass  # Ignore print errors
            
            with yt_dlp.YoutubeDL(current_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                # Safely get title, replacing problematic characters
                title = info.get("title", "audio")
                safe_title = title.encode('ascii', 'ignore').decode('ascii') if title else "audio"
                if not safe_title:
                    safe_title = "audio"
                
                try:
                    print(f"[SUCCESS] Downloaded: {safe_title}")
                except Exception:
                    print(f"[SUCCESS] Downloaded video")
                
                # Find the downloaded file - should be named {unique_id}.wav now
                audio_path = os.path.join(AUDIO_DIR, f"{unique_id}.wav")
                
                if not os.path.exists(audio_path):
                    # Fallback: search for any wav file starting with unique_id
                    for file in os.listdir(AUDIO_DIR):
                        if file.startswith(unique_id) and file.endswith(".wav"):
                            audio_path = os.path.join(AUDIO_DIR, file)
                            break
                
                if not audio_path or not os.path.exists(audio_path):
                    raise FileNotFoundError("Audio file was not created")
                
                return audio_path, safe_title
                
        except yt_dlp.utils.DownloadError as e:
            last_error = e
            error_msg = str(e)
            try:
                print(f"[ERROR] Strategy {i+1} failed: {error_msg[:200]}")
            except Exception:
                print(f"[ERROR] Strategy {i+1} failed (error message contains special characters)")
            
            # If last strategy, give up
            if i == len(strategies) - 1:
                raise last_error
            
            # Continue to next strategy
            continue
        except Exception as e:
            last_error = e
            try:
                print(f"[ERROR] Unexpected error in strategy {i+1}: {str(e)[:200]}")
            except Exception:
                print(f"[ERROR] Unexpected error in strategy {i+1} (error message contains special characters)")
            
            if i == len(strategies) - 1:
                raise
            continue
    
    # Should never reach here, but just in case
    if last_error:
        raise last_error
    raise RuntimeError("All download strategies failed")


def separate_vocals(audio_path: str, unique_id: str) -> str:
    """
    Use Demucs to separate vocals from the audio.
    Returns the path to the isolated vocals file.
    """
    output_dir = os.path.join(SEPARATED_DIR, unique_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Verify audio file exists and is readable
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    file_size = os.path.getsize(audio_path)
    print(f"[DEBUG] Audio file size: {file_size} bytes")
    if file_size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")
    
    # Get the Python executable from the current environment
    # This ensures we use the venv Python where demucs is installed
    python_exe = sys.executable
    
    # Run demucs to separate vocals
    # Using htdemucs with two-stems mode for MUCH faster processing (2-3x faster than mdx_extra)
    # --two-stems only separates vocals vs everything else (no need to separate drums/bass/other)
    cmd = [
        python_exe, "-m", "demucs",
        "-n", "htdemucs",  # Better quality and supports --two-stems
        "--two-stems", "vocals",  # CRITICAL: Only separate vocals (much faster than 4-stem)
        "-d", "cpu",  # Device
        "--segment", "5",  # Slightly larger segments for better quality (was 3)
        "--overlap", "0.15",  # Better overlap for smoother blending
        "-j", "1",  # Use only 1 job to reduce memory usage
        "-o", output_dir,
        "--mp3",  # Output as MP3
        "--mp3-bitrate", "256",  # Higher quality (was 192)
        audio_path
    ]
    
    try:
        print(f"[DEBUG] Running demucs command: {' '.join(cmd)}")
        print(f"[DEBUG] Audio path: {audio_path}")
        print(f"[DEBUG] Output dir: {output_dir}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',  # Replace problematic characters instead of crashing
            timeout=600,  # 10 minute timeout for processing
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}  # Force UTF-8 for subprocess
        )
        
        # Log output for debugging
        if result.stdout:
            print(f"[DEBUG] Demucs stdout: {result.stdout[-500:]}")  # Last 500 chars
        if result.stderr:
            print(f"[DEBUG] Demucs stderr: {result.stderr[-500:]}")  # Last 500 chars
        print(f"[DEBUG] Demucs return code: {result.returncode}")
        
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            if not error_msg:
                error_msg = "Unknown error (no output from demucs)"
            
            # Handle specific error codes
            if result.returncode == -9:
                raise RuntimeError(
                    "Vocal separation was killed due to insufficient memory. "
                    "The audio file may be too long or the server has limited resources. "
                    "Please try a shorter video or contact support."
                )
            elif result.returncode == 137:
                raise RuntimeError(
                    "Vocal separation was killed (likely out of memory). "
                    "Please try a shorter video."
                )
            
            raise RuntimeError(f"Demucs failed with return code {result.returncode}: {error_msg}")
        
        # Find the vocals file
        # Demucs outputs to: output_dir/<model_name>/<filename>/vocals.mp3
        # (model_name could be mdx_extra, htdemucs, etc.)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        print(f"[DEBUG] Searching for vocals file in: {output_dir}")
        # Search for the vocals file in the output directory
        # mdx_extra outputs: vocals.mp3, drums.mp3, bass.mp3, other.mp3
        # htdemucs with --two-stems outputs: vocals.mp3, no_vocals.mp3
        vocals_path = None
        for root, dirs, files in os.walk(output_dir):
            print(f"[DEBUG] Checking directory: {root}, files: {files}")
            for file in files:
                if file == "vocals.mp3":
                    vocals_path = os.path.join(root, file)
                    print(f"[DEBUG] Found vocals file: {vocals_path}")
                    break
            if vocals_path:
                break
        
        if not vocals_path or not os.path.exists(vocals_path):
            # List all files for debugging
            all_files = []
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    all_files.append(os.path.join(root, file))
            print(f"[ERROR] Vocals file not found. Available files: {all_files}")
            raise FileNotFoundError(f"Vocals file not found in {output_dir}. Available files: {all_files}")
        
        return vocals_path
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Vocal separation timed out. The audio may be too long.")


def convert_to_mp3(audio_path: str, output_path: str) -> str:
    """Convert audio file to MP3 using FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-codec:a", "libmp3lame",
        "-b:a", "192k",
        output_path
    ]

    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True,
        encoding='utf-8',
        errors='replace',
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    )
    
    if result.returncode != 0:
        error_msg = result.stderr.encode('ascii', 'ignore').decode('ascii')
        raise RuntimeError(f"FFmpeg conversion failed: {error_msg}")
    
    return output_path


@app.post(
    "/convert",
    response_class=FileResponse,
    summary="Convert YouTube video to audio",
    description="Takes a YouTube URL and returns the audio as an MP3 file. Optionally extract only vocals.",
    responses={
        200: {
            "description": "Audio file returned successfully",
            "content": {"audio/mpeg": {}},
        },
        400: {"description": "Invalid YouTube URL"},
        500: {"description": "Conversion failed"},
    },
)
async def convert_youtube_to_audio(request: YouTubeRequest):
    """
    Convert a YouTube video to audio (MP3) format.

    - **url**: The full YouTube video URL (e.g., https://www.youtube.com/watch?v=...)
    - **extract_vocals**: If true, removes instruments and returns only the vocals

    Returns the audio file as a downloadable MP3.
    """
    # Validate URL
    print(f"[DEBUG] Validating request URL: {request.url}")
    if not request.url or not request.url.strip():
        print(f"[ERROR] URL is empty")
        raise HTTPException(
            status_code=400,
            detail="URL is required"
        )
    
    if "youtube.com" not in request.url and "youtu.be" not in request.url:
        print(f"[ERROR] Invalid URL format: {request.url}")
        raise HTTPException(
            status_code=400,
            detail="Invalid YouTube URL. Please provide a valid YouTube video URL."
        )
    print(f"[DEBUG] URL validation passed")
    
    unique_id = str(uuid.uuid4())[:8]
    
    try:
        print(f"[DEBUG] Starting conversion for URL: {request.url}")
        print(f"[DEBUG] Extract vocals: {request.extract_vocals}")
        print(f"[DEBUG] Unique ID: {unique_id}")
        
        # Download the audio from YouTube
        print(f"[DEBUG] Downloading audio from YouTube...")
        audio_path, title = download_youtube_audio(request.url, unique_id)
        print(f"[DEBUG] Audio downloaded: {audio_path}")
        print(f"[DEBUG] Title: {title}")
        
        if request.extract_vocals:
            # Separate vocals using Demucs
            vocals_path = separate_vocals(audio_path, unique_id)
            
            return FileResponse(
                path=vocals_path,
                media_type="audio/mpeg",
                filename=f"{title}_vocals.mp3",
                headers={
                    "Content-Disposition": f'attachment; filename="{title}_vocals.mp3"'
                }
            )
        else:
            # Convert to MP3 and return full audio
            mp3_path = audio_path.replace(".wav", ".mp3")
            convert_to_mp3(audio_path, mp3_path)
            
            return FileResponse(
                path=mp3_path,
                media_type="audio/mpeg",
                filename=f"{title}.mp3",
                headers={
                    "Content-Disposition": f'attachment; filename="{title}.mp3"'
                }
            )

    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e)
        print(f"[ERROR] yt_dlp DownloadError: {error_msg}")
        if "Private video" in error_msg or "This video is private" in error_msg:
            detail = "This video is private and cannot be downloaded."
        elif "Video unavailable" in error_msg or "unavailable" in error_msg.lower():
            detail = "This video is unavailable. It may have been deleted or restricted."
        elif "Sign in to confirm your age" in error_msg:
            detail = "This video requires age verification and cannot be downloaded."
        elif "403" in error_msg or "Forbidden" in error_msg:
            detail = "YouTube blocked the download (403 Forbidden). This may be temporary. Please try again in a few minutes, or try a different video. You may need to update yt-dlp: pip install --upgrade yt-dlp"
        elif "HTTP Error" in error_msg:
            detail = f"YouTube download error: {error_msg[:150]}. Please try again or use a different video."
        else:
            detail = f"Failed to download video: {error_msg[:200]}"
        
        print(f"[ERROR] Returning 400 error: {detail}")
        raise HTTPException(
            status_code=400,
            detail=detail
        )
    except FileNotFoundError as e:
        print(f"[ERROR] FileNotFoundError: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Audio processing failed: {str(e)}"
        )
    except RuntimeError as e:
        print(f"[ERROR] RuntimeError: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}"
        )
    except Exception as e:
        print(f"[ERROR] Unexpected exception: {type(e).__name__}: {str(e)}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during conversion: {str(e)}"
        )


@app.post(
    "/extract-vocals",
    response_class=FileResponse,
    summary="Extract vocals from YouTube video",
    description="Takes a YouTube URL, downloads the audio, and returns only the isolated vocals (no instruments)",
    responses={
        200: {
            "description": "Vocals audio file returned successfully",
            "content": {"audio/mpeg": {}},
        },
        400: {"description": "Invalid YouTube URL"},
        500: {"description": "Vocal extraction failed"},
    },
)
async def extract_vocals_from_youtube(request: YouTubeRequest):
    """
    Extract only vocals from a YouTube video (removes all instruments).

    - **url**: The full YouTube video URL (e.g., https://www.youtube.com/watch?v=...)

    Uses AI-powered source separation (Demucs) to isolate vocals.
    Returns the vocals-only audio as a downloadable MP3.
    
    Note: Processing may take a few minutes depending on the video length.
    """
    # Force extract_vocals to True for this endpoint
    request.extract_vocals = True
    return await convert_youtube_to_audio(request)


@app.post(
    "/extract-vocals-video",
    response_class=FileResponse,
    summary="Extract vocals and return video without background music",
    description="Takes a YouTube URL, extracts vocals, and returns the video with only vocals (no background music)",
    responses={
        200: {
            "description": "Video file with vocals only returned successfully",
            "content": {"video/mp4": {}},
        },
        400: {"description": "Invalid YouTube URL"},
        500: {"description": "Processing failed"},
    },
)
async def extract_vocals_video(request: YouTubeRequest):
    """
    Extract vocals from YouTube video and return video with vocals only (no background music).
    
    - **url**: The full YouTube video URL
    
    Returns MP4 video with clean vocals, background music removed.
    """
    print(f"[DEBUG] Starting video + vocals extraction for: {request.url}")
    
    # Validate URL
    if not request.url or not request.url.strip():
        raise HTTPException(status_code=400, detail="URL is required")
    
    if "youtube.com" not in request.url and "youtu.be" not in request.url:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    unique_id = str(uuid.uuid4())[:8]
    
    try:
        # Step 1: Download video with audio
        print(f"[DEBUG] Step 1: Downloading video...")
        video_path = download_youtube_video(request.url, unique_id)
        
        # Step 2: Extract audio from video
        print(f"[DEBUG] Step 2: Extracting audio from video...")
        audio_path = os.path.join(AUDIO_DIR, f"{unique_id}_audio.wav")
        extract_audio_from_video(video_path, audio_path)
        
        # Step 3: Separate vocals using Demucs
        print(f"[DEBUG] Step 3: Separating vocals with AI...")
        vocals_path = separate_vocals(audio_path, unique_id)
        
        # Step 4: Combine video + vocals into new video
        print(f"[DEBUG] Step 4: Combining video with clean vocals...")
        output_path = os.path.join(AUDIO_DIR, f"{unique_id}_vocals_video.mp4")
        combine_video_with_vocals(video_path, vocals_path, output_path)
        
        print(f"[SUCCESS] Video with vocals ready: {output_path}")
        
        return FileResponse(
            path=output_path,
            media_type="video/mp4",
            filename="video_vocals_only.mp4",
            headers={
                "Content-Disposition": 'attachment; filename="video_vocals_only.mp4"'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Video processing failed: {str(e)}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {str(e)}"
        )


def download_youtube_video(url: str, unique_id: str) -> str:
    """Download YouTube video (with audio) and return the file path."""
    output_path = os.path.join(AUDIO_DIR, f"{unique_id}_video.mp4")
    
    base_opts = {
        "outtmpl": output_path,
        "quiet": False,
        "no_warnings": False,
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "referer": "https://www.youtube.com/",
        "nocheckcertificate": True,
        "retries": 3,
        "fragment_retries": 3,
        "encoding": "utf-8",
        "restrictfilenames": True,
    }
    
    # Try multiple strategies with different player clients
    strategies = [
        {"format": "best[ext=mp4]/best", "player_client": ["ios"]},
        {"format": "best[ext=mp4]/best", "player_client": ["android"]},
        {"format": "worst[ext=mp4]/worst", "player_client": ["ios"]},  # Lower quality but more available
        {"format": "best[ext=mp4]/best", "player_client": []},  # Default with no special client
        {"format": "18/22", "player_client": []},  # Specific format IDs (mp4 360p/720p)
    ]
    
    last_error = None
    for i, strategy in enumerate(strategies):
        try:
            current_opts = base_opts.copy()
            current_opts["format"] = strategy["format"]
            
            if strategy["player_client"]:
                current_opts["extractor_args"] = {
                    "youtube": {
                        "player_client": strategy["player_client"],
                        "player_skip": ["webpage"],
                    }
                }
            else:
                current_opts.pop("extractor_args", None)
            
            print(f"[DEBUG] Video download attempt {i+1}/{len(strategies)}: format={strategy['format']}, client={strategy['player_client']}")
            
            with yt_dlp.YoutubeDL(current_opts) as ydl:
                ydl.download([url])
            
            if not os.path.exists(output_path):
                # Check for any mp4 file with the unique_id prefix
                for file in os.listdir(AUDIO_DIR):
                    if file.startswith(unique_id) and file.endswith(".mp4"):
                        output_path = os.path.join(AUDIO_DIR, file)
                        print(f"[DEBUG] Found video file: {output_path}")
                        break
            
            if not os.path.exists(output_path):
                raise FileNotFoundError("Video file was not created")
            
            print(f"[SUCCESS] Video downloaded: {output_path}")
            return output_path
            
        except yt_dlp.utils.DownloadError as e:
            last_error = e
            error_msg = str(e)
            print(f"[ERROR] Strategy {i+1} failed: {error_msg[:200]}")
            
            # If it's a bot detection error, try next strategy
            if "bot" in error_msg.lower() or "Sign in" in error_msg:
                if i == len(strategies) - 1:
                    # Last strategy failed
                    raise HTTPException(
                        status_code=400,
                        detail="YouTube is blocking automated access. Please try again in a few minutes or use a different video."
                    )
                continue
            
            # If last strategy, give up
            if i == len(strategies) - 1:
                if "403" in error_msg or "Forbidden" in error_msg:
                    raise HTTPException(status_code=400, detail="YouTube blocked the download. Please try again later.")
                raise HTTPException(status_code=400, detail=f"Failed to download video: {error_msg[:200]}")
            
            continue
        except Exception as e:
            last_error = e
            print(f"[ERROR] Unexpected error in strategy {i+1}: {str(e)[:200]}")
            if i == len(strategies) - 1:
                raise HTTPException(status_code=400, detail=f"Failed to download video: {str(e)[:200]}")
            continue
    
    # Should never reach here, but just in case
    if last_error:
        raise HTTPException(status_code=400, detail=f"Failed to download video: {str(last_error)[:200]}")
    raise HTTPException(status_code=400, detail="All download strategies failed")


def extract_audio_from_video(video_path: str, audio_path: str) -> str:
    """Extract audio from video file using FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # WAV format
        "-ar", "44100",  # Sample rate
        "-ac", "2",  # Stereo
        audio_path
    ]
    
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True,
        encoding='utf-8',
        errors='replace',
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {result.stderr[:200]}")
    
    return audio_path


def combine_video_with_vocals(video_path: str, vocals_path: str, output_path: str) -> str:
    """Combine video with vocals audio using FFmpeg with proper sync."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,  # Input video
        "-i", vocals_path,  # Input vocals audio
        "-c:v", "libx264",  # Re-encode video for sync (was copy)
        "-preset", "fast",  # Fast encoding
        "-crf", "23",  # Good quality
        "-c:a", "aac",  # Re-encode audio
        "-b:a", "256k",  # Match our demucs output quality
        "-map", "0:v:0",  # Use video from first input
        "-map", "1:a:0",  # Use audio from second input
        "-shortest",  # End when shortest stream ends
        "-vsync", "cfr",  # Constant frame rate for sync
        "-async", "1",  # Audio sync method
        "-movflags", "+faststart",  # Web optimization
        output_path
    ]
    
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True,
        encoding='utf-8',
        errors='replace',
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Video combining failed: {result.stderr[:200]}")
    
    return output_path


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using FFprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get video duration: {result.stderr[:200]}")
    
    try:
        duration = float(result.stdout.strip())
        return duration
    except ValueError:
        raise RuntimeError("Could not parse video duration")


def split_video_into_chunks(video_path: str, unique_id: str, chunk_duration: int = 20) -> list[dict]:
    """
    Split video into chunks of specified duration (default 20 seconds).
    Returns a list of dictionaries with chunk information.
    """
    duration = get_video_duration(video_path)
    chunks = []
    chunk_dir = os.path.join(AUDIO_DIR, f"{unique_id}_chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    
    chunk_index = 0
    start_time = 0.0
    
    while start_time < duration:
        end_time = min(start_time + chunk_duration, duration)
        chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_index:03d}.mp4")
        
        # Extract chunk using FFmpeg with RE-ENCODING for precise timing
        # NOTE: Using -c copy can cause sync issues due to keyframe alignment
        # Re-encoding ensures exact timing and prevents audio/video drift
        actual_duration = end_time - start_time
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),  # Seek to start time
            "-i", video_path,
            "-t", str(actual_duration),  # Exact duration
            "-c:v", "libx264",  # Re-encode video for precise timing
            "-preset", "fast",  # Fast encoding
            "-crf", "23",  # Good quality
            "-c:a", "aac",  # Re-encode audio
            "-b:a", "192k",  # Audio bitrate
            "-movflags", "+faststart",  # web optimization
            "-reset_timestamps", "1",  # Reset timestamps to start at 0
            chunk_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        )
        
        if result.returncode != 0:
            print(f"[WARNING] Failed to create chunk {chunk_index}: {result.stderr[:200]}")
            # If this is the last chunk and it's very short, skip it
            if end_time >= duration:
                break
            start_time += chunk_duration
            continue
        
        if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 0:
            chunks.append({
                "index": chunk_index,
                "path": chunk_path,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time
            })
            chunk_index += 1
        else:
            print(f"[WARNING] Chunk {chunk_index} was not created properly")
        
        start_time += chunk_duration
    
    return chunks


@app.post(
    "/extract-vocals-video-chunks",
    summary="Extract vocals and return video chunks info (10 seconds each)",
    description="Takes a YouTube URL, splits video into chunks, returns chunk metadata. Chunks are processed on-demand.",
    responses={
        200: {
            "description": "Video chunks metadata returned successfully",
            "content": {"application/json": {}},
        },
        400: {"description": "Invalid YouTube URL"},
        500: {"description": "Processing failed"},
    },
)
async def extract_vocals_video_chunks(request: YouTubeRequest):
    """
    Split YouTube video into chunks and return metadata. Chunks are processed on-demand when requested.
    This allows streaming: process chunk 1 → play → process chunk 2 → play, etc.
    
    - **url**: The full YouTube video URL
    - **chunk_duration**: Optional chunk duration in seconds (default: 20)
    
    Returns JSON with chunk metadata (chunks are NOT processed yet).
    """
    print(f"[DEBUG] Starting chunked video setup for: {request.url}")
    
    # Validate URL
    if not request.url or not request.url.strip():
        raise HTTPException(status_code=400, detail="URL is required")
    
    if "youtube.com" not in request.url and "youtu.be" not in request.url:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    unique_id = str(uuid.uuid4())[:8]
    
    # Get chunk duration from request or default to 20 seconds
    chunk_duration = getattr(request, 'chunk_duration', 20) or 20
    
    try:
        # Step 1: Download video with audio
        print(f"[DEBUG] Step 1: Downloading video...")
        video_path = download_youtube_video(request.url, unique_id)
        
        # Step 2: Get video duration and split into chunks (raw chunks, not processed)
        print(f"[DEBUG] Step 2: Splitting video into {chunk_duration}-second chunks...")
        video_chunks = split_video_into_chunks(video_path, unique_id, chunk_duration=chunk_duration)
        
        if not video_chunks:
            raise HTTPException(status_code=500, detail="Failed to create video chunks")
        
        print(f"[DEBUG] Created {len(video_chunks)} raw chunks")
        
        # Return chunk metadata only (chunks will be processed on-demand)
        chunk_metadata = []
        for chunk in video_chunks:
            chunk_metadata.append({
                "index": chunk["index"],
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "duration": chunk["duration"],
                "chunk_id": f"{unique_id}_chunk_{chunk['index']}"
            })
        
        return JSONResponse({
            "unique_id": unique_id,
            "total_chunks": len(chunk_metadata),
            "chunks": chunk_metadata,
            "video_path": video_path  # Keep reference for processing
        })
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Chunked video setup failed: {str(e)}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {str(e)}"
        )


@app.get(
    "/chunk/{unique_id}/{chunk_index}",
    response_class=FileResponse,
    summary="Get a specific video chunk (processes on-demand)",
    description="Processes and returns a specific 10-second video chunk with vocals only. Processes chunk if not already processed.",
    responses={
        200: {
            "description": "Video chunk returned successfully",
            "content": {"video/mp4": {}},
        },
        404: {"description": "Chunk not found"},
        500: {"description": "Chunk processing failed"},
    },
)
async def get_video_chunk(unique_id: str, chunk_index: int):
    """
    Get a specific video chunk by unique_id and chunk index.
    If chunk is not processed yet, processes it on-demand (extract audio → separate vocals → combine).
    This allows streaming: process chunk 1 → play → process chunk 2 → play, etc.
    """
    chunk_path = os.path.join(AUDIO_DIR, f"{unique_id}_chunk_{chunk_index}_vocals.mp4")
    
    # If chunk already processed, return it immediately
    if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 0:
        print(f"[DEBUG] Chunk {chunk_index} already processed, returning cached version")
        return FileResponse(
            path=chunk_path,
            media_type="video/mp4",
            filename=f"chunk_{chunk_index:03d}_vocals.mp4",
            headers={
                "Content-Disposition": f'attachment; filename="chunk_{chunk_index:03d}_vocals.mp4"'
            }
        )
    
    # Chunk not processed yet - process it on-demand
    print(f"[DEBUG] Processing chunk {chunk_index} on-demand...")
    
    try:
        # Find the raw chunk file
        chunk_dir = os.path.join(AUDIO_DIR, f"{unique_id}_chunks")
        raw_chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_index:03d}.mp4")
        
        if not os.path.exists(raw_chunk_path):
            raise HTTPException(status_code=404, detail=f"Raw chunk {chunk_index} not found")
        
        # Process chunk: extract audio → separate vocals → combine
        print(f"[DEBUG] Step 1: Extracting audio from chunk {chunk_index}...")
        chunk_audio_path = os.path.join(AUDIO_DIR, f"{unique_id}_chunk_{chunk_index}_audio.wav")
        extract_audio_from_video(raw_chunk_path, chunk_audio_path)
        
        print(f"[DEBUG] Step 2: Separating vocals from chunk {chunk_index}...")
        chunk_unique_id = f"{unique_id}_chunk_{chunk_index}"
        vocals_path = separate_vocals(chunk_audio_path, chunk_unique_id)
        
        print(f"[DEBUG] Step 3: Combining video with vocals for chunk {chunk_index}...")
        combine_video_with_vocals(raw_chunk_path, vocals_path, chunk_path)
        
        print(f"[SUCCESS] Chunk {chunk_index} processed successfully")
        
        if not os.path.exists(chunk_path):
            raise HTTPException(status_code=500, detail=f"Failed to create processed chunk {chunk_index}")
        
        return FileResponse(
            path=chunk_path,
            media_type="video/mp4",
            filename=f"chunk_{chunk_index:03d}_vocals.mp4",
            headers={
                "Content-Disposition": f'attachment; filename="chunk_{chunk_index:03d}_vocals.mp4"'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Failed to process chunk {chunk_index}: {str(e)}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chunk {chunk_index}: {str(e)}"
        )


@app.get("/", summary="Health check", description="Check if the API is running")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "YouTube to Audio Converter API with Vocal Isolation",
        "docs": "/docs",
        "endpoints": {
            "/convert": "Convert YouTube to audio (optional vocal extraction)",
            "/extract-vocals": "Extract only vocals from YouTube video (MP3)",
            "/extract-vocals-video": "Extract vocals and return video without background music (MP4)",
            "/extract-vocals-video-chunks": "Extract vocals and return video in 20-second chunks",
            "/chunk/{unique_id}/{chunk_index}": "Get a specific video chunk"
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)