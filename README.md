# Mutify Backend API

FastAPI backend for Mutify - YouTube vocal extraction service.

## Features

- Download YouTube videos as audio
- Extract vocals using AI (Demucs)
- Combine vocals with video (remove background music)
- RESTful API with automatic documentation

## Setup

### Prerequisites

- Python 3.9+
- FFmpeg (required for audio/video processing)
- pip

### Installation

1. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

2. **Activate virtual environment**
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install FFmpeg**
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
   - Linux: `sudo apt-get install ffmpeg`
   - Mac: `brew install ffmpeg`

## Running Locally

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/

## API Endpoints

### `POST /extract-vocals-video`
Extract vocals from YouTube video and return video with vocals only.

**Request:**
```json
{
  "url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

**Response:** MP4 video file with vocals only

### `POST /extract-vocals`
Extract vocals from YouTube video (audio only).

**Request:**
```json
{
  "url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

**Response:** MP3 audio file with vocals only

### `POST /convert`
Convert YouTube video to audio with optional vocal extraction.

**Request:**
```json
{
  "url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "extract_vocals": false
}
```

**Response:** MP3 audio file

### `GET /`
Health check endpoint.

## Environment Variables

- `PORT` - Server port (default: 8000)
- `PYTHONIOENCODING` - Set to `utf-8` for Windows compatibility

## Deployment

### Render

1. Connect your GitHub repository
2. Create new Web Service
3. Use the provided `render.yaml` or configure manually:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Railway

1. Connect your GitHub repository
2. Railway will auto-detect Python
3. Use the `Procfile` for start command
4. Set environment variables as needed

## Project Structure

```
backend/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── Procfile            # Railway/Heroku deployment
├── render.yaml         # Render deployment config
└── README.md           # This file
```

## Dependencies

- **FastAPI**: Modern web framework
- **Uvicorn**: ASGI server
- **yt-dlp**: YouTube video downloader
- **Demucs**: AI-powered source separation
- **FFmpeg**: Audio/video processing

## Troubleshooting

### FFmpeg not found
- Ensure FFmpeg is installed and in your PATH
- Test with: `ffmpeg -version`

### Demucs installation issues
- Demucs requires PyTorch, which can be large
- Installation may take several minutes
- Ensure you have sufficient disk space

### Port already in use
- Change the port: `uvicorn main:app --port 8001`
- Or kill the process using port 8000

## License

See main project LICENSE file.

