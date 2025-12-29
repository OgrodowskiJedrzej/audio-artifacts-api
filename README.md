# Hearing Aid Audio Artifact Classification API

A **machine learning inference API** for classifying audio artifacts in hearing aid test recordings.
Provides programmatic access for automated detection, visualization, and report generation, and integrates with external systems such as the Audio Precision API.

Model training was performed [here](https://github.com/OgrodowskiJedrzej/audio-artifacts-classification?tab=readme-ov-file).

---

## Table of Contents

* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [API Endpoints](#api-endpoints)
* [Project Structure](#project-structure)
* [License](#license)

---

## Features

* ML-powered detection of audio artifacts in test recordings
* Single-file and batch inference endpoints
* Structured prediction output compatible with reporting and visualization tools
* Integration-ready for external APIs (e.g., Audio Precision)
* Health check endpoint for service monitoring

---

## Installation

```bash
# Clone repository
git clone https://github.com/OgrodowskiJedrzej/audio-artifacts-api.git
cd audio-artifacts-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS

# Install dependencies
uv sync

# Run FastAPI server
uvicorn main:app --reload
```

---

## Usage

Once running, the API is available at:

```
http://localhost:8000
```

Interactive Swagger UI:

```
http://localhost:8000/docs
```

---

## API Endpoints

### 1. **Health Check**

**GET** `/health`

Check if the service is running.

**Response:**

```json
{
  "status": "ok"
}
```

---

### 2. **Single-File Inference**

**POST** `/api/v1/inference/file`

Upload a single audio file for artifact classification.

**Request (multipart/form-data):**

| Field | Type       | Description                |
| ----- | ---------- | -------------------------- |
| file  | UploadFile | WAV audio file to classify |

**Response:**

```json
{
  "filename": "example.wav",
  "predicted_class": predicted label
}
```

---

### 3. **Future Integration / Report Generation**

* The API is designed to output structured predictions suitable for visualization dashboards or automated report pipelines.
* Can integrate with external systems such as the Audio Precision API for test automation.

---

## Project Structure

```
app/
├── main.py                 # FastAPI app entrypoint
├── api/
│   └── v1/
│       ├── inference.py    # API endpoints
│       └── router.py       # Router aggregation
├── inference.py            # ML model logic and prediction functions
├── schemas/
│   └── inference.py        # Pydantic request/response models
│── audio_utils.py          # Audio preprocessing helpers
│── utils.py                # Helpers
```

---

## License

MIT License © 2025
