from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from flask import send_file
from flask import Flask, jsonify, request, render_template
import cv2
import whisper
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from ultralytics import RTDETR
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Project folders
UPLOAD_FOLDER = Path("uploads")
RESULTS_FOLDER = Path("analysis_results")
VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
VIDEO_RESULTS_FOLDER = Path("video_results")
VIDEO_RESULTS_FOLDER.mkdir(exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)
AUDIO_EXTENSIONS = {"wav", "mp3", "m4a", "ogg", "flac"}
AUDIO_RESULTS_FOLDER = Path("audio_results")
AUDIO_RESULTS_FOLDER.mkdir(exist_ok=True)

audio_model = None

# Allowed image types and maximum upload size
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Load the AI model only when needed
model = None


def get_model():
    global model

    if model is None:
        model = RTDETR("rtdetr-l.pt")

    return model
def get_audio_model():
    global audio_model

    if audio_model is None:
        audio_model = whisper.load_model("base")

    return audio_model

def read_image(file):
    """Validate an uploaded image and return it as RGB."""

    filename = secure_filename(file.filename or "")

    if "." not in filename:
        return None

    extension = filename.rsplit(".", 1)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        return None

    try:
        image = Image.open(file.stream)
        image.verify()

        file.stream.seek(0)
        image = Image.open(file.stream)

        return image.convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError):
        return None


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "Image is too large. Maximum size is 10 MB."
    }), 413


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")
    return jsonify({
        "project": "CrimeVision AI",
        "status": "Backend is running successfully",
        "features": [
            "Image upload",
            "AI object detection",
            "JSON analysis reports"
        ]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/upload-image", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]

    if not file.filename:
        return jsonify({"error": "Please select an image"}), 400

    image = read_image(file)

    if image is None:
        return jsonify({
            "error": "Invalid image. Upload a valid JPG or PNG file."
        }), 400

    extension = secure_filename(file.filename).rsplit(".", 1)[1].lower()
    saved_name = f"{uuid.uuid4().hex}.{extension}"
    saved_path = UPLOAD_FOLDER / saved_name

    image.save(saved_path, format="JPEG" if extension in {"jpg", "jpeg"} else "PNG")

    return jsonify({
        "message": "Image uploaded successfully",
        "filename": saved_name,
        "width": image.width,
        "height": image.height,
        "format": image.format or extension.upper()
    }), 201


@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]

    if not file.filename:
        return jsonify({"error": "Please select an image"}), 400

    image = read_image(file)

    if image is None:
        return jsonify({
            "error": "Invalid image. Upload a valid JPG or PNG file."
        }), 400

    try:
        detector = get_model()

        prediction = detector.predict(
            source=image,
            conf=0.25,
            verbose=False
        )[0]

        detections = []

        for box in prediction.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            detections.append({
                "object": prediction.names[class_id],
                "confidence": round(confidence, 3),
                "box": [
                    round(float(value), 1)
                    for value in box.xyxy[0].tolist()
                ]
            })

        analysis_id = uuid.uuid4().hex

        record = {
            "analysis_id": analysis_id,
            "filename": secure_filename(file.filename),
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "model": "RT-DETR",
            "objects_detected": len(detections),
            "detections": detections
        }

        result_path = RESULTS_FOLDER / f"{analysis_id}.json"

        with result_path.open("w", encoding="utf-8") as output:
            json.dump(record, output, indent=2)

        return jsonify(record), 200

    except Exception:
        app.logger.exception("Image analysis failed")

        return jsonify({
            "error": "Image analysis failed. Check the server terminal for details."
        }), 500


@app.route("/analysis/<analysis_id>", methods=["GET"])
def get_analysis(analysis_id):
    # Accept only IDs generated by this application
    if len(analysis_id) != 32 or any(
        char not in "0123456789abcdef" for char in analysis_id
    ):
        return jsonify({"error": "Invalid analysis ID"}), 400

    result_path = RESULTS_FOLDER / f"{analysis_id}.json"

    if not result_path.is_file():
        return jsonify({"error": "Analysis not found"}), 404

    try:
        with result_path.open("r", encoding="utf-8") as file:
            record = json.load(file)

        return jsonify(record), 200

    except (OSError, json.JSONDecodeError):
        return jsonify({"error": "Could not read analysis result"}), 500

@app.route("/transcribe-audio", methods=["POST"])
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["audio"]

    if not file.filename:
        return jsonify({"error": "Please select an audio file"}), 400

    filename = secure_filename(file.filename)
    if "." not in filename:
        return jsonify({"error": "Audio file must have an extension"}), 400

    extension = filename.rsplit(".", 1)[1].lower()

    if extension not in AUDIO_EXTENSIONS:
        return jsonify({
            "error": "Supported formats: WAV, MP3, M4A, OGG, FLAC"
        }), 400

    audio_path = UPLOAD_FOLDER / f"{uuid.uuid4().hex}.{extension}"

    try:
        file.save(audio_path)

        model = get_audio_model()
        result = model.transcribe(str(audio_path))

        analysis_id = uuid.uuid4().hex

        record = {
            "analysis_id": analysis_id,
            "filename": filename,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "model": "Whisper base",
            "language": result.get("language"),
            "transcription": result.get("text", "").strip()
        }

        result_path = AUDIO_RESULTS_FOLDER / f"{analysis_id}.json"

        with result_path.open("w", encoding="utf-8") as output:
            json.dump(record, output, indent=2, ensure_ascii=False)

        return jsonify(record), 200

    except Exception:
        app.logger.exception("Audio transcription failed")
        return jsonify({
            "error": "Audio transcription failed. Check the server terminal."
        }), 500

    finally:
        if audio_path.exists():
            audio_path.unlink()


@app.route("/analyze-video", methods=["POST"])
def analyze_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]

    if not file.filename:
        return jsonify({"error": "Please select a video"}), 400

    filename = secure_filename(file.filename)

    if "." not in filename:
        return jsonify({"error": "Invalid video filename"}), 400

    extension = filename.rsplit(".", 1)[1].lower()

    if extension not in VIDEO_EXTENSIONS:
        return jsonify({
            "error": "Supported formats: MP4, AVI, MOV, MKV"
        }), 400

    video_path = UPLOAD_FOLDER / f"{uuid.uuid4().hex}.{extension}"

    try:
        file.save(video_path)

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return jsonify({"error": "Could not open video"}), 400

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        timeline = []
        frame_interval = max(1, int(fps * 2)) if fps > 0 else 1
        frame_index = 0

        detector = get_model()

        try:
            while True:
                success, frame = cap.read()

                if not success:
                    break

                if frame_index % frame_interval == 0:
                    timestamp = frame_index / fps if fps > 0 else 0

                    prediction = detector.predict(
                        source=frame,
                        conf=0.25,
                        verbose=False
                    )[0]

                    detections = []

                    for box in prediction.boxes:
                        class_id = int(box.cls[0].item())
                        detections.append({
                            "object": prediction.names[class_id],
                            "confidence": round(
                                float(box.conf[0].item()), 3
                            ),
                            "box": [
                                round(float(value), 1)
                                for value in box.xyxy[0].tolist()
                            ]
                        })

                    timeline.append({
                        "timestamp_seconds": round(timestamp, 2),
                        "objects_detected": len(detections),
                        "detections": detections
                    })

                frame_index += 1

                if len(timeline) >= 100:
                    break

        finally:
            cap.release()

        analysis_id = uuid.uuid4().hex

        record = {
            "analysis_id": analysis_id,
            "filename": filename,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "model": "RT-DETR",
            "duration_seconds": round(duration, 2),
            "frames_analyzed": len(timeline),
            "timeline": timeline
        }

        result_path = VIDEO_RESULTS_FOLDER / f"{analysis_id}.json"

        with result_path.open("w", encoding="utf-8") as output:
            json.dump(record, output, indent=2)

        return jsonify(record), 200

    except Exception:
        app.logger.exception("Video analysis failed")
        return jsonify({
            "error": "Video analysis failed. Check the Flask terminal."
        }), 500

    finally:
        if video_path.exists():
            video_path.unlink()


@app.route("/report/<analysis_id>", methods=["GET"])
def download_report(analysis_id):
    if (
        len(analysis_id) != 32
        or any(c not in "0123456789abcdef" for c in analysis_id)
    ):
        return jsonify({"error": "Invalid analysis ID"}), 400

    result_path = RESULTS_FOLDER / f"{analysis_id}.json"

    if not result_path.is_file():
        return jsonify({"error": "Image analysis not found"}), 404

    try:
        with result_path.open("r", encoding="utf-8") as file:
            record = json.load(file)

        pdf_path = RESULTS_FOLDER / f"{analysis_id}.pdf"
        pdf = canvas.Canvas(str(pdf_path), pagesize=A4)
        width, height = A4
        y = height - 50

        pdf.setTitle("CrimeVision AI Analysis Report")
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(50, y, "CrimeVision AI")
        y -= 30

        pdf.setFont("Helvetica-Bold", 13)
        pdf.drawString(50, y, "Image Analysis Report")
        y -= 25

        pdf.setFont("Helvetica", 10)

        details = [
            f"Analysis ID: {record['analysis_id']}",
            f"File: {record['filename']}",
            f"Analyzed at: {record['analyzed_at']}",
            f"Model: {record['model']}",
            f"Objects detected: {record['objects_detected']}",
            "",
            "DETECTION DETAILS"
        ]

        for line in details:
            pdf.drawString(50, y, line)
            y -= 18

        for index, item in enumerate(record["detections"], start=1):
            line = (
                f"{index}. {item['object']} | "
                f"Confidence: {item['confidence']} | "
                f"Box: {item['box']}"
            )

            if y < 60:
                pdf.showPage()
                y = height - 50
                pdf.setFont("Helvetica", 10)

            pdf.drawString(50, y, line[:110])
            y -= 18

        pdf.showPage()
        pdf.save()

        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=f"CrimeVision_Report_{analysis_id}.pdf",
            mimetype="application/pdf"
        )

    except Exception:
        app.logger.exception("PDF report generation failed")
        return jsonify({"error": "Could not generate the PDF report"}), 500

@app.route("/dashboard-stats", methods=["GET"])
def dashboard_stats():
    def count_json_files(folder):
        if not folder.exists():
            return 0
        return sum(1 for path in folder.glob("*.json") if path.is_file())

    image_count = count_json_files(RESULTS_FOLDER)
    audio_count = count_json_files(AUDIO_RESULTS_FOLDER)
    video_count = count_json_files(VIDEO_RESULTS_FOLDER)

    return jsonify({
        "image_analyses": image_count,
        "audio_transcriptions": audio_count,
        "video_analyses": video_count,
        "total_analyses": image_count + audio_count + video_count
    })

if __name__ == "__main__":
    app.run(debug=True)
