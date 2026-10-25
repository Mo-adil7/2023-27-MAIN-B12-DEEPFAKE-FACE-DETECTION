from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid

from classifier import predict_video

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "development-only-change-this-secret")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = PROJECT_ROOT / "uploads"
RESULTS_FOLDER = PROJECT_ROOT / "results" / "web"
DETECTION_SCRIPT = PROJECT_ROOT / "face_detection_backup.py" / "face_detection_unique.py"
DETECTED_FOLDER = PROJECT_ROOT / "detected_faces"
for directory in (UPLOAD_FOLDER, RESULTS_FOLDER, DETECTED_FOLDER):
    directory.mkdir(parents=True, exist_ok=True)
app.config.update(UPLOAD_FOLDER=str(UPLOAD_FOLDER), MAX_CONTENT_LENGTH=500 * 1024 * 1024)
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "1234")
ALLOWED = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
latest_results = None
_face_lock = threading.Lock()


def valid_run(run_id):
    return re.fullmatch(r"[a-f0-9]{32}", run_id) is not None


def problem(message, status=400):
    if request.accept_mimetypes.best == "application/json":
        return jsonify(error=message), status
    return render_template("index.html", error=message), status


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global latest_results
    face_layout = request.form.get("face_layout", "standard")
    if face_layout not in {"standard", "repeated_3x3"}:
        return problem("Choose a supported face counting layout.")
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return problem("Choose a video to begin.")
    filename = secure_filename(uploaded.filename)
    if not filename or Path(filename).suffix.lower() not in ALLOWED:
        return problem("Choose an MP4, MOV, AVI, MKV, WEBM or M4V video.")
    run_id = uuid.uuid4().hex
    stored_name = f"{run_id}_{filename}"
    filepath = UPLOAD_FOLDER / stored_name
    folder = RESULTS_FOLDER / run_id
    started = time.perf_counter()
    try:
        uploaded.save(filepath)
        if filepath.stat().st_size == 0:
            return problem("The selected file is empty. Choose another video.")
        result = predict_video(filepath, folder)
    except ValueError as exc:
        app.logger.exception("Video preparation or inference failed")
        return problem(str(exc))
    except Exception:
        app.logger.exception("Classifier failed")
        return problem("Analysis could not finish. Check the PowerShell error, then try again.", 500)

    result.update(run_id=run_id, filename=filename, stored_name=stored_name,
                  file_size_mb=round(filepath.stat().st_size / (1024 * 1024), 2),
                  face_analysis=None, face_warning=None, faces_detected=None,
                  face_layout=face_layout)
    if request.form.get("include_faces") == "on" or face_layout == "repeated_3x3":
        if not DETECTION_SCRIPT.is_file():
            result["face_warning"] = "Optional face analysis script was not found. Video classification completed."
        else:
            try:
                # The existing face script may use shared output folders.
                command = [sys.executable, str(DETECTION_SCRIPT), str(filepath)]
                if face_layout == "repeated_3x3":
                    command += ["--repeated-grid", "3"]
                with _face_lock:
                    process = subprocess.run(
                        command,
                        cwd=str(PROJECT_ROOT), capture_output=True, text=True,
                        timeout=300,
                    )
                if process.returncode != 0:
                    raise RuntimeError(process.stderr[-2000:])
                lines = process.stdout.strip().splitlines()
                payload = json.loads(lines[-1])
                if not isinstance(payload, dict):
                    raise ValueError("Face script must return a JSON object.")
                result["face_analysis"] = payload
                result["faces_detected"] = payload.get("faces_detected")
            except Exception:
                app.logger.exception("Optional face analysis failed")
                result["face_warning"] = "Optional face analysis did not finish. The classifier result is available below."
    result["elapsed"] = round(time.perf_counter() - started, 2)
    try:
        (folder / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    except OSError:
        app.logger.exception("Could not save the result")
        return problem("The result could not be saved. Check available disk space.", 500)
    latest_results = result
    target = url_for("result_page", run_id=run_id)
    if request.accept_mimetypes.best == "application/json":
        return jsonify(redirect=target)
    return redirect(target, code=303)


@app.route("/results/<run_id>")
def result_page(run_id):
    if not valid_run(run_id):
        return problem("This analysis could not be found.", 404)
    try:
        results = json.loads((RESULTS_FOLDER / run_id / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return problem("This analysis could not be found.", 404)
    return render_template("result.html", results=results)


@app.route("/analysis-assets/<run_id>/<filename>")
def analysis_asset(run_id, filename):
    if not valid_run(run_id) or not re.fullmatch(r"frame_0[0-7]\.png", filename):
        return "Not found", 404
    return send_from_directory(RESULTS_FOLDER / run_id, filename)


@app.route("/video/<run_id>")
def uploaded_video(run_id):
    if not valid_run(run_id):
        return "Not found", 404
    try:
        result = json.loads((RESULTS_FOLDER / run_id / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "Not found", 404
    return send_from_directory(UPLOAD_FOLDER, result["stored_name"], conditional=True)


@app.route("/report/<run_id>")
def download_report(run_id):
    if not valid_run(run_id):
        return "Not found", 404
    return send_from_directory(RESULTS_FOLDER / run_id, "result.json", as_attachment=True,
                               download_name=f"deepfake-analysis-{run_id[:8]}.json")


@app.route("/admin/access", methods=["GET", "POST"])
def admin_access():
    if request.method == "POST":
        if request.form.get("username", "") == ADMIN_USERNAME and request.form.get("password", "") == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            return redirect(url_for("admin"))
        return render_template("admin_access.html", error="Invalid username or password")
    return render_template("admin_access.html")


@app.route("/admin")
def admin():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_access"))
    return render_template("admin.html", results=latest_results)


@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("admin_access"))


@app.errorhandler(413)
def file_too_large(_error):
    return problem("This file exceeds the 500 MB upload limit. Choose a smaller video.", 413)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
