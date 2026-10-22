from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
import subprocess
import sys
import json

app = Flask(__name__)

# Basic session configuration
# Change this secret before using the app beyond local testing.
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "development-only-change-this-secret"
)

# Absolute paths based on the project directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
DETECTED_FOLDER = os.path.join(PROJECT_ROOT, "detected_faces")

DETECTION_SCRIPT = os.path.join(
    PROJECT_ROOT,
    "face_detection_backup.py",
    "face_detection_unique.py"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTED_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB limit

# Demo-only admin credentials.
# Replace with environment variables and proper authentication before deployment.
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "1234")

latest_results = None


# ---------------- HOME PAGE ----------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------- ADMIN LOGIN ----------------

@app.route("/admin/access", methods=["GET", "POST"])
def admin_access():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            return redirect(url_for("admin"))

        return render_template(
            "admin_access.html",
            error="Invalid username or password"
        )

    return render_template("admin_access.html")


# ---------------- ADMIN DASHBOARD ----------------

@app.route("/admin")
def admin():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_access"))

    return render_template("admin.html", results=latest_results)


# ---------------- ADMIN LOGOUT ----------------

@app.route("/admin/logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("admin_access"))


# ---------------- VIDEO UPLOAD ----------------

@app.route("/upload", methods=["POST"])
def upload():
    global latest_results

    if "file" not in request.files:
        return render_template(
            "result.html",
            message="No file selected",
            filename=None,
            results=None
        )

    uploaded_file = request.files["file"]

    if not uploaded_file.filename:
        return render_template(
            "result.html",
            message="No file selected",
            filename=None,
            results=None
        )

    filename = secure_filename(uploaded_file.filename)

    if not filename:
        return render_template(
            "result.html",
            message="Invalid filename",
            filename=None,
            results=None
        )

    # Save the upload using an absolute path.
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    uploaded_file.save(filepath)

    # Check the script path before attempting to run it.
    if not os.path.isfile(DETECTION_SCRIPT):
        app.logger.error("Detection script not found: %s", DETECTION_SCRIPT)
        latest_results = None

        return render_template(
            "result.html",
            message="Detection script is missing. Check the project files.",
            filename=filename,
            results=None
        )

    command = [
        sys.executable,
        DETECTION_SCRIPT,
        filepath
    ]

    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=300
        )
    except subprocess.TimeoutExpired:
        latest_results = None
        return render_template(
            "result.html",
            message="Video processing timed out. Try a shorter video.",
            filename=filename,
            results=None
        )
    except OSError:
        app.logger.exception("Could not start the detection process")
        latest_results = None
        return render_template(
            "result.html",
            message="Could not start video detection. Check the server log.",
            filename=filename,
            results=None
        )

    if process.stdout:
        app.logger.info("Detection output:\n%s", process.stdout)

    if process.stderr:
        app.logger.error("Detection errors:\n%s", process.stderr)

    if process.returncode != 0:
        latest_results = None
        return render_template(
            "result.html",
            message="Face detection failed. Check the server terminal for details.",
            filename=filename,
            results=None
        )

    # The detection script must print its JSON result on the final stdout line.
    output_lines = process.stdout.strip().splitlines()
    latest_results = None

    if output_lines:
        try:
            latest_results = json.loads(output_lines[-1])
        except json.JSONDecodeError:
            app.logger.error("Detection script did not return valid JSON.")

    return render_template(
        "result.html",
        message="Detection completed." if latest_results is not None
        else "Processing finished, but no valid result was returned.",
        filename=filename,
        results=latest_results
    )


# ---------------- ERROR HANDLERS ----------------

@app.errorhandler(413)
def file_too_large(_error):
    return "File is too large. The maximum upload size is 500 MB.", 413


# ---------------- RUN APPLICATION ----------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )

