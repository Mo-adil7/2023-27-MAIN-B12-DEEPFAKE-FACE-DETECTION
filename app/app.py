from flask import Flask, render_template, request, redirect, url_for, session
import os
import subprocess
import sys
import json

app = Flask(__name__)

# Session security
app.secret_key = "deepfake_project_secret_key_2026"

# Folder configuration
UPLOAD_FOLDER = "uploads"
DETECTED_FOLDER = "detected_faces"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTED_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "1234"

# Stores the latest detection results while the server is running
latest_results = None


# ---------------- HOME PAGE ----------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------- ADMIN LOGIN ----------------

@app.route("/admin/access", methods=["GET", "POST"])
def admin_access():

    if request.method == "POST":

        username = request.form.get("username")
        password = request.form.get("password")

        if (
            username == ADMIN_USERNAME
            and password == ADMIN_PASSWORD
        ):
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

    return render_template(
        "admin.html",
        results=latest_results
    )


# ---------------- ADMIN LOGOUT ----------------

@app.route("/admin/logout")
def admin_logout():

    session.pop("admin_logged_in", None)

    return redirect(url_for("admin_access"))


# ---------------- VIDEO UPLOAD ----------------

@app.route("/upload", methods=["POST"])
def upload():

    global latest_results

    # Check whether a file was submitted
    if "file" not in request.files:

        return render_template(
            "result.html",
            message="No file selected",
            filename=None,
            results=None
        )

    file = request.files["file"]

    # Check whether the user selected a file
    if file.filename == "":

        return render_template(
            "result.html",
            message="No file selected",
            filename=None,
            results=None
        )

    # Save the uploaded file
    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    # Run the face detection program
    command = [
        sys.executable,
        "face_detection.py",
        filepath
    ]

    process = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    # Print output in the VS Code terminal
    print(process.stdout)

    if process.stderr:
        print("Face detection errors:")
        print(process.stderr)

    # Check whether face detection completed successfully
    if process.returncode != 0:

        latest_results = None

        return render_template(
            "result.html",
            message="Face detection failed. Please try another video.",
            filename=file.filename,
            results=None
        )

    # Read the JSON results printed by face_detection.py
    output_lines = process.stdout.strip().splitlines()

    latest_results = None

    if output_lines:

        try:
            latest_results = json.loads(output_lines[-1])

        except json.JSONDecodeError:

            latest_results = None

    # Display the result to the user
    return render_template(
        "result.html",
        message="Face detection completed!",
        filename=file.filename,
        results=latest_results
    )


# ---------------- RUN APPLICATION ----------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )