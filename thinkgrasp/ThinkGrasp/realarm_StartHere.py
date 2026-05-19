from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
import os

from flask import jsonify, request
from werkzeug.utils import secure_filename

os.environ.setdefault(
    "THINKGRASP_CHECKPOINT_GRASP_PATH",
    "logs/checkpoint_fgc.tar",
)

from realarm_server_safe import app, get_args_parser, get_grasp_pose as _path_based_get_grasp_pose


def _save_upload(field_name, upload_dir, default_filename):
    uploaded_file = request.files.get(field_name)
    if uploaded_file is None or uploaded_file.filename == "":
        raise ValueError(f"Missing uploaded file field: {field_name}")

    filename = secure_filename(uploaded_file.filename) or default_filename
    destination = upload_dir / filename
    uploaded_file.save(destination)
    return destination


def get_grasp_pose_from_uploads():
    try:
        with TemporaryDirectory(prefix="thinkgrasp_upload_") as tmpdir_name:
            upload_dir = Path(tmpdir_name)
            rgb_image_path = _save_upload("image", upload_dir, "rgb.png")
            depth_image_path = _save_upload("depth", upload_dir, "depth.png")
            text_path = _save_upload("text", upload_dir, "instruction.txt")

            payload = {
                "image_path": str(rgb_image_path),
                "depth_path": str(depth_image_path),
                "text_path": str(text_path),
            }

            with app.test_request_context("/grasp_pose", method="POST", json=payload):
                return _path_based_get_grasp_pose()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


app.view_functions["get_grasp_pose"] = get_grasp_pose_from_uploads


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "ThinkGrasp upload Flask server",
        parents=[get_args_parser()],
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=5000)
