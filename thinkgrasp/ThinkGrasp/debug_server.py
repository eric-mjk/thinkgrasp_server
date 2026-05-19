"""
Minimal debug server — same upload interface as sim_realarm_StartHere.py
but only prints image/depth info instead of running the full pipeline.

Start: python debug_server.py
POST:  curl -F image=@rgb.png -F depth=@depth.png -F text=@instruction.txt \
            http://localhost:5000/grasp_pose
"""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)


def _save_upload(field_name, upload_dir, default_filename):
    f = request.files.get(field_name)
    if f is None or f.filename == "":
        raise ValueError(f"Missing field: {field_name}")
    filename = secure_filename(f.filename) or default_filename
    dest = upload_dir / filename
    f.save(dest)
    return dest


@app.route("/grasp_pose", methods=["POST"])
def grasp_pose():
    try:
        with TemporaryDirectory(prefix="debug_upload_") as tmp:
            upload_dir = Path(tmp)
            rgb_path   = _save_upload("image", upload_dir, "rgb.png")
            depth_path = _save_upload("depth", upload_dir, "depth.png")
            text_path  = _save_upload("text",  upload_dir, "instruction.txt")

            # --- RGB ---
            rgb = np.array(Image.open(rgb_path))
            print("\n=== RGB image ===")
            print(f"  shape : {rgb.shape}")
            print(f"  dtype : {rgb.dtype}")
            print(f"  min/max: {rgb.min()} / {rgb.max()}")

            # --- Depth ---
            depth = np.array(Image.open(depth_path))
            print("\n=== Depth image ===")
            print(f"  shape : {depth.shape}")
            print(f"  dtype : {depth.dtype}")
            print(f"  min   : {depth.min()}")
            print(f"  max   : {depth.max()}")
            print(f"  mean  : {depth.mean():.2f}")
            nonzero = depth[depth > 0]
            if nonzero.size:
                print(f"  nonzero min/max: {nonzero.min()} / {nonzero.max()}")
                print(f"  nonzero mean   : {nonzero.mean():.2f}")
                print(f"  nonzero count  : {nonzero.size} / {depth.size} pixels")
            else:
                print("  WARNING: all depth values are zero!")

            # --- Text ---
            text = text_path.read_text().strip()
            print(f"\n=== Instruction ===\n  {text!r}")
            print()

            return jsonify({"status": "debug ok", "rgb_shape": list(rgb.shape),
                            "depth_shape": list(depth.shape), "text": text})

    except Exception as exc:
        print(f"ERROR: {exc}")
        return jsonify({"error": str(exc)}), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
