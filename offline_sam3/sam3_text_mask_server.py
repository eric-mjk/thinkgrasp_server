import argparse
import base64
import contextlib
import io
import json
import os
import socket
import struct
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image


HOST = "0.0.0.0"
PORT = 5050


def add_local_sam3_to_path() -> None:
    current = Path(__file__).resolve()
    for base in (current.parent, current.parents[1]):
        sam3_root = base / "sam3"
        if sam3_root.exists() and str(sam3_root) not in sys.path:
            sys.path.insert(0, str(sam3_root))
            return


def recv_exact(sock: socket.socket, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Connection closed")
        data += chunk
    return data


def recv_request(sock: socket.socket) -> Tuple[Dict[str, Any], bytes]:
    meta_len_bytes = recv_exact(sock, 4)
    (meta_len,) = struct.unpack(">I", meta_len_bytes)
    metadata = json.loads(recv_exact(sock, meta_len).decode("utf-8"))

    payload_len_bytes = recv_exact(sock, 4)
    (payload_len,) = struct.unpack(">I", payload_len_bytes)
    payload = recv_exact(sock, payload_len)
    return metadata, payload


def send_json(sock: socket.socket, result: Dict[str, Any]) -> None:
    result_bytes = json.dumps(result).encode("utf-8")
    sock.sendall(struct.pack(">I", len(result_bytes)))
    sock.sendall(result_bytes)


def encode_mask_png(mask: np.ndarray) -> str:
    mask_u8 = mask.astype(np.uint8) * 255
    buffer = io.BytesIO()
    Image.fromarray(mask_u8, mode="L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def load_processor(
    device: str,
    confidence_threshold: float,
    checkpoint_path: Optional[str],
    enable_inst_interactivity: bool,
):
    add_local_sam3_to_path()
    import sam3
    import torch
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else contextlib.nullcontext()
    )

    with torch.inference_mode(), autocast:
        model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=device,
            checkpoint_path=checkpoint_path,
            load_from_HF=checkpoint_path is None,
            enable_inst_interactivity=enable_inst_interactivity,
        )
    return Sam3Processor(
        model,
        device=device,
        confidence_threshold=confidence_threshold,
    )


def make_autocast(processor):
    import torch

    use_cuda_autocast = getattr(processor, "device", None) == "cuda"
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_cuda_autocast
        else contextlib.nullcontext()
    )


def run_text_sam3(
    processor,
    image_bytes: bytes,
    prompt: str,
    return_instances: bool,
) -> Dict[str, Any]:
    import torch

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size

    autocast = make_autocast(processor)
    with torch.inference_mode(), autocast:
        state = processor.set_image(image)
        processor.reset_all_prompts(state)
        output = processor.set_text_prompt(state=state, prompt=prompt)

    masks = output["masks"].detach().cpu().squeeze(1).numpy().astype(bool)
    boxes = output["boxes"].detach().cpu().tolist()
    scores = output["scores"].detach().cpu().tolist()

    if masks.shape[0] == 0:
        return {
            "status": "ok",
            "mode": "text",
            "prompt": prompt,
            "image_size": [width, height],
            "mask_shape": [height, width],
            "num_masks": 0,
            "boxes_xyxy": [],
            "scores": [],
            "mask_png_b64": None,
        }

    union_mask = np.any(masks, axis=0)
    result = {
        "status": "ok",
        "mode": "text",
        "prompt": prompt,
        "image_size": [width, height],
        "mask_shape": [height, width],
        "num_masks": int(masks.shape[0]),
        "boxes_xyxy": boxes,
        "scores": scores,
        "mask_png_b64": encode_mask_png(union_mask),
    }

    if return_instances:
        result["instance_masks_png_b64"] = [encode_mask_png(mask) for mask in masks]

    return result


def collect_points(metadata: Dict[str, Any]) -> Tuple[list, list]:
    positive_points = metadata.get("positive_points") or []
    negative_points = metadata.get("negative_points") or []

    point_coords = []
    point_labels = []
    for point in positive_points:
        point_coords.append([float(point[0]), float(point[1])])
        point_labels.append(1)
    for point in negative_points:
        point_coords.append([float(point[0]), float(point[1])])
        point_labels.append(0)
    return point_coords, point_labels


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def format_grounding_result(
    output: Dict[str, Any],
    mode: str,
    width: int,
    height: int,
    return_instances: bool,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    masks = output["masks"].detach().cpu().squeeze(1).numpy().astype(bool)
    boxes = output["boxes"].detach().cpu().tolist()
    scores = output["scores"].detach().cpu().tolist()

    result = {
        "status": "ok",
        "mode": mode,
        "image_size": [width, height],
        "mask_shape": [height, width],
        "num_masks": int(masks.shape[0]),
        "boxes_xyxy": boxes,
        "scores": scores,
        "mask_png_b64": None,
    }
    if extra:
        result.update(extra)

    if masks.shape[0] == 0:
        return result

    union_mask = np.any(masks, axis=0)
    result["mask_png_b64"] = encode_mask_png(union_mask)
    if return_instances:
        result["instance_masks_png_b64"] = [encode_mask_png(mask) for mask in masks]
    return result


def run_text_geometric_sam3(
    processor,
    image_bytes: bytes,
    metadata: Dict[str, Any],
    return_instances: bool,
) -> Dict[str, Any]:
    import torch

    prompt = metadata.get("prompt")
    if not prompt:
        return {
            "status": "error",
            "message": "Text+geometry mode requires metadata prompt.",
        }

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    point_coords, point_labels = collect_points(metadata)
    box = metadata.get("box_xyxy")

    if not point_coords and box is None:
        return run_text_sam3(processor, image_bytes, prompt, return_instances)

    autocast = make_autocast(processor)
    with torch.inference_mode(), autocast:
        state = processor.set_image(image)
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=prompt)

        if point_coords:
            normalized_points = [
                [
                    clamp(point[0] / width, 0.0, 1.0),
                    clamp(point[1] / height, 0.0, 1.0),
                ]
                for point in point_coords
            ]
            points = torch.tensor(
                normalized_points,
                device=processor.device,
                dtype=torch.float32,
            ).view(-1, 1, 2)
            labels = torch.tensor(
                point_labels,
                device=processor.device,
                dtype=torch.long,
            ).view(-1, 1)
            state["geometric_prompt"].append_points(points, labels)

        if box is not None:
            raw_x0, raw_y0, raw_x1, raw_y1 = [float(value) for value in box]
            x0 = clamp(min(raw_x0, raw_x1), 0.0, width)
            x1 = clamp(max(raw_x0, raw_x1), 0.0, width)
            y0 = clamp(min(raw_y0, raw_y1), 0.0, height)
            y1 = clamp(max(raw_y0, raw_y1), 0.0, height)
            cx = ((x0 + x1) * 0.5) / width
            cy = ((y0 + y1) * 0.5) / height
            box_w = (x1 - x0) / width
            box_h = (y1 - y0) / height
            boxes = torch.tensor(
                [cx, cy, box_w, box_h],
                device=processor.device,
                dtype=torch.float32,
            ).view(1, 1, 4)
            box_labels = torch.tensor(
                [True],
                device=processor.device,
                dtype=torch.bool,
            ).view(1, 1)
            state["geometric_prompt"].append_boxes(boxes, box_labels)

        output = processor._forward_grounding(state)

    return format_grounding_result(
        output=output,
        mode="text_geometric",
        width=width,
        height=height,
        return_instances=return_instances,
        extra={
            "prompt": prompt,
            "positive_points": metadata.get("positive_points") or [],
            "negative_points": metadata.get("negative_points") or [],
            "box_xyxy": box,
        },
    )


def run_interactive_sam3(
    processor,
    image_bytes: bytes,
    metadata: Dict[str, Any],
    return_instances: bool,
) -> Dict[str, Any]:
    import torch

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    box = metadata.get("box_xyxy")
    multimask_output = bool(metadata.get("multimask_output", True))
    point_coords, point_labels = collect_points(metadata)

    point_coords_np = (
        np.array(point_coords, dtype=np.float32) if point_coords else None
    )
    point_labels_np = (
        np.array(point_labels, dtype=np.int32) if point_labels else None
    )
    box_np = np.array(box, dtype=np.float32) if box is not None else None

    if point_coords_np is None and box_np is None:
        return {
            "status": "error",
            "message": "Interactive mode requires at least one point or box.",
        }
    if processor.model.inst_interactive_predictor is None:
        return {
            "status": "error",
            "message": "Server was not started with interactive prompt support.",
        }

    autocast = make_autocast(processor)
    with torch.inference_mode(), autocast:
        state = processor.set_image(image)
        masks, scores, _ = processor.model.predict_inst(
            state,
            point_coords=point_coords_np,
            point_labels=point_labels_np,
            box=box_np,
            multimask_output=multimask_output,
        )

    masks = masks.astype(bool)
    scores = scores.tolist()
    if masks.shape[0] == 0:
        return {
            "status": "ok",
            "mode": "interactive",
            "image_size": [width, height],
            "mask_shape": [height, width],
            "num_masks": 0,
            "scores": [],
            "selected_index": None,
            "mask_png_b64": None,
        }

    selected_index = int(np.argmax(scores))
    result = {
        "status": "ok",
        "mode": "interactive",
        "image_size": [width, height],
        "mask_shape": [height, width],
        "num_masks": int(masks.shape[0]),
        "scores": scores,
        "selected_index": selected_index,
        "positive_points": metadata.get("positive_points") or [],
        "negative_points": metadata.get("negative_points") or [],
        "box_xyxy": box,
        "mask_png_b64": encode_mask_png(masks[selected_index]),
    }

    if return_instances:
        result["instance_masks_png_b64"] = [encode_mask_png(mask) for mask in masks]

    return result


def serve(args: argparse.Namespace) -> None:
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading SAM3 on {device} ...")
    processor = load_processor(
        device,
        args.confidence,
        args.checkpoint,
        enable_inst_interactivity=not args.disable_click_prompts,
    )
    print(f"Listening on {args.host}:{args.port}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)

        while True:
            conn, addr = server.accept()
            print(f"Connected from {addr}")
            with conn:
                while True:
                    try:
                        metadata, payload = recv_request(conn)
                    except ConnectionError:
                        print("Client disconnected")
                        break

                    try:
                        mode = metadata.get("mode", "text")
                        if mode == "text":
                            prompt = metadata.get("prompt")
                            if not prompt:
                                result = {
                                    "status": "error",
                                    "message": "Text mode requires metadata prompt.",
                                }
                            else:
                                result = run_text_sam3(
                                    processor=processor,
                                    image_bytes=payload,
                                    prompt=prompt,
                                    return_instances=bool(
                                        metadata.get("return_instances", False)
                                    ),
                                )
                        elif mode == "interactive":
                            result = run_interactive_sam3(
                                processor=processor,
                                image_bytes=payload,
                                metadata=metadata,
                                return_instances=bool(
                                    metadata.get("return_instances", False)
                                ),
                            )
                        elif mode == "text_geometric":
                            result = run_text_geometric_sam3(
                                processor=processor,
                                image_bytes=payload,
                                metadata=metadata,
                                return_instances=bool(
                                    metadata.get("return_instances", False)
                                ),
                            )
                        else:
                            result = {
                                "status": "error",
                                "message": f"Unknown request mode: {mode}",
                            }
                    except Exception as exc:
                        result = {"status": "error", "message": str(exc)}

                    send_json(conn, result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal SAM3 text-prompt mask server over TCP sockets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="Run the SAM3 mask server")
    serve_parser.add_argument("--host", default=HOST)
    serve_parser.add_argument("--port", type=int, default=PORT)
    serve_parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    serve_parser.add_argument("--confidence", type=float, default=0.5)
    serve_parser.add_argument("--checkpoint", default=None)
    serve_parser.add_argument(
        "--disable-click-prompts",
        action="store_true",
        help="Load less model state and reject point/box interactive requests.",
    )
    serve_parser.set_defaults(func=serve)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
