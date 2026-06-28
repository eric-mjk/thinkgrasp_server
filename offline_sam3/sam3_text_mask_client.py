import argparse
import base64
import json
import socket
import struct
from pathlib import Path
from typing import Any, Dict


PORT = 5050


def recv_exact(sock: socket.socket, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Connection closed")
        data += chunk
    return data


def send_request(sock: socket.socket, metadata: Dict[str, Any], payload: bytes) -> None:
    meta_bytes = json.dumps(metadata).encode("utf-8")
    sock.sendall(struct.pack(">I", len(meta_bytes)))
    sock.sendall(meta_bytes)
    sock.sendall(struct.pack(">I", len(payload)))
    sock.sendall(payload)


def recv_json(sock: socket.socket) -> Dict[str, Any]:
    result_len_bytes = recv_exact(sock, 4)
    (result_len,) = struct.unpack(">I", result_len_bytes)
    return json.loads(recv_exact(sock, result_len).decode("utf-8"))


def write_b64_png(path: Path, png_b64: str) -> None:
    path.write_bytes(base64.b64decode(png_b64.encode("ascii")))


def request(args: argparse.Namespace) -> None:
    image_path = Path(args.image)
    payload = image_path.read_bytes()
    positive_points = args.point or []
    negative_points = args.negative_point or []
    has_prompt = bool(args.prompt)
    has_geometry = bool(positive_points or negative_points or args.box)

    metadata = {"image_name": image_path.name, "return_instances": args.return_instances}
    if has_prompt and has_geometry:
        metadata.update(
            {
                "type": "sam3_text_geometric_mask",
                "mode": "text_geometric",
                "prompt": args.prompt,
                "positive_points": positive_points,
                "negative_points": negative_points,
                "box_xyxy": args.box,
            }
        )
    elif has_geometry:
        metadata.update(
            {
                "type": "sam3_interactive_mask",
                "mode": "interactive",
                "positive_points": positive_points,
                "negative_points": negative_points,
                "box_xyxy": args.box,
                "multimask_output": not args.single_mask,
            }
        )
    elif has_prompt:
        metadata.update(
            {
                "type": "sam3_text_mask",
                "mode": "text",
                "prompt": args.prompt,
            }
        )
    else:
        raise SystemExit("Provide --prompt, --point/--negative-point, or --box.")

    with socket.create_connection((args.server_ip, args.port), timeout=args.timeout) as sock:
        send_request(sock, metadata, payload)
        result = recv_json(sock)

    printable = {k: v for k, v in result.items() if not k.endswith("_b64")}
    print(json.dumps(printable, indent=2))

    if result.get("status") != "ok":
        raise SystemExit(1)

    if args.out_mask and result.get("mask_png_b64"):
        out_mask = Path(args.out_mask)
        write_b64_png(out_mask, result["mask_png_b64"])
        print(f"Wrote union mask to {out_mask}")

    instance_masks = result.get("instance_masks_png_b64") or []
    if args.out_instances_dir and instance_masks:
        out_dir = Path(args.out_instances_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for index, mask_png_b64 in enumerate(instance_masks):
            out_path = out_dir / f"instance_{index:03d}.png"
            write_b64_png(out_path, mask_png_b64)
        print(f"Wrote {len(instance_masks)} instance masks to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Client for the SAM3 text/click mask server. No torch/PIL/numpy required."
    )
    parser.add_argument("--server-ip", required=True)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument(
        "--point",
        type=float,
        nargs=2,
        action="append",
        metavar=("X", "Y"),
        help="Positive foreground click in image pixel coordinates. Can be repeated.",
    )
    parser.add_argument(
        "--negative-point",
        type=float,
        nargs=2,
        action="append",
        metavar=("X", "Y"),
        help="Negative/background click in image pixel coordinates. Can be repeated.",
    )
    parser.add_argument(
        "--box",
        type=float,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Interactive box prompt in image pixel coordinates.",
    )
    parser.add_argument(
        "--single-mask",
        action="store_true",
        help="For interactive mode, ask SAM3 for one mask instead of multimask output.",
    )
    parser.add_argument("--out-mask", default="sam3_mask.png")
    parser.add_argument("--return-instances", action="store_true")
    parser.add_argument("--out-instances-dir", default=None)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    request(args)


if __name__ == "__main__":
    main()
