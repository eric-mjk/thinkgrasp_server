# SAM3 Mask Server

The server can run three SAM3 image modes:

- Text/concept mode: send an image plus a prompt such as `"the microwave"`.
- Interactive mode: send an image plus foreground clicks, background clicks, or a box.
- Hybrid mode: send an image plus both a text prompt and point/box geometry.

The client script uses only Python standard-library modules. The local/client PC
does not need `torch`, `PIL`, `numpy`, or SAM3 installed.

## Server PC

Run this on the GPU/server machine where SAM3, PyTorch, CUDA, and the checkpoint
are available.

```bash
python3 /workspace/offline_sam3/sam3_text_mask_server.py serve \
  --host 0.0.0.0 \
  --port 5050 \
  --device cuda
```

Click/box prompt support is enabled by default. To disable it and load less model
state:

```bash
python3 /workspace/offline_sam3/sam3_text_mask_server.py serve \
  --host 0.0.0.0 \
  --port 5050 \
  --device cuda \
  --disable-click-prompts
```

If you have a local checkpoint file:

```bash
python3 /workspace/offline_sam3/sam3_text_mask_server.py serve \
  --host 0.0.0.0 \
  --port 5050 \
  --device cuda \
  --checkpoint /path/to/sam3_checkpoint.pt
```

## Find Server IP

On the server PC:

```bash
hostname -I
```

Use the LAN IP, not `127.0.0.1`.

## Local PC / Client

Replace `192.168.0.71` with the server PC IP.

### Text Prompt

```bash
python3 /workspace/offline_sam3/sam3_text_mask_client.py \
  --server-ip 192.168.0.71 \
  --port 5050 \
  --image /path/to/kitchen.jpg \
  --prompt "the microwave" \
  --out-mask microwave_mask.png
```

### Click Prompt

Send a foreground click in image pixel coordinates without a text prompt:

```bash
python3 /workspace/offline_sam3/sam3_text_mask_client.py \
  --server-ip 192.168.0.71 \
  --port 5050 \
  --image /path/to/kitchen.jpg \
  --point 520 375 \
  --out-mask clicked_object_mask.png
```

Add a background click to exclude another region:

```bash
python3 /workspace/offline_sam3/sam3_text_mask_client.py \
  --server-ip 192.168.0.71 \
  --port 5050 \
  --image /path/to/kitchen.jpg \
  --point 520 375 \
  --negative-point 760 410 \
  --out-mask clicked_object_mask.png
```

For a single click, SAM3 can return multiple masks. The saved mask is the
highest-score mask. To ask for only one mask:

```bash
python3 /workspace/offline_sam3/sam3_text_mask_client.py \
  --server-ip 192.168.0.71 \
  --port 5050 \
  --image /path/to/kitchen.jpg \
  --point 520 375 \
  --single-mask \
  --out-mask clicked_object_mask.png
```

### Box Prompt

Send a box in image pixel coordinates as `x0 y0 x1 y1` without a text prompt:

```bash
python3 /workspace/offline_sam3/sam3_text_mask_client.py \
  --server-ip 192.168.0.71 \
  --port 5050 \
  --image /path/to/kitchen.jpg \
  --box 420 250 760 540 \
  --single-mask \
  --out-mask boxed_object_mask.png
```

### Text + Click Prompt

Send both the text prompt and a positive point. The point is passed in image
pixel coordinates; the server normalizes it internally for SAM3 grounding.

```bash
python3 /workspace/offline_sam3/sam3_text_mask_client.py \
  --server-ip 192.168.0.71 \
  --port 5050 \
  --image /path/to/kitchen.jpg \
  --prompt "the microwave" \
  --point 520 375 \
  --out-mask microwave_at_click_mask.png
```

You can also add a negative/background point:

```bash
python3 /workspace/offline_sam3/sam3_text_mask_client.py \
  --server-ip 192.168.0.71 \
  --port 5050 \
  --image /path/to/kitchen.jpg \
  --prompt "the microwave" \
  --point 520 375 \
  --negative-point 760 410 \
  --out-mask microwave_at_click_mask.png
```

### Text + Box Prompt

```bash
python3 /workspace/offline_sam3/sam3_text_mask_client.py \
  --server-ip 192.168.0.71 \
  --port 5050 \
  --image /path/to/kitchen.jpg \
  --prompt "the microwave" \
  --box 420 250 760 540 \
  --out-mask microwave_in_box_mask.png
```

### Save All Returned Masks

```bash
python3 /workspace/offline_sam3/sam3_text_mask_client.py \
  --server-ip 192.168.0.71 \
  --port 5050 \
  --image /path/to/kitchen.jpg \
  --point 520 375 \
  --return-instances \
  --out-instances-dir clicked_instances \
  --out-mask clicked_object_mask.png
```

## Notes

- Start the server before running the client.
- The server loads SAM3 once, then handles repeated image/prompt requests.
- Text mode follows `sam3_image_predictor_example.ipynb`.
- Point-only and box-only prompts follow `sam3_for_sam1_task_example.ipynb`: the model is
  built with `enable_inst_interactivity=True` and inference uses
  `model.predict_inst(...)`.
- Text+point and text+box use SAM3 grounding through `Sam3Processor` with
  `state["geometric_prompt"].append_points(...)` or `.append_boxes(...)`.
- If no mask is found, the response has `num_masks: 0` and no output mask PNG is written.
