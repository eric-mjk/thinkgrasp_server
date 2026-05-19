from pathlib import Path
from tempfile import TemporaryDirectory
import argparse

from flask import jsonify, request
from werkzeug.utils import secure_filename

from realarm_server_safe import (
    Image,
    SHOW_OPEN3D,
    app,
    client,
    create_cropping_box_from_boxes,
    cv2,
    get_args_parser,
    langsam_actor,
    langsamutils,
    load_image_as_base64,
    logging,
    np,
    o3d,
    process_grasping_result,
    ray,
    select_fallback_object,
    utils,
    visualize_cropping_box,
    _save_open3d_vis,
)
from grasp_detetor import Graspnet


SYSTEM_PROMPT = (
    "Given a 640x480 input image and the provided instruction, perform the following steps:\n"
    "Target Object Selection:\n"
    "Identify the object in the image that best matches the instruction. If the target object is found, select it as the target object.\n"
    "If the target object is not visible, select the most cost-effective object or object part considering ease of grasping, importance, and safety.\n"
    "If the object has a handle or a part that is easier or safer to grasp, strongly prefer to select that part.\n"
    "Consider the geometric shape of the objects and the gripper's success rate when selecting the target object or object part.\n"
    "Output the name of the selected object or object part as [object:color and name] or [object part:color and name].\n"
    "Cropping Box Calculation:\n"
    "Calculate a cropping box that includes the target object and all surrounding objects that might be relevant for grasping.\n"
    "Provide the coordinates of the cropping box in the format (top-left x, top-left y, bottom-right x, bottom-right y).\n"
    "Object Properties within Cropping Box:\n"
    "For each object within the cropping box, provide the following properties:\n"
    "Grasping Score: Evaluate the ease or difficulty of grasping the object on a scale from 0 to 100.\n"
    "Material Composition: Evaluate the material composition of the object on a scale from 0 to 100.\n"
    "Surface Texture: Evaluate the texture of the object's surface on a scale from 0 to 100.\n"
    "Stability Assessment: Assess the stability of the object on a scale from 0 to 100.\n"
    "Centroid Coordinates: Provide the coordinates (x, y) of the object's center of mass across the entire image.\n"
    "Preferred Grasping Location: Divide the cropping box into a 3x3 grid and return a number from 1 to 9.\n"
    "Output should be in the following format:\n"
    "Selected Object/Object Part: [object:color and name] or [object part:color and name]\n"
    "Cropping Box Coordinates: (top-left x, top-left y, bottom-right x, bottom-right y)\n"
    "Objects and Their Properties:\n"
    "Object: [color and name]\n"
    "Grasping Score: [value]\n"
    "Material Composition: [value]\n"
    "Surface Texture: [value]\n"
    "Stability Assessment: [value]\n"
    "Centroid Coordinates: (x, y)\n"
    "Preferred Grasping Location: [value]\n"
)


def _save_upload(field_name, upload_dir, default_filename):
    uploaded_file = request.files.get(field_name)
    if uploaded_file is None or uploaded_file.filename == "":
        raise ValueError(f"Missing uploaded file field: {field_name}")

    filename = secure_filename(uploaded_file.filename) or default_filename
    destination = upload_dir / filename
    uploaded_file.save(destination)
    return destination


def _paths_from_request():
    if request.files:
        upload_dir_context = TemporaryDirectory(prefix="thinkgrasp_graspnet_upload_")
        upload_dir = Path(upload_dir_context.name)
        return (
            upload_dir_context,
            _save_upload("image", upload_dir, "rgb.png"),
            _save_upload("depth", upload_dir, "depth.png"),
            _save_upload("text", upload_dir, "instruction.txt"),
        )

    data = request.json
    if not data:
        raise ValueError("Expected multipart uploads or JSON path payload")
    return (
        None,
        Path(data["image_path"]),
        Path(data["depth_path"]),
        Path(data["text_path"]),
    )


def _choose_grasp_with_graspnet(pcd):
    graspnet = Graspnet()
    gg = graspnet.compute_grasp_pose(pcd)
    if gg is None or gg.translations.shape[0] == 0:
        raise RuntimeError("GraspNet returned no grasps")

    gg.nms()
    gg.sort_by_score()
    if gg.translations.shape[0] == 0:
        raise RuntimeError("GraspNet returned no grasps after filtering")

    return gg, gg.translations[0], gg.rotation_matrices[0], gg.depths[0]


def get_grasp_pose_graspnet():
    temp_context = None
    try:
        temp_context, rgb_image_path, depth_image_path, text_path = _paths_from_request()

        rgb_image_path = str(rgb_image_path)
        depth_image_path = str(depth_image_path)
        text_path = str(text_path)

        img_ori = cv2.imread(rgb_image_path)
        if img_ori is None:
            raise ValueError(f"Could not read RGB image: {rgb_image_path}")
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        depth_ori = np.array(Image.open(depth_image_path))
        with open(text_path, "r") as file:
            input_text = file.read()

        image_pil = langsamutils.load_image(rgb_image_path)
        base64_image = load_image_as_base64(rgb_image_path)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": input_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                ],
            },
        ]

        response = client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=messages,
            temperature=0,
            max_tokens=713,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        output = response.choices[0].message.content
        logging.info(output)
        result = process_grasping_result(output)

        if result["selected_object"]:
            goal = result["selected_object"]
        else:
            fallback_object = select_fallback_object(result["objects"])
            goal = fallback_object["name"] if fallback_object else input_text

        masks, boxes, phrases, logits = ray.get(langsam_actor.predict.remote(image_pil, goal))
        if masks is None or masks.numel() == 0:
            masks, boxes, phrases, logits = ray.get(langsam_actor.predict.remote(image_pil, input_text))
            if masks is None or masks.numel() == 0:
                masks, boxes, phrases, logits = ray.get(langsam_actor.predict.remote(image_pil, "object"))

        boxes_list = boxes.cpu().numpy().tolist()
        cropping_box = create_cropping_box_from_boxes(boxes_list, (img_ori.shape[1], img_ori.shape[0]))

        visualize_cropping_box(img_ori, cropping_box)
        ray.get(langsam_actor.save.remote(masks, boxes, phrases, logits, image_pil))

        _, pcd = utils.get_and_process_data(cropping_box, img_ori, depth_ori)
        gg, chosen_xyz, chosen_rot, chosen_depth = _choose_grasp_with_graspnet(pcd)

        grippers = gg.to_open3d_geometry_list()
        if SHOW_OPEN3D and grippers:
            _save_open3d_vis([pcd, grippers[0]], "outputs/graspnet/vis_final_grasp.png")

        return jsonify(
            {
                "xyz": np.asarray(chosen_xyz).tolist(),
                "rot": np.asarray(chosen_rot).tolist(),
                "dep": float(chosen_depth),
            }
        )
    except Exception as exc:
        logging.exception("Error while handling GraspNet /grasp_pose request")
        return jsonify({"error": repr(exc)}), 500
    finally:
        if temp_context is not None:
            temp_context.cleanup()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "backend": "graspnet"})


app.view_functions["get_grasp_pose"] = get_grasp_pose_graspnet


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ThinkGrasp regular GraspNet upload Flask server", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=5000)
