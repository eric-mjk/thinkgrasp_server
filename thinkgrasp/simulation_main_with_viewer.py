import os
import sys

THINKGRASP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ThinkGrasp")
if THINKGRASP_DIR not in sys.path:
    sys.path.insert(0, THINKGRASP_DIR)
os.chdir(THINKGRASP_DIR)

import json
import logging
import random
import threading

import matplotlib
import numpy as np
import pybullet as p
import ray
import torch
import wandb
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

import simulation_main as sim_base
import utils
from constants import WORKSPACE_LIMITS
from environment_sim import Environment
from grasp_detetor import Graspnet
from langsam import langsamutils
from langsam.langsam_actor import LangSAM
from logger import Logger
from openai import OpenAI


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def build_messages(input_text, base64_image):
    return [
        {
            "role": "system",
            "content": (
                "Given a 224x224 input image and the provided instruction, perform the following steps:\n"
                "Target Object Selection:\n"
                "Identify the object in the image that best matches the instruction. If the target object is found, select it as the target object.\n"
                "If the target object is not visible, select the most cost-effective object or object part considering ease of grasping, importance, and safety.\n"
                "If the object has a handle or a part that is easier or safer to grasp, strongly prefer to select that part.\n"
                "Consider the geometric shape of the objects and the gripper's success rate when selecting the target object or object part.\n"
                "Output the name of the selected object or object part as [object:color and name] or [object part:color and name]..\n"
                "Round object means like ball. Cup is different from mug."
                "Cropping Box Calculation:\n"
                "Calculate a cropping box that includes the target object and all surrounding objects that might be relevant for grasping.\n"
                "Provide the coordinates of the cropping box in the format (top-left x, top-left y, bottom-right x, bottom-right y).\n"
                "Object Properties within Cropping Box:\n"
                "For each object within the cropping box, provide the following properties:\n"
                "Grasping Score: Evaluate the ease or difficulty of grasping the object on a scale from 0 to 100 (0 being extremely difficult, 100 being extremely easy).\n"
                "Material Composition: Evaluate the material composition of the object on a scale from 0 to 100 (0 being extremely weak, 100 being extremely strong).\n"
                "Surface Texture: Evaluate the texture of the object's surface on a scale from 0 to 100 (0 being extremely smooth, 100 being extremely rough).\n"
                "Stability Assessment: Assess the stability of the object on a scale from 0 to 100 (0 being extremely unstable, 100 being extremely stable).\n"
                "Centroid Coordinates: Provide the coordinates (x, y) of the object's center of mass across the entire image.\n"
                "Preferred Grasping Location: Divide the cropping box into a 3x3 grid and return a number from 1 to 9 indicating the preferred grasping location (1 for top-left, 9 for bottom-right).\n"
                "Additionally, consider the preferred grasping location that is most successful for the UR5 robotic arm and gripper.\n"
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
                "...\n"
                "Example Output:\n"
                "Selected Object/Object Part: [object:blue ball]\n"
                "Cropping Box Coordinates: (50, 50, 200, 200)\n"
                "Objects and Their Properties:\n"
                "Object: Blue Ball\n"
                "Grasping Score: 90\n"
                "Material Composition: 80\n"
                "Surface Texture: 20\n"
                "Stability Assessment: 95\n"
                "Centroid Coordinates: (125, 125)\n"
                "Preferred Grasping Location: 5\n"
                "Object: Yellow Bottle\n"
                "Grasping Score: 75\n"
                "Material Composition: 70\n"
                "Surface Texture: 30\n"
                "Stability Assessment: 80\n"
                "Centroid Coordinates: (100, 150)\n"
                "Preferred Grasping Location: 3\n"
                "Object: Black and Blue Scissors\n"
                "Grasping Score: 60\n"
                "Material Composition: 85\n"
                "Surface Texture: 40\n"
                "Stability Assessment: 70\n"
                "Centroid Coordinates: (175, 175)\n"
                "Preferred Grasping Location: 7"
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": input_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            ],
        },
    ]


def draw_gpt_overlay(ax, image, result):
    ax.imshow(image)
    img_h, img_w = image.shape[:2]

    cx0, cy0, cx1, cy1 = 0, 0, img_w, img_h
    if result.get("cropping_box"):
        cx0, cy0, cx1, cy1 = result["cropping_box"]
        rect = patches.Rectangle(
            (cx0, cy0),
            cx1 - cx0,
            cy1 - cy0,
            linewidth=2.5,
            edgecolor="yellow",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)

    colors = plt.cm.Set1(np.linspace(0, 1, max(len(result.get("objects", [])), 1)))
    label_x = cx1 + 12
    if label_x + 220 > img_w:
        label_x = cx0 - 12
        anchor_side = "right"
    else:
        anchor_side = "left"

    objects = result.get("objects", [])
    crop_height = max(cy1 - cy0, 1)
    slot_height = crop_height / max(len(objects), 1)
    selected_object = (result.get("selected_object") or "").lower()

    for i, (obj, color) in enumerate(zip(objects, colors)):
        centroid_x, centroid_y = obj["centroid_coordinates"]
        name = obj["name"]
        selected = name.lower() == selected_object
        is_part = result.get("is_part", False) and selected
        tag = ""
        if selected:
            tag = " [SELECTED - PART]" if is_part else " [SELECTED]"
        label = (
            f"{name}{tag}\n"
            f"Score {obj['grasping_score']}  Material {obj['material_composition']}\n"
            f"Texture {obj['surface_texture']}  Stability {obj['stability_assessment']}\n"
            f"Centroid ({centroid_x}, {centroid_y})  Zone {obj['preferred_grasping_location']}/9"
        )
        label_y = cy0 + slot_height * i + slot_height * 0.5
        ax.plot(
            centroid_x,
            centroid_y,
            marker="*" if selected else "o",
            markersize=14 if selected else 10,
            color=color,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=5,
        )
        ax.annotate(
            label,
            xy=(centroid_x, centroid_y),
            xytext=(label_x, label_y),
            fontsize=8,
            color="white",
            ha="left" if anchor_side == "left" else "right",
            va="center",
            bbox=dict(facecolor=color, alpha=0.85, boxstyle="round,pad=0.4"),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            zorder=6,
        )

    title = f"GPT Selection: {result.get('selected_object', 'N/A') or 'N/A'}"
    if result.get("is_part", False):
        title += " (part)"
    ax.set_title(title)
    ax.axis("off")


def draw_detection_overlay(ax, image, boxes_list, cropping_box, goal):
    ax.imshow(image)
    for box in boxes_list:
        bx1, by1, bx2, by2 = box
        ax.add_patch(
            patches.Rectangle(
                (bx1, by1),
                bx2 - bx1,
                by2 - by1,
                edgecolor="lime",
                facecolor="none",
                linewidth=2,
            )
        )
    x1, y1, x2, y2 = cropping_box
    ax.add_patch(
        patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            edgecolor="red",
            facecolor="none",
            linewidth=2,
            linestyle="--",
        )
    )
    ax.set_title(f'LangSAM boxes and crop: "{goal}"')
    ax.axis("off")


def workspace_to_pixel(wx, wy, image_width, image_height, workspace_limits):
    px = int((wy - workspace_limits[1][0]) / (workspace_limits[1][1] - workspace_limits[1][0]) * image_width)
    py = int((wx - workspace_limits[0][0]) / (workspace_limits[0][1] - workspace_limits[0][0]) * image_height)
    return px, py


def draw_preferred_grasp_cell(ax, box, preferred_grasping_location):
    if preferred_grasping_location < 1 or preferred_grasping_location > 9:
        preferred_grasping_location = 5

    x0, y0, x1, y1 = [int(v) for v in box]
    width = x1 - x0
    height = y1 - y0
    grid_size = 3
    cell_width = width // grid_size
    cell_height = height // grid_size
    row = (preferred_grasping_location - 1) // grid_size
    col = (preferred_grasping_location - 1) % grid_size

    for i in range(1, grid_size):
        ax.plot([x0 + i * cell_width, x0 + i * cell_width], [y0, y1], color="cyan", linewidth=1.0, alpha=0.9)
        ax.plot([x0, x1], [y0 + i * cell_height, y0 + i * cell_height], color="cyan", linewidth=1.0, alpha=0.9)

    cell_x0 = x0 + col * cell_width
    cell_y0 = y0 + row * cell_height
    cell_x1 = cell_x0 + cell_width
    cell_y1 = cell_y0 + cell_height
    ax.add_patch(
        patches.Rectangle(
            (cell_x0, cell_y0),
            cell_x1 - cell_x0,
            cell_y1 - cell_y0,
            linewidth=2.5,
            edgecolor="deepskyblue",
            facecolor="deepskyblue",
            alpha=0.28,
        )
    )
    ax.text(
        cell_x0,
        max(cell_y0 - 4, 0),
        f"selected cell {preferred_grasping_location}",
        color="deepskyblue",
        fontsize=9,
        bbox=dict(facecolor="black", alpha=0.55, pad=1),
    )


def draw_topdown_grasps(
    ax,
    color_image,
    boxes_list,
    pos_bboxes,
    grasp_pose_set,
    action_idx,
    preferred_grasping_location,
):
    img_h, img_w = color_image.shape[:2]
    ax.imshow(color_image)

    if not boxes_list:
        ax.set_title("Action-selection view: no LangSAM box")
        ax.axis("off")
        return

    active_box = boxes_list[0]
    bx1, by1, bx2, by2 = active_box
    ax.add_patch(
        patches.Rectangle(
            (bx1, by1),
            bx2 - bx1,
            by2 - by1,
            linewidth=3,
            edgecolor="gold",
            facecolor="gold",
            alpha=0.12,
        )
    )
    ax.text(
        bx1,
        max(by1 - 18, 0),
        "Active LangSAM detection box",
        color="gold",
        fontsize=9,
        bbox=dict(facecolor="black", alpha=0.6, pad=1),
    )
    draw_preferred_grasp_cell(ax, active_box, preferred_grasping_location)

    if pos_bboxes is not None:
        pb = pos_bboxes[0, 0].detach().cpu().numpy()
        tb_px, tb_py = workspace_to_pixel(pb[0], pb[1], img_w, img_h, WORKSPACE_LIMITS)
        ax.plot(
            tb_px,
            tb_py,
            "*",
            markersize=18,
            color="gold",
            markeredgecolor="black",
            markeredgewidth=1,
            zorder=6,
        )
        ax.text(
            tb_px + 4,
            tb_py + 4,
            "selected cell center",
            color="gold",
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.6, pad=1),
        )

    if grasp_pose_set and 0 <= action_idx < len(grasp_pose_set):
        tx, ty, _ = grasp_pose_set[action_idx][:3]
        px, py = workspace_to_pixel(tx, ty, img_w, img_h, WORKSPACE_LIMITS)
        ax.plot(
            px,
            py,
            marker="*",
            markersize=16,
            color="magenta",
            markeredgecolor="white",
            markeredgewidth=0.9,
            zorder=7,
        )
        ax.text(
            px + 4,
            py - 8,
            "selected grasp",
            color="magenta",
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.6, pad=1),
        )

    ax.text(
        0.02,
        0.02,
        "Gold box: active LangSAM detection\n"
        "Cyan grid: 3x3 preferred-grasp grid\n"
        "Blue cell: GPT-selected grasp area\n"
        "Gold star: center of selected cell\n"
        "Magenta star: selected grasp pose",
        transform=ax.transAxes,
        color="white",
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="black", alpha=0.7, pad=4),
    )
    ax.set_title(f"Action-selection view: selected grasp #{action_idx}")
    ax.axis("off")


def create_bbox_gallery(remain_bbox_images, action_idx, thumb_size=96, max_items=6):
    if not remain_bbox_images:
        return np.full((thumb_size, thumb_size * 2, 3), 245, dtype=np.uint8)

    items = remain_bbox_images[:max_items]
    tiles = []
    for i, img in enumerate(items):
        pil_img = Image.fromarray(img).convert("RGB").resize((thumb_size, thumb_size))
        tile = np.array(pil_img)
        if i == action_idx:
            border = 6
            highlighted = np.full((thumb_size + border * 2, thumb_size + border * 2, 3), 215, dtype=np.uint8)
            highlighted[:] = np.array([255, 215, 0], dtype=np.uint8)
            highlighted[border:-border, border:-border] = tile
            tile = highlighted
        else:
            pad = 6
            padded = np.full((thumb_size + pad * 2, thumb_size + pad * 2, 3), 230, dtype=np.uint8)
            padded[pad:-pad, pad:-pad] = tile
            tile = padded
        tiles.append(tile)
    return np.concatenate(tiles, axis=1)


def set_equal_3d_axes(ax, xs, ys, zs):
    if len(xs) == 0:
        return
    x_mid = (xs.max() + xs.min()) * 0.5
    y_mid = (ys.max() + ys.min()) * 0.5
    z_mid = (zs.max() + zs.min()) * 0.5
    max_range = max(xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min(), 1e-3) * 0.5
    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)


def draw_3d_scene(ax, cropped_pcd, remain_gg, action_idx):
    pts = np.asarray(cropped_pcd.points)
    clrs = np.asarray(cropped_pcd.colors)
    if len(pts) == 0:
        ax.set_title("3D scene unavailable")
        return

    stride = max(len(pts) // 5000, 1)
    pts_sample = pts[::stride]
    clrs_sample = clrs[::stride] if len(clrs) else None
    ax.scatter(
        pts_sample[:, 0],
        pts_sample[:, 1],
        pts_sample[:, 2],
        c=clrs_sample,
        s=1.5,
        depthshade=False,
    )

    for i, geom in enumerate(remain_gg[: min(len(remain_gg), 30)]):
        verts = np.asarray(geom.vertices)
        tris = np.asarray(geom.triangles)
        if len(verts) == 0 or len(tris) == 0:
            continue
        face_vertices = verts[tris]
        mesh = Poly3DCollection(face_vertices, linewidths=0.1)
        if i == action_idx:
            mesh.set_facecolor((1.0, 0.84, 0.0, 0.95))
            mesh.set_edgecolor((0.5, 0.35, 0.0, 0.6))
        else:
            mesh.set_facecolor((1.0, 0.1, 0.1, 0.22))
            mesh.set_edgecolor((0.7, 0.1, 0.1, 0.15))
        ax.add_collection3d(mesh)

    set_equal_3d_axes(ax, pts[:, 0], pts[:, 1], pts[:, 2])
    ax.view_init(elev=25, azim=-60)
    ax.set_title(f"3D grasps: selected #{action_idx}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def show_pre_action_review(
    lang_goal,
    result,
    goal,
    color_image,
    depth_image,
    mask_image,
    boxes_list,
    cropping_box,
    remain_bbox_images,
    pos_bboxes,
    grasp_pose_set,
    cropped_pcd,
    remain_gg,
    action_idx,
    preferred_grasping_location,
):
    continue_event = threading.Event()

    fig = plt.figure(figsize=(18, 10))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05])
    fig.suptitle(f'Pre-action review | instruction: "{lang_goal}" | target: "{goal}"', fontsize=14)

    ax_color = fig.add_subplot(grid[0, 0])
    ax_depth = fig.add_subplot(grid[0, 1])
    ax_mask = fig.add_subplot(grid[0, 2])
    ax_gallery = fig.add_subplot(grid[0, 3])
    ax_gpt = fig.add_subplot(grid[1, 0])
    ax_detect = fig.add_subplot(grid[1, 1])
    ax_top = fig.add_subplot(grid[1, 2])
    ax_3d = fig.add_subplot(grid[1, 3], projection="3d")

    ax_color.imshow(color_image)
    ax_color.set_title("Color")
    ax_color.axis("off")

    ax_depth.imshow(depth_image, cmap="gray")
    ax_depth.set_title("Depth")
    ax_depth.axis("off")

    ax_mask.imshow(mask_image, cmap="nipy_spectral")
    ax_mask.set_title("Mask")
    ax_mask.axis("off")

    gallery = create_bbox_gallery(remain_bbox_images, action_idx)
    ax_gallery.imshow(gallery)
    ax_gallery.set_title("Candidate crops (selected in gold)")
    ax_gallery.axis("off")

    draw_gpt_overlay(ax_gpt, color_image, result)
    draw_detection_overlay(ax_detect, color_image, boxes_list, cropping_box, goal)
    draw_topdown_grasps(
        ax_top,
        color_image,
        boxes_list,
        pos_bboxes,
        grasp_pose_set,
        action_idx,
        preferred_grasping_location,
    )
    draw_3d_scene(ax_3d, cropped_pcd, remain_gg, action_idx)

    button_ax = fig.add_axes([0.45, 0.02, 0.12, 0.05])
    button = Button(button_ax, "Continue", color="#e6e6e6", hovercolor="#d0f0c0")

    def on_continue(_event):
        continue_event.set()
        plt.close(fig)

    def on_close(_event):
        continue_event.set()

    button.on_clicked(on_continue)
    fig.canvas.mpl_connect("close_event", on_close)
    plt.tight_layout(rect=[0, 0.07, 1, 0.96])
    plt.show(block=True)
    continue_event.wait()


def main():
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError("OPENAI_API_KEY must be set before running this script.")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    wandb.init(project="robotic-grasping1.0")
    ray.init(num_gpus=1)
    use_gpu = torch.cuda.is_available()
    gpu_allocation = 1 if use_gpu else 0
    actor_options = {"num_gpus": gpu_allocation}
    langsam_actor = LangSAM.options(**actor_options).remote(use_gpu=use_gpu)

    args = sim_base.parse_args()

    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    num_episode = args.num_episode
    env = Environment(gui=args.gui)
    env.seed(args.seed)
    logger = Logger(case_dir=args.testing_case_dir)
    graspnet = Graspnet()

    if os.path.exists(args.testing_case_dir):
        filelist = os.listdir(args.testing_case_dir)
        filelist.sort(key=lambda x: int(x[4:6]))
    else:
        filelist = []

    if args.testing_case is not None:
        filelist = [args.testing_case]

    case = 0
    iteration = 0

    try:
        for f in filelist:
            #####
            if case < 1:
                case += 1
                continue
            #####
            f = os.path.join(args.testing_case_dir, f)

            logger.episode_reward_logs = []
            logger.episode_step_logs = []
            logger.episode_success_logs = []

            for episode in range(num_episode):
                episode_reward = 0
                episode_steps = 0
                done = False
                reset = False

                while not reset:
                    env.reset()
                    reset, lang_goal = env.add_object_push_from_file(f)
                    print(f"\033[032m Reset environment of episode {episode}, language goal {lang_goal}\033[0m")

                while not done:
                    out_of_workspace = []
                    for obj_id in env.target_obj_ids:
                        pos, _, _ = env.obj_info(obj_id)
                        if (
                            pos[0] < WORKSPACE_LIMITS[0][0]
                            or pos[0] > WORKSPACE_LIMITS[0][1]
                            or pos[1] < WORKSPACE_LIMITS[1][0]
                            or pos[1] > WORKSPACE_LIMITS[1][1]
                        ):
                            out_of_workspace.append(obj_id)
                    if len(out_of_workspace) == len(env.target_obj_ids):
                        print("\033[031m Target objects are not in the scene!\033[0m")
                        break

                    color_image, depth_image, mask_image = utils.get_true_heightmap(env)
                    image = "color_map.png"
                    image_pil = langsamutils.load_image(image)
                    base64_image = sim_base.load_image_as_base64(image)

                    messages = build_messages(lang_goal, base64_image)
                    result = {
                        "selected_object": None,
                        "cropping_box": None,
                        "objects": [],
                        "is_part": False,
                    }

                    try:
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
                        result = sim_base.process_grasping_result(output)
                        sim_base.print_grasping_result(result)
                        with open("grasping_result_log.json", "w") as log_file:
                            json.dump(result, log_file, indent=4)
                        wandb.log({"gpt4o_output": output})
                    except Exception as e:
                        logging.error(f"Error with OpenAI API request: {e}")

                    if not result["selected_object"]:
                        fallback_object = sim_base.select_fallback_object(result["objects"])
                        goal = fallback_object["name"] if fallback_object else lang_goal
                    else:
                        goal = result["selected_object"]

                    preferred_grasping_location = 5
                    for obj in result["objects"]:
                        if obj["name"] == goal:
                            preferred_grasping_location = obj.get("preferred_grasping_location", 5)

                    masks, boxes, phrases, logits = ray.get(langsam_actor.predict.remote(image_pil, goal))
                    if masks is None or masks.numel() == 0:
                        masks, boxes, phrases, logits = ray.get(langsam_actor.predict.remote(image_pil, lang_goal))
                        if masks is None or masks.numel() == 0:
                            masks, boxes, phrases, logits = ray.get(langsam_actor.predict.remote(image_pil, "object"))

                    boxes_list = boxes.cpu().numpy().tolist()
                    cropping_box = sim_base.create_cropping_box_from_boxes(
                        boxes_list, (color_image.shape[1], color_image.shape[0])
                    )
                    ray.get(langsam_actor.save.remote(masks, boxes, phrases, logits, image_pil, gui=args.gui))
                    bbox_images, bbox_positions = utils.convert_output(
                        image_pil,
                        boxes,
                        phrases,
                        logits,
                        color_image,
                        depth_image,
                        mask_image,
                        preferred_grasping_location,
                    )

                    pcd = utils.get_fuse_pointcloud(env)
                    cropped_pcd = sim_base.crop_pointcloud(
                        pcd, cropping_box, color_image, depth_image, WORKSPACE_LIMITS
                    )
                    if cropped_pcd is None or len(np.asarray(cropped_pcd.points)) == 0:
                        print("\033[031m Cropped point cloud is empty!\033[0m")
                        continue

                    with torch.no_grad():
                        grasp_pose_set, _, remain_gg = graspnet.grasp_detection(
                            cropped_pcd, env.get_true_object_poses()
                        )
                    print("Number of grasping poses", len(grasp_pose_set))
                    logging.info(f"Number of grasping poses: {len(grasp_pose_set)}")

                    if len(grasp_pose_set) == 0:
                        with torch.no_grad():
                            grasp_pose_set, _, remain_gg = graspnet.grasp_detection(
                                pcd, env.get_true_object_poses()
                            )
                        print("Number of grasping poses", len(grasp_pose_set))
                        logging.info(f"Number of grasping poses: {len(grasp_pose_set)}")
                        if len(grasp_pose_set) == 0:
                            break

                    remain_bbox_images, bboxes, pos_bboxes, grasps = utils.preprocess(
                        bbox_images, bbox_positions, grasp_pose_set, (args.patch_size, args.patch_size)
                    )
                    logger.save_bbox_images(iteration, remain_bbox_images)
                    logger.save_heightmaps(iteration, color_image, depth_image)
                    if bboxes is None:
                        break

                    if len(grasp_pose_set) == 1:
                        action_idx = 0
                    else:
                        with torch.no_grad():
                            action_idx = sim_base.select_action(bboxes, pos_bboxes, lang_goal, grasps)
                    action = grasp_pose_set[action_idx]

                    show_pre_action_review(
                        lang_goal=lang_goal,
                        result=result,
                        goal=goal,
                        color_image=color_image,
                        depth_image=depth_image,
                        mask_image=mask_image,
                        boxes_list=boxes_list,
                        cropping_box=cropping_box,
                        remain_bbox_images=remain_bbox_images,
                        pos_bboxes=pos_bboxes,
                        grasp_pose_set=grasp_pose_set,
                        cropped_pcd=cropped_pcd,
                        remain_gg=remain_gg,
                        action_idx=action_idx,
                        preferred_grasping_location=preferred_grasping_location,
                    )

                    reward, done = env.step(action)
                    iteration += 1
                    episode_steps += 1
                    episode_reward += reward
                    print(
                        "\033[034m Episode: {}, step: {}, reward: {}\033[0m".format(
                            episode, episode_steps, round(reward, 2)
                        )
                    )
                    wandb.log({"episode": episode, "step": episode_steps, "reward": reward})

                    if episode_steps == args.max_episode_step:
                        break

                logger.episode_reward_logs.append(episode_reward)
                logger.episode_step_logs.append(episode_steps)
                logger.episode_success_logs.append(done)
                logger.write_to_log("episode_reward", logger.episode_reward_logs)
                logger.write_to_log("episode_step", logger.episode_step_logs)
                logger.write_to_log("episode_success", logger.episode_success_logs)
                print(
                    "\033[034m Episode: {}, episode steps: {}, episode reward: {}, success: {}\033[0m".format(
                        episode, episode_steps, round(episode_reward, 2), done
                    )
                )

                if episode == num_episode - 1:
                    avg_success = sum(logger.episode_success_logs) / len(logger.episode_success_logs)
                    avg_reward = sum(logger.episode_reward_logs) / len(logger.episode_reward_logs)
                    avg_step = sum(logger.episode_step_logs) / len(logger.episode_step_logs)

                    success_steps = []
                    for i in range(len(logger.episode_success_logs)):
                        if logger.episode_success_logs[i]:
                            success_steps.append(logger.episode_step_logs[i])
                    avg_success_step = sum(success_steps) / len(success_steps) if success_steps else 1000

                    result_file = os.path.join(logger.result_directory, "case" + str(case) + ".txt")
                    with open(result_file, "w") as out_file:
                        out_file.write(
                            "%s %.18e %.18e %.18e %.18e\n"
                            % (
                                lang_goal,
                                avg_success,
                                avg_step,
                                avg_success_step,
                                avg_reward,
                            )
                        )
                    case += 1
                    print(
                        "\033[034m Language goal: {}, average steps: {}/{}, average reward: {}, average success: {}\033[0m".format(
                            lang_goal, avg_step, avg_success_step, avg_reward, avg_success
                        )
                    )
                    logging.info(
                        f"Language goal: {lang_goal}, average steps: {avg_step}/{avg_success_step}, average reward: {avg_reward}, average success: {avg_success}"
                    )
                    wandb.log(
                        {
                            "lang_goal": lang_goal,
                            "avg_success": avg_success,
                            "avg_step": avg_step,
                            "avg_success_step": avg_success_step,
                            "avg_reward": avg_reward,
                        }
                    )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
