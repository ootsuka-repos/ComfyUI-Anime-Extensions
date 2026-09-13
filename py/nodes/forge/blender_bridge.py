"""Headless Blender bridge for AvatarStage.

This file intentionally uses only Blender's bundled Python APIs.  It is run by
``blender --background --python blender_bridge.py -- ...`` and is kept outside
the web process so a faulty third-party VRM cannot execute in the API worker.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import struct
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

_VRM_ADDON = "VRM_Addon_for_Blender-release"
_GLTF_COMPONENT_FORMAT = {5126: "f", 5125: "I", 5123: "H", 5122: "h", 5121: "B", 5120: "b"}
_GLTF_COMPONENT_SIZE = {5126: 4, 5125: 4, 5123: 2, 5122: 2, 5121: 1, 5120: 1}
_GLTF_TYPE_SIZE = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
_FINGER_HUMANOID_KEYS = (
    "left_thumb_metacarpal", "left_thumb_proximal", "left_thumb_distal",
    "right_thumb_metacarpal", "right_thumb_proximal", "right_thumb_distal",
    *(f"{side}_{finger}_{part}"
      for side in ("left", "right")
      for finger in ("index", "middle", "ring", "little")
      for part in ("proximal", "intermediate", "distal")),
)


def _args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("starter", "retarget"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--vrm-output", required=True)
    parser.add_argument("--poster-output", required=True)
    parser.add_argument("--blend-output", required=True)
    parser.add_argument("--input-vrm")
    parser.add_argument("--motion-glb")
    parser.add_argument("--motion-2d")
    parser.add_argument("--vrma-output")
    parser.add_argument(
        "--skip-vrma",
        action="store_true",
        help="Render a retargeted video without exporting a portable VRMA file.",
    )
    parser.add_argument("--video-output")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument(
        "--frame-end", type=int,
        help="Optional inclusive final frame for a short retarget validation render.",
    )
    parser.add_argument(
        "--frame-start", type=int,
        help="Optional initial frame for a short retarget validation render.",
    )
    return parser.parse_args(argv)


def _enable_vrm_addon() -> None:
    """Enable the installed official add-on without relying on UI state."""

    # bpy.ops synthesizes attributes even for unregistered operators, so hasattr
    # cannot tell whether the addon is enabled in a fresh background process.
    bpy.utils.refresh_script_paths()
    importlib.import_module(_VRM_ADDON)
    if _VRM_ADDON not in bpy.context.preferences.addons:
        result = bpy.ops.preferences.addon_enable(module=_VRM_ADDON)
        if result != {"FINISHED"}:
            raise RuntimeError(f"could not enable VRM Add-on: {result}")
    try:
        bpy.ops.export_scene.vrm.get_rna_type()
        bpy.ops.icyp.make_basic_armature.get_rna_type()
    except RuntimeError as exc:
        raise RuntimeError("VRM Add-on did not register its operators") from exc


def _clear_scene() -> None:
    for obj in tuple(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in (bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for item in tuple(collection):
            collection.remove(item)


def _material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Roughness"].default_value = 0.72
    return material


def _parent_to_bone(obj: bpy.types.Object, armature: bpy.types.Object, bone_name: str) -> None:
    """Parent while preserving the object's current world transform."""

    world = obj.matrix_world.copy()
    obj.parent = armature
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    obj.matrix_world = world


def _add_sphere(
    armature: bpy.types.Object,
    bone_name: str,
    location: Vector,
    scale: tuple[float, float, float],
    material: bpy.types.Material,
    *, align_to_bone: bool = False,
) -> None:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, location=location)
    obj = bpy.context.object
    obj.scale = scale
    if align_to_bone:
        bone = armature.data.bones[bone_name]
        direction = armature.matrix_world.to_3x3() @ (bone.tail_local - bone.head_local)
        obj.location = armature.matrix_world @ ((bone.head_local + bone.tail_local) * 0.5)
        obj.scale.z = direction.length * 0.55
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(direction.normalized())
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    _parent_to_bone(obj, armature, bone_name)


def _bone_position(armature: bpy.types.Object, bone_name: str, *, tail: bool = False) -> Vector:
    bone = armature.data.bones.get(bone_name)
    if bone is None:
        raise RuntimeError(f"starter armature is missing bone: {bone_name}")
    local = bone.tail_local if tail else bone.head_local
    return armature.matrix_world @ local


def _set_metadata(armature: bpy.types.Object, name: str) -> None:
    armature.data.vrm_addon_extension.spec_version = "1.0"
    meta = armature.data.vrm_addon_extension.vrm1.meta
    meta.vrm_name = name
    meta.version = "1.0.0"
    meta.authors.add().value = "Doujin Forge AvatarStage"
    meta.avatar_permission = "onlyAuthor"
    meta.allow_excessively_violent_usage = False
    meta.allow_excessively_sexual_usage = False
    meta.commercial_usage = "personalNonProfit"
    meta.allow_political_or_religious_usage = False
    meta.allow_antisocial_or_hate_usage = False
    meta.credit_notation = "required"
    meta.allow_redistribution = False
    meta.modification = "prohibited"


def _make_starter(name: str) -> bpy.types.Object:
    """Create a clearly-labelled technical VRM mannequin, not a claimed 3D clone."""

    _clear_scene()
    bpy.ops.icyp.make_basic_armature()
    armature = bpy.context.active_object
    if armature is None or armature.type != "ARMATURE":
        raise RuntimeError("VRM Add-on could not create a humanoid armature")
    _set_metadata(armature, name)

    skin = _material("AvatarStage Skin", (0.93, 0.55, 0.46, 1.0))
    hair = _material("AvatarStage Hair", (0.18, 0.10, 0.30, 1.0))
    outfit = _material("AvatarStage Outfit", (0.24, 0.38, 0.90, 1.0))
    accent = _material("AvatarStage Accent", (0.96, 0.34, 0.58, 1.0))

    _add_sphere(armature, "head", _bone_position(armature, "head", tail=True), (0.20, 0.20, 0.24), skin)
    _add_sphere(armature, "head", _bone_position(armature, "head", tail=True) + Vector((0, 0.02, 0.13)), (0.22, 0.22, 0.12), hair)
    _add_sphere(armature, "chest", _bone_position(armature, "chest"), (0.28, 0.18, 0.36), outfit)
    _add_sphere(armature, "hips", _bone_position(armature, "hips"), (0.25, 0.17, 0.18), accent)
    for bone_name, scale, material in (
        ("upper_arm.L", (0.08, 0.09, 0.22), outfit), ("lower_arm.L", (0.07, 0.08, 0.20), skin),
        ("upper_arm.R", (0.08, 0.09, 0.22), outfit), ("lower_arm.R", (0.07, 0.08, 0.20), skin),
        ("upper_leg.L", (0.12, 0.12, 0.28), outfit), ("lower_leg.L", (0.10, 0.10, 0.27), skin),
        ("upper_leg.R", (0.12, 0.12, 0.28), outfit), ("lower_leg.R", (0.10, 0.10, 0.27), skin),
        ("hand.L", (0.09, 0.07, 0.09), skin), ("hand.R", (0.09, 0.07, 0.09), skin),
        ("foot.L", (0.11, 0.20, 0.08), accent), ("foot.R", (0.11, 0.20, 0.08), accent),
    ):
        _add_sphere(armature, bone_name, _bone_position(armature, bone_name, tail=True), scale, material,
                    align_to_bone="arm." in bone_name or "leg." in bone_name)
    return armature


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def _setup_stage(armature: bpy.types.Object, width: int, height: int) -> None:
    scene = bpy.context.scene
    # Blender 4.0 exposes the Eevee Next renderer as ``BLENDER_EEVEE``;
    # later versions also accept the same stable enum name.
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.color = (0.035, 0.045, 0.09)

    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = "VSB_Floor"
    floor.data.materials.append(_material("AvatarStage Floor", (0.065, 0.08, 0.15, 1.0)))

    # Frame the imported avatar rather than the small starter mannequin.  The
    # earlier fixed camera distance made real VRM avatars occupy only a small
    # part of a portrait render.
    subject_bounds = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.name.startswith("VSB_"):
            continue
        subject_bounds.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if subject_bounds:
        low = Vector(map(min, zip(*subject_bounds, strict=True)))
        high = Vector(map(max, zip(*subject_bounds, strict=True)))
        focus = (low + high) * 0.5
        avatar_height = max(1.0, high.z - low.z)
    else:
        head = armature.data.bones.get("head")
        focus = armature.matrix_world @ (head.head_local if head else Vector((0, 0, 1.25)))
        avatar_height = 1.7
    # Leave headroom for raised arms and fast dance poses in a portrait frame.
    camera_distance = max(3.1, avatar_height * 2.15)
    bpy.ops.object.camera_add(
        location=focus + Vector((avatar_height * 0.20, -camera_distance, avatar_height * 0.16))
    )
    camera = bpy.context.object
    camera.data.lens = 44
    _look_at(camera, focus)
    scene.camera = camera
    for location, energy, size in (((4.0, -3.0, 6.0), 1200, 4.0), ((-3.0, -1.0, 3.5), 850, 3.0)):
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.data.energy = energy
        light.data.shape = "DISK"
        light.data.size = size
        _look_at(light, focus)


def _write_starter(args: argparse.Namespace) -> None:
    armature = _make_starter(args.name)
    _setup_stage(armature, args.width, args.height)
    bpy.context.scene.render.filepath = args.poster_output
    bpy.ops.render.render(write_still=True)
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    for obj in bpy.context.scene.objects:
        if obj.parent == armature:
            obj.select_set(True)
    bpy.context.view_layer.objects.active = armature
    result = bpy.ops.export_scene.vrm(filepath=args.vrm_output, export_only_selections=True)
    if result != {"FINISHED"}:
        raise RuntimeError(f"VRM export failed: {result}")
    bpy.ops.wm.save_as_mainfile(filepath=args.blend_output)


def _read_glb_positions(path: Path) -> dict[str, list[tuple[float, float, float]]]:
    """Read named translation tracks from ComfyUI's portable OpenPose GLB."""

    payload = path.read_bytes()
    if len(payload) < 20 or payload[:4] != b"glTF":
        raise RuntimeError("motion file is not a GLB")
    _magic, _version, total = struct.unpack_from("<4sII", payload, 0)
    if total != len(payload):
        raise RuntimeError("motion GLB has an invalid length")
    offset = 12
    document = None
    binary = None
    while offset + 8 <= len(payload):
        length, kind = struct.unpack_from("<I4s", payload, offset)
        offset += 8
        chunk = payload[offset : offset + length]
        offset += length
        if kind == b"JSON":
            document = json.loads(chunk.decode("utf-8"))
        elif kind == b"BIN\x00":
            binary = chunk
    if not isinstance(document, dict) or binary is None:
        raise RuntimeError("motion GLB has no JSON/BIN chunks")

    accessors = document.get("accessors", [])
    views = document.get("bufferViews", [])

    def accessor(index: int) -> list[tuple[float, ...]]:
        spec = accessors[index]
        view = views[spec["bufferView"]]
        component_type = int(spec["componentType"])
        scalar = _GLTF_COMPONENT_FORMAT.get(component_type)
        count_per = _GLTF_TYPE_SIZE.get(str(spec["type"]))
        if scalar is None or count_per is None:
            raise RuntimeError("unsupported GLB accessor")
        component_size = _GLTF_COMPONENT_SIZE[component_type]
        stride = int(view.get("byteStride", component_size * count_per))
        start = int(view.get("byteOffset", 0)) + int(spec.get("byteOffset", 0))
        fmt = "<" + scalar * count_per
        return [struct.unpack_from(fmt, binary, start + stride * item) for item in range(int(spec["count"]))]

    names = {index: str(node.get("name", "")) for index, node in enumerate(document.get("nodes", []))}
    tracks: dict[str, list[tuple[float, float, float]]] = {}
    for animation in document.get("animations", []):
        samplers = animation.get("samplers", [])
        for channel in animation.get("channels", []):
            target = channel.get("target", {})
            node = target.get("node")
            if target.get("path") != "translation" or not isinstance(node, int):
                continue
            name = names.get(node, "")
            sampler = samplers[int(channel["sampler"])]
            values = accessor(int(sampler["output"]))
            if values and len(values[0]) == 3:
                tracks[name] = [(float(x), float(y), float(z)) for x, y, z in values]
    if not tracks:
        raise RuntimeError("motion GLB has no landmark translation tracks")
    return tracks


def _landmark_tracks(raw: dict[str, list[tuple[float, float, float]]]) -> dict[str, list[Vector]]:
    """Convert glTF Y-up landmarks to Blender's Z-up coordinates."""

    result: dict[str, list[Vector]] = {}
    for name, frames in raw.items():
        converted = [Vector((x, -z, y)) for x, y, z in frames]
        if name.startswith("openpose_") and not name.startswith("openpose_face_"):
            result[name.removeprefix("openpose_")] = converted
        # ``trackNN`` is SAM 3D Body's person root.  With centered camera
        # translation enabled it carries the per-frame screen movement which
        # is deliberately absent from the individual, body-relative joints.
        elif name.startswith("track"):
            result["Root"] = converted
    if not result:
        raise RuntimeError("motion GLB has no OpenPose landmarks")
    count = min(len(frames) for frames in result.values())
    if count < 2:
        raise RuntimeError("motion GLB has fewer than two frames")
    for name, frames in tuple(result.items()):
        result[name] = frames[:count]
    if "RHip" in result and "LHip" in result:
        result["MidHip"] = [
            (right + left) * 0.5
            for right, left in zip(result["RHip"], result["LHip"], strict=True)
        ]
    return result


def _read_2d_tracks(path: Path) -> dict[str, list[Vector]]:
    """Read RTMW's source-image landmarks into Blender's screen plane."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("2D motion file is not valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("kind") != "rtmw_2d_openpose_body":
        raise RuntimeError("2D motion file has an unsupported schema")
    raw_points = payload.get("points")
    if not isinstance(raw_points, dict):
        raise RuntimeError("2D motion file has no landmark points")
    tracks: dict[str, list[Vector]] = {}
    for name, frames in raw_points.items():
        if not isinstance(name, str) or not isinstance(frames, list):
            continue
        converted: list[Vector] = []
        for value in frames:
            if not isinstance(value, list) or len(value) != 2:
                raise RuntimeError(f"2D motion landmark {name} has an invalid frame")
            x, y = value
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                raise RuntimeError(f"2D motion landmark {name} has non-numeric coordinates")
            # Image Y grows downwards; Blender's stage Z grows upwards.
            converted.append(Vector((float(x), 0.0, -float(y))))
        if converted:
            tracks[name] = converted
    if not tracks:
        raise RuntimeError("2D motion file contains no usable landmarks")
    count = min(len(frames) for frames in tracks.values())
    if count < 2:
        raise RuntimeError("2D motion file has fewer than two frames")
    for name, frames in tuple(tracks.items()):
        tracks[name] = frames[:count]
    return tracks


def _find_armature() -> bpy.types.Object:
    candidates = [item for item in bpy.context.scene.objects if item.type == "ARMATURE"]
    if not candidates:
        raise RuntimeError("imported VRM contains no armature")
    candidates.sort(key=lambda item: len(item.data.bones), reverse=True)
    return candidates[0]


def _humanoid_bone(armature: bpy.types.Object, key: str) -> str | None:
    try:
        name = getattr(armature.data.vrm_addon_extension.vrm1.humanoid.human_bones, key).node.bone_name
    except AttributeError:
        name = ""
    if name and armature.pose.bones.get(name):
        return name
    return None


def _safe_direction(start: Vector, end: Vector, fallback: Vector) -> Vector:
    direction = end - start
    return direction.normalized() if direction.length > 1e-6 else fallback.normalized()


def _animate_retarget(armature: bpy.types.Object, landmarks: dict[str, list[Vector]]) -> int:
    """Retarget a frontal single-person OpenPose track onto a VRM rig.

    The target rig's own rest matrices preserve its bone rolls and hierarchy.
    For this product's frontal dance inputs, the reliable screen-plane joint
    directions are used while SAM3D's unstable camera-depth estimates are
    intentionally excluded from the final character pose.
    """

    required = ("Neck", "RHip", "LHip", "RShoulder", "LShoulder", "RElbow", "LElbow", "RWrist", "LWrist", "RKnee", "LKnee", "RAnkle", "LAnkle")
    missing = [name for name in required if name not in landmarks]
    if missing:
        raise RuntimeError(f"motion GLB is missing required landmarks: {', '.join(missing)}")
    count = min(len(track) for track in landmarks.values())
    names = {key: _humanoid_bone(armature, key) for key in (
        "hips", "spine", "chest", "upper_chest", "neck", "head",
        "left_upper_arm", "left_lower_arm", "left_hand",
        "right_upper_arm", "right_lower_arm", "right_hand",
        "left_upper_leg", "left_lower_leg", "left_foot",
        "right_upper_leg", "right_lower_leg", "right_foot",
        *_FINGER_HUMANOID_KEYS,
    )}
    pose = armature.pose.bones
    rest = {key: armature.data.bones[name].matrix_local.copy() for key, name in names.items() if name}
    # The rest local +Y axis is Blender's bone length axis.  Calibrating each
    # segment against it avoids assuming an arbitrary VRM has TRACK_Y roll.
    rest_axis = {key: (matrix.to_3x3() @ Vector((0.0, 1.0, 0.0))).normalized() for key, matrix in rest.items()}
    initial_midhip = (landmarks["RHip"][0] + landmarks["LHip"][0]) * 0.5
    initial_root = landmarks.get("Root", [initial_midhip])[0]
    target_height = max((landmarks["Neck"][0] - initial_midhip).length, 1e-5)
    model_height = max((armature.data.bones[names["head"]].head_local - armature.data.bones[names["hips"]].head_local).length if names.get("head") and names.get("hips") else 1.0, 1e-5)
    # Root depth is perceived against the whole avatar, not only its short
    # head-to-hips segment. Using the latter made a clear source approach turn
    # into an imperceptible few centimetres on compact anime VRMs.
    depth_height = model_height
    if names.get("head"):
        feet = [
            armature.data.bones[names[key]].head_local
            for key in ("left_foot", "right_foot")
            if names.get(key)
        ]
        if feet:
            head = armature.data.bones[names["head"]].head_local
            depth_height = max((head - foot).length for foot in feet)
    is_image_plane_motion = target_height > 10.0
    # Semantic GLBs are measured in metres and previously needed a defensive
    # lower scale bound.  RTMW tracks are source pixels: that bound would turn
    # a 100px crouch into a half-metre launch.  Their natural torso-height
    # ratio is exactly the pixel-to-avatar conversion we need.
    translation_scale = model_height / target_height
    if is_image_plane_motion:
        translation_scale = min(0.02, max(0.0001, translation_scale))
    else:
        translation_scale = min(2.0, max(0.25, translation_scale))

    # A number of production VRMs (including Lulum) make the semantic hips a
    # connected child of a non-humanoid root bone.  Connected bones cannot be
    # translated, so putting the root-motion keys on Hips silently produces a
    # static body.  Find the actual top-level rig bone once and move that.
    root_pose = pose[names["hips"]] if names.get("hips") else None
    while root_pose is not None and root_pose.parent is not None:
        root_pose = root_pose.parent
    root_rest = (
        armature.data.bones[root_pose.name].matrix_local.to_3x3().copy()
        if root_pose is not None
        else None
    )
    foot_pose = [
        pose[names[key]]
        for key in ("left_foot", "right_foot")
        if names.get(key)
    ]
    support_height: float | None = None

    def oriented_rest(
        key: str,
        direction: Vector,
        weight: float = 1.0,
        max_angle: float | None = None,
    ) -> Matrix:
        """Make a world-space bone orientation while preserving its rest head."""

        delta = rest_axis[key].rotation_difference(direction)
        if max_angle is not None and delta.angle > max_angle:
            axis, _angle = delta.to_axis_angle()
            delta = Quaternion(axis, max_angle)
        if weight < 1.0:
            delta = Quaternion((1.0, 0.0, 0.0, 0.0)).slerp(delta, weight)
        return (
            Matrix.Translation(rest[key].translation)
            @ delta.to_matrix().to_4x4()
            @ rest[key].to_3x3().to_4x4()
        )

    def segment(key: str, start: str, end: str, frame: int, rotations: dict[str, Matrix]) -> None:
        if key not in rest or not names.get(key):
            return
        raw_direction = landmarks[end][frame] - landmarks[start][frame]
        # The source is a normal front-facing dance video. SAM3D's inferred
        # camera-depth component is unstable for crossed hands and loose
        # clothing, whereas its screen-plane landmarks are stable. Retarget
        # the measured screen pose directly instead of treating the first
        # dance pose as the avatar's T-pose calibration.
        direction = _safe_direction(
            Vector(),
            Vector((raw_direction.x, 0.0, raw_direction.z)),
            rest_axis[key],
        )
        # ``rest[key]`` contains the bone-head position as well as its
        # orientation. Rotate only the latter: left-multiplying the complete
        # rest matrix rotates a leg's head around the world origin and is what
        # previously pulled the avatar apart.
        rotations[key] = oriented_rest(key, direction)

    def finger_bend_specs() -> dict[str, tuple[str, str, str, float, float]]:
        """Return optional HumanBone bend controls keyed by semantic bone name."""

        specs: dict[str, tuple[str, str, str, float, float]] = {}
        for source_side, humanoid_side in (("L", "left"), ("R", "right")):
            root = f"{source_side}HandRoot"
            # A VRM thumb has a metacarpal plus two phalanges.  The first
            # measured bend belongs at Thumb1, then each following joint.
            specs[f"{humanoid_side}_thumb_metacarpal"] = (
                root, f"{source_side}Thumb1", f"{source_side}Thumb2", math.radians(25), 0.35,
            )
            specs[f"{humanoid_side}_thumb_proximal"] = (
                f"{source_side}Thumb1", f"{source_side}Thumb2", f"{source_side}Thumb3", math.radians(45), 0.70,
            )
            specs[f"{humanoid_side}_thumb_distal"] = (
                f"{source_side}Thumb2", f"{source_side}Thumb3", f"{source_side}Thumb4", math.radians(60), 0.75,
            )
            for source_finger, humanoid_finger in (
                ("Index", "index"), ("Middle", "middle"), ("Ring", "ring"), ("Little", "little"),
            ):
                specs[f"{humanoid_side}_{humanoid_finger}_proximal"] = (
                    root, f"{source_side}{source_finger}1", f"{source_side}{source_finger}2", math.radians(70), 0.60,
                )
                specs[f"{humanoid_side}_{humanoid_finger}_intermediate"] = (
                    f"{source_side}{source_finger}1", f"{source_side}{source_finger}2", f"{source_side}{source_finger}3", math.radians(95), 0.85,
                )
                specs[f"{humanoid_side}_{humanoid_finger}_distal"] = (
                    f"{source_side}{source_finger}2", f"{source_side}{source_finger}3", f"{source_side}{source_finger}4", math.radians(65), 0.75,
                )
        return specs

    finger_bends = finger_bend_specs()

    def apply_finger_bend(key: str, frame: int) -> None:
        """Apply measured 2D joint flexion as a constrained local finger bend.

        Absolute image directions are intentionally not assigned to finger
        bones: near-camera hands flip those directions under tiny detector
        noise. 2D does not reveal hand-front/hand-back, so only an unsigned
        local flexion is retained and the VRM's own hand roll is left intact.
        """

        spec = finger_bends.get(key)
        bone_name = names.get(key)
        if spec is None or not bone_name or any(point not in landmarks for point in spec[:3]):
            return
        start, joint, end, maximum, gain = spec
        incoming = landmarks[joint][frame] - landmarks[start][frame]
        outgoing = landmarks[end][frame] - landmarks[joint][frame]
        incoming = Vector((incoming.x, 0.0, incoming.z))
        outgoing = Vector((outgoing.x, 0.0, outgoing.z))
        root_name = "LHandRoot" if start.startswith("L") else "RHandRoot"
        middle_name = "LMiddle1" if start.startswith("L") else "RMiddle1"
        if root_name not in landmarks or middle_name not in landmarks:
            return
        palm_length = (landmarks[middle_name][frame] - landmarks[root_name][frame]).length
        if palm_length < 12.0 or incoming.length < palm_length * 0.06 or outgoing.length < palm_length * 0.06:
            return
        incoming.normalize()
        outgoing.normalize()
        bend = math.acos(max(-1.0, min(1.0, incoming.dot(outgoing))))
        # 2D has no reliable hand-front/hand-back orientation. Keep only the
        # non-negative flexion amount so a mirrored palm cannot reverse-curl
        # a finger through the mesh.
        amount = min(maximum, bend * gain)
        bone = pose[bone_name]
        bone.rotation_mode = "QUATERNION"
        bone.rotation_quaternion = Quaternion((0.0, 0.0, 1.0), amount)
        bone.keyframe_insert(data_path="rotation_quaternion", frame=frame + 1)

    for index in range(count):
        midhip = (landmarks["RHip"][index] + landmarks["LHip"][index]) * 0.5
        midshoulder = (landmarks["RShoulder"][index] + landmarks["LShoulder"][index]) * 0.5
        # For a front-facing video, the hip-to-shoulder vector is a robust
        # screen-plane measure of a lean or a crouch.  Its camera-depth term
        # is intentionally ignored, but it must not be discarded entirely:
        # otherwise the avatar is a floating torso with rotating limbs.
        torso_direction = _safe_direction(
            Vector(),
            Vector((midshoulder.x - midhip.x, 0.0, midshoulder.z - midhip.z)),
            rest_axis.get("spine", Vector((0.0, 0.0, 1.0))),
        )
        rotations: dict[str, Matrix] = {}
        # Spread the measured lean through the torso instead of applying the
        # whole angle to one joint.  This keeps a real VRM's dress, hair and
        # spring-bone hierarchy stable while making the weight shift visible.
        # The values are *local* increments and must add up to less than one
        # full source angle through a connected VRM chain.  Applying a full
        # angle at every serial bone is what makes an upright dancer fold over.
        for key, weight in (("hips", 0.08), ("spine", 0.14), ("chest", 0.22), ("upper_chest", 0.28)):
            if key in rest:
                rotations[key] = oriented_rest(key, torso_direction, weight)
        # The face landmarks offer a stable screen-plane head axis in a
        # front-facing source.  Use only its modest side tilt; depth/yaw is not
        # observed reliably enough here to justify a guessed head turn.
        face_points = [
            landmarks[name][index]
            for name in ("Nose", "REye", "LEye")
            if name in landmarks
        ]
        if face_points:
            face_center = sum(face_points, Vector()) / len(face_points)
            head_direction = _safe_direction(
                Vector(),
                Vector((face_center.x - landmarks["Neck"][index].x, 0.0, face_center.z - landmarks["Neck"][index].z)),
                rest_axis.get("head", Vector((0.0, 0.0, 1.0))),
            )
            if "neck" in rest:
                rotations["neck"] = oriented_rest("neck", head_direction, weight=0.32, max_angle=math.radians(12))
            if "head" in rest:
                rotations["head"] = oriented_rest("head", head_direction, weight=0.58, max_angle=math.radians(18))
        else:
            for key in ("neck", "head"):
                if key in rest:
                    rotations[key] = rest[key]
        segment("left_upper_arm", "LShoulder", "LElbow", index, rotations)
        segment("left_lower_arm", "LElbow", "LWrist", index, rotations)
        segment("right_upper_arm", "RShoulder", "RElbow", index, rotations)
        segment("right_lower_arm", "RElbow", "RWrist", index, rotations)
        segment("left_upper_leg", "LHip", "LKnee", index, rotations)
        segment("left_lower_leg", "LKnee", "LAnkle", index, rotations)
        segment("right_upper_leg", "RHip", "RKnee", index, rotations)
        segment("right_lower_leg", "RKnee", "RAnkle", index, rotations)
        # OpenPose has no toe keypoint.  Keep a foot's rest pitch but orient it
        # with the detected body yaw, rather than extending the shin as a fake toe.
        for key in ("left_foot", "right_foot", "left_hand", "right_hand"):
            if key in rest:
                rotations[key] = rest[key]

        # Assign matrices parent-first. Blender converts the global desired
        # matrix to each pose bone's local basis, retaining valid hierarchy.
        for key, bone_name in names.items():
            if not bone_name or key not in rotations:
                continue
            pb = pose[bone_name]
            pb.matrix = rotations[key]
            pb.rotation_mode = "QUATERNION"
            pb.keyframe_insert(data_path="rotation_quaternion", frame=index + 1)
        for key in finger_bends:
            apply_finger_bend(key, index)
        if root_pose is not None and root_rest is not None:
            # The 2D route supplies ``Root`` as the image-plane pelvis
            # midpoint, which preserves the performer moving across the
            # stage. Individual joints are body-relative; retain the old
            # midpoint fallback for legacy GLB tracks. Work in the source
            # screen plane only:
            # x is horizontal and z is vertical after the Y-up -> Z-up swap.
            source_root = landmarks.get("Root", [midhip] * count)[index]
            source_origin = initial_root if "Root" in landmarks else initial_midhip
            screen_delta = source_root - source_origin
            target_world_delta = Vector((
                max(-0.80, min(0.80, screen_delta.x * translation_scale)),
                0.0,
                max(-0.50, min(0.50, screen_delta.z * translation_scale)),
            ))
            # Coherent shoulder/hip/torso image-scale change is the only
            # reliable depth cue available to this 2D route. A positive scale
            # means the performer is closer to the source camera, which is
            # negative stage Y because AvatarStage's camera faces +Y.
            if "Depth" in landmarks:
                source_depth = landmarks["Depth"][index].x - landmarks["Depth"][0].x
                # This is not reconstructed 3D translation: it is only the
                # coherent, low-frequency scale cue from a fixed camera.
                # Limit it to eight percent of avatar height (and 12 cm) so
                # uncertain image scale can never turn into a camera lunge.
                depth_cap = min(0.12, depth_height * 0.08)
                target_world_delta.y = max(-depth_cap, min(depth_cap, -source_depth * depth_height * 0.85))
            # PoseBone.location is expressed along the root bone's *local*
            # axes; converting avoids accidentally mapping vertical movement
            # into camera depth on rigs whose root is aligned with +Z.
            root_pose.location = root_rest.inverted() @ target_world_delta
            # A hip midpoint alone cannot tell whether the dancer is crouching
            # on one leg or taking off.  In this frontal, full-body route the
            # lower posed foot is the support foot, so retain its initial floor
            # height.  This removes the persistent "floating puppet" look
            # without inventing unstable camera-depth motion.
            if foot_pose:
                bpy.context.view_layer.update()
                lowest_foot = min(bone.matrix.translation.z for bone in foot_pose)
                if support_height is None:
                    support_height = lowest_foot
                else:
                    correction = support_height - lowest_foot
                    root_pose.location += root_rest.inverted() @ Vector((0.0, 0.0, correction))
            root_pose.keyframe_insert(data_path="location", frame=index + 1)
    return count


def _write_retarget(args: argparse.Namespace) -> None:
    if not args.input_vrm or not (args.motion_glb or args.motion_2d) or not args.video_output:
        raise RuntimeError("retarget mode requires input VRM, a motion file, and video output")
    if not args.skip_vrma and not args.vrma_output:
        raise RuntimeError("retarget mode requires --vrma-output unless --skip-vrma is set")
    _clear_scene()
    result = bpy.ops.import_scene.vrm(filepath=args.input_vrm)
    if result != {"FINISHED"}:
        raise RuntimeError(f"VRM import failed: {result}")
    armature = _find_armature()
    landmarks = (
        _read_2d_tracks(Path(args.motion_2d))
        if args.motion_2d
        else _landmark_tracks(_read_glb_positions(Path(args.motion_glb)))
    )
    frame_count = _animate_retarget(armature, landmarks)
    scene = bpy.context.scene
    # VRM spring-bone colliders are helper objects. They are not visible parts
    # of the avatar, but keeping them render-visible forces Eevee to sync a
    # large collider hierarchy for every dance frame.
    for obj in scene.objects:
        if obj.name.startswith("VSB_"):
            obj.hide_render = True
    scene.frame_start = max(1, min(frame_count, args.frame_start or 1))
    scene.frame_end = min(frame_count, args.frame_end) if args.frame_end else frame_count
    if scene.frame_end < scene.frame_start:
        raise RuntimeError("--frame-end must not be before --frame-start")
    scene.render.fps = max(1, min(120, round(args.fps)))
    _setup_stage(armature, args.width, args.height)

    scene.frame_set(scene.frame_start)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = args.poster_output
    bpy.ops.render.render(write_still=True)

    # FK keys above are already portable pose animation. Only constraint-based
    # authoring paths need the exceptionally expensive visual NLA bake.
    has_constraints = any(pb.constraints for pb in armature.pose.bones)
    if has_constraints and not args.skip_vrma:
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.select_all(action="DESELECT")
        armature.select_set(True)
        bpy.ops.object.mode_set(mode="POSE")
        bpy.ops.pose.select_all(action="SELECT")
        bake_result = bpy.ops.nla.bake(
            frame_start=scene.frame_start,
            frame_end=scene.frame_end,
            only_selected=True,
            visual_keying=True,
            clear_constraints=True,
            clear_parents=False,
            use_current_action=True,
            bake_types={"POSE"},
        )
        if bake_result != {"FINISHED"}:
            raise RuntimeError(f"could not bake VRM retarget pose: {bake_result}")
        bpy.ops.object.mode_set(mode="OBJECT")
    if not args.skip_vrma:
        vrma_result = bpy.ops.export_scene.vrma(filepath=args.vrma_output)
        if vrma_result != {"FINISHED"}:
            raise RuntimeError(f"VRMA export failed: {vrma_result}")

    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.filepath = args.video_output
    bpy.ops.render.render(animation=True)
    bpy.ops.wm.save_as_mainfile(filepath=args.blend_output)


def main() -> None:
    args = _args()
    _enable_vrm_addon()
    if args.mode == "starter":
        _write_starter(args)
    else:
        _write_retarget(args)


if __name__ == "__main__":
    main()
