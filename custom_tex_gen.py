# Models, pipelines and other functions and interfaces are to be used according to the terms
# dictated in the TENCENT HUNYUAN 3D 2.0 COMMUNITY LICENSE AGREEMENT

from hy3dgen.texgen import Hunyuan3DPaintPipeline
import torch
import trimesh
import argparse
from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
from hy3dgen.shapegen import FloaterRemover, DegenerateFaceRemover, FaceReducer
import os


def reduce_faces(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh, max_facenum=max_faces)
    return mesh


def process_3d_file(file_path: str) -> trimesh.Trimesh:
    try:
        mesh = trimesh.load(file_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh.dump())
        if mesh.faces.shape[0] > 50000:
            mesh = reduce_faces(mesh, 50000)
            print("Mesh has been reduced to 50,000 faces")
        return mesh

    except Exception as e:
        raise RuntimeError(f"Error processing the mesh file '{file_path}': {e}")


def validate_args(config: dict):
    # Validate mesh path exists
    if not os.path.exists(config.mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {config.mesh_path}")

    # Validate image prompts folder has at least one image file
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    image_files = [
        os.path.join(config.image_prompts, f)
        for f in os.listdir(config.image_prompts)
        if f.lower().endswith(valid_extensions)
    ]

    if not image_files:
        raise FileNotFoundError(
            f"No image files found in the directory: {config.image_prompts}"
        )

    # Validate mesh file extension
    if not config.mesh_path.lower().endswith((".glb", ".gltf")):
        raise ValueError("Mesh file must be a .glb or .gltf")

    # Check for high render resolution
    if config.render_size > 2048:
        print(
            "Warning: Render resolution above 2048 or more will significantly impact inference time. You can cancel with Ctrl+C."
        )

    # Check for multiple image prompts
    if len(image_files) > 1:
        print(
            "Warning: Using multiple image prompts will significantly impact inference time. You can cancel with Ctrl+C."
        )
    return image_files


if __name__ == "__main__":
    # Create a pipeline
    pipe = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
    pipe.config.device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(
        description="Generate textures for 3D models using Hunyuan3D-2 with added supprot for custom 3D models, multiple image prompts, custom texture resolution, and render size."
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        required=True,
        help="The path to the mesh file, should be a .glb/.gltf",
    )
    parser.add_argument(
        "--image_prompts",
        type=str,
        required=True,
        help="Directory with only images to be fed to the model as prompts",
    )
    parser.add_argument(
        "--texture_resolution",
        type=int,
        default=2048,
        help="Resolution to generate texture, negligible impact on inference speed",
    )
    parser.add_argument(
        "--render_size",
        type=int,
        default=1024,
        help="Resolution of the renders of the 3D model generated and fed to the diffusion model",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="The output file path, e.g., output.glb",
    )

    parser.add_argument(
        "--inference_steps",
        type=int,
        default=25,
        help="The steps the diffusion model takes to generate the texture. Too many or too few steps may produce poor results.",
    )
    config = parser.parse_args()

    pipe.render = MeshRender(
        default_resolution=config.render_size, texture_size=config.texture_resolution
    )

    image_files = validate_args(config)

    mesh = process_3d_file(config.mesh_path)

    mesh = pipe(mesh, images=image_files, inf_steps=config.inference_steps)
    mesh.export(config.output_path, include_normals=True)
