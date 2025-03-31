import trimesh


def ply_to_obj(ply_file, obj_file=None, prettify_gradio=True):
    """
    Convert a .ply file to a .obj file.
    """
    if obj_file is None:
        obj_file = ply_file.replace(".ply", ".obj")

    mesh = trimesh.load_mesh(ply_file)
    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {ply_file}")
    mesh.export(obj_file)

    return obj_file


def convert_hex_to_rgb(hex_color, normalize=True):
    """
    Convert a hex color to an RGB color.
    """
    hex_color = hex_color.lstrip("#")
    hex_color = tuple(int(hex_color[i: i + 2], 16) for i in (0, 2, 4))
    if normalize:
        return tuple([c / 255.0 for c in hex_color])
    return hex_color
