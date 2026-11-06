import trimesh
import numpy as np

def normalize_mesh(input_file, output_file):
    """
    标准化网格：居中并缩放
    """
    mesh = trimesh.load(input_file)
    
    # 居中网格
    centroid = mesh.centroid
    mesh.apply_translation(-centroid)
    
    # 缩放网格到单位球内
    extents = mesh.extents
    if np.any(extents):
        mesh.apply_scale(1.0 / extents.max())
    
    # 保存标准化后的网格
    mesh.export(output_file)
    print(f"标准化完成，保存到 {output_file}")

# 使用示例
normalize_mesh("../data/T1_processed.obj", "../data/T1_normalized.obj")