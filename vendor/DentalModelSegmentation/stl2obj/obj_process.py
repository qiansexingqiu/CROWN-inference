import trimesh
import numpy as np

def preprocess_mesh(input_file, output_file):
    """
    预处理网格：修复、清理和标准化
    """
    # 加载网格
    mesh = trimesh.load(input_file)
    
    # 修复网格
    if not mesh.is_watertight:
        print("网格不是封闭的，尝试修复...")
        mesh.fill_holes()
    
    # 修复法线
    mesh.fix_normals()
    
    # 移除孤立的顶点
    mesh.remove_unreferenced_vertices()
    
    # 保存处理后的网格
    mesh.export(output_file)
    print(f"预处理完成，保存到 {output_file}")
    
    # 输出网格信息
    print(f"顶点数: {len(mesh.vertices)}")
    print(f"面片数: {len(mesh.faces)}")
    print(f"是否封闭: {mesh.is_watertight}")

# 使用示例
preprocess_mesh("../data/T1.obj", "../data/T1_processed.obj")