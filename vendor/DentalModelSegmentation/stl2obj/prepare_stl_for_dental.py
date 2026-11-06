import trimesh
import numpy as np
import argparse
import os

def process_stl_for_dental_segmentation(stl_file, output_obj_file):
    """
    将STL文件处理为适合牙齿分割的OBJ文件
    """
    print("开始处理STL文件...")
    
    # 1. 加载STL文件
    print("1. 加载STL文件...")
    mesh = trimesh.load(stl_file)
    print(f"   原始网格信息: {len(mesh.vertices)} 顶点, {len(mesh.faces)} 面片")
    
    # 2. 修复网格
    print("2. 修复网格...")
    if hasattr(mesh, 'fill_holes'):
        mesh.fill_holes()
    mesh.fix_normals()
    mesh.remove_unreferenced_vertices()
    
    # 3. 标准化网格
    print("3. 标准化网格...")
    # 居中
    centroid = mesh.centroid
    mesh.apply_translation(-centroid)
    # 缩放到单位球
    extents = mesh.extents
    if np.any(extents):
        mesh.apply_scale(1.0 / extents.max())
    
    # 4. 保存为OBJ格式
    print("4. 保存为OBJ格式...")
    mesh.export(output_obj_file)
    
    print(f"处理完成!")
    print(f"   最终网格信息: {len(mesh.vertices)} 顶点, {len(mesh.faces)} 面片")
    print(f"   输出文件: {output_obj_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将STL文件处理为适合牙齿分割的OBJ文件")
    parser.add_argument("input_stl", help="输入STL文件路径")
    parser.add_argument("output_obj", help="输出OBJ文件路径")
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_stl):
        print(f"错误: 输入文件 {args.input_stl} 不存在")
        exit(1)
    
    # 处理文件
    process_stl_for_dental_segmentation(args.input_stl, args.output_obj)