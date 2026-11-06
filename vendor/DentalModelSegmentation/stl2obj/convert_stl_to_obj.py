import trimesh
import os
import argparse

def convert_stl_to_obj(stl_file, obj_file):
    """
    将STL文件转换为OBJ文件
    """
    # 加载STL文件
    mesh = trimesh.load(stl_file)
    
    # 保存为OBJ格式
    mesh.export(obj_file)
    print(f"已将 {stl_file} 转换为 {obj_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将STL文件转换为OBJ文件")
    parser.add_argument("input", help="输入STL文件路径")
    parser.add_argument("output", help="输出OBJ文件路径")
    
    args = parser.parse_args()
    
    convert_stl_to_obj(args.input, args.output)