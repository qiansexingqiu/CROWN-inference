import argparse
import json
import math
import os
import sys
import time
import setproctitle   
setproctitle.setproctitle("dental")

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 将项目根目录添加到Python路径
sys.path.insert(0, project_root)

import torch
from loguru import logger
from scipy import linalg
from sklearn.neighbors import KDTree

# =============== Global variables ===============
print("初始化模型...")
# 修复导入问题，明确导入所需的模型类
from src.models import *
from src.models.pct_models import Pct, PctPatchRefine

model_rotation = Pct({})
model_centroid_prediction = PointnetCentroidPredictionNet()
model_centroid_prediction_10k = PointnetCentroidPredictionNet(out_points=1024)
model_all_tooth_seg = AllToothSegNet(3)
model_refine_pct_16k = PctPatchRefine(2)
model_refine_pct_32k = PctPatchRefine()

model_cls = ToothClsNet()
print("模型初始化完成")

torch.random.manual_seed(2000228)
torch.cuda.manual_seed(2000228)


# ============= Program init methods =============
def init_logger():
    logger.remove()
    logger.add(sys.stdout, colorize=True, enqueue=True, backtrace=True, diagnose=True)


def make_arg_parser() -> argparse.ArgumentParser:
    """
    Prepare CLI arg parser

    :return: argparse instance
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    # Basic information
    parser.add_argument('--input', '-i', type=str, help='Path of input model', required=True)
    parser.add_argument('--output', '-o', type=str, help='Path of output results', required=False)
    parser.add_argument('--model_path', '-m', type=str, help='Path of trained models', required=False,
                        default='../model')
    return parser


def check_and_prepare_files() -> None:
    """
    Check and prepare needed files
    """
    print("检查和准备文件...")
    if not os.path.exists(ARGS.input) or os.path.isdir(ARGS.input):
        logger.error('Invalid input [{}]', ARGS.input)
        exit(-1)

    if not os.path.exists(os.path.dirname(ARGS.output)):
        os.makedirs(os.path.dirname(ARGS.output), exist_ok=True)

    if not os.path.exists(os.path.dirname(ARGS.model_path)):
        logger.error('Model path [{}] not exists', ARGS.model_path)
        exit(-1)
    print("文件检查和准备完成")


# =========== Data persistence methods ===========
def load_checkpoint(model_path) -> None:
    print(f"开始加载模型检查点，路径: {model_path}")
    model_rotation_file = os.path.join(model_path, 'model_rotation')
    model_centroid_prediction_file = os.path.join(model_path, 'model_pred_centroid')
    model_centroid_prediction_10k_file = os.path.join(model_path, 'model_pred_centroid_10k')
    model_all_tooth_seg_file = os.path.join(model_path, 'model_all_tooth_seg')
    model_refine_file = os.path.join(model_path, 'model_pct_refine_16k')
    model_refine_file_32k = os.path.join(model_path, 'model_pct_refine_32k')
    model_cls_file = os.path.join(model_path, 'model_cls')
    
    print("正在加载各个模型组件...")
    print(f"  - 旋转模型: {model_rotation_file}")
    model_rotation.load_state_dict(torch.load(model_rotation_file))
    
    print(f"  - 质心预测模型 (10k): {model_centroid_prediction_10k_file}")
    model_centroid_prediction_10k.load_state_dict(torch.load(model_centroid_prediction_10k_file))
    
    print(f"  - 质心预测模型: {model_centroid_prediction_file}")
    model_centroid_prediction.load_state_dict(torch.load(model_centroid_prediction_file))
    
    print(f"  - 全牙分割模型: {model_all_tooth_seg_file}")
    model_all_tooth_seg.load_state_dict(torch.load(model_all_tooth_seg_file))
    
    print(f"  - 精炼模型 (16k): {model_refine_file}")
    model_refine_pct_16k.load_state_dict(torch.load(model_refine_file))
    
    print(f"  - 精炼模型 (32k): {model_refine_file_32k}")
    model_refine_pct_32k.load_state_dict(torch.load(model_refine_file_32k))
    
    print(f"  - 分类模型: {model_cls_file}")
    model_cls.load_state_dict(torch.load(model_cls_file))
    
    print("所有模型检查点加载完成")


def rotation_matrix_torch(axis, theta):
    """
    Generalized 3d rotation via Euler-Rodriguez formula, https://www.wikiwand.com/en/Euler%E2%80%93Rodrigues_formula
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    if torch.sqrt(torch.dot(axis, axis)) < 1e-8:
        return torch.eye(3, requires_grad=True).cuda()
    axis = axis / torch.sqrt(torch.dot(axis, axis))

    a = torch.cos(theta / 2.0)
    b, c, d = -axis * torch.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return torch.tensor([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def apply_rotation(nd_data, axis_up, axis_forward):
    rot_axis = np.cross(axis_up, np.array([0, 0, 1]))
    cos_theta = np.sum(axis_up * np.array([0, 0, 1])) / np.linalg.norm(axis_up)
    rot_angle = np.arccos(np.clip(cos_theta, -1, 1))
    rot_matrix = linalg.expm(np.cross(np.eye(3), rot_axis / linalg.norm(rot_axis) * rot_angle))
    nd_data[:, 0:3] = np.matmul(rot_matrix, nd_data[:, 0:3, np.newaxis])[:, :, 0]
    nd_data[:, 3:6] = np.matmul(rot_matrix, nd_data[:, 3:6, np.newaxis])[:, :, 0]
    axis_up = np.matmul(rot_matrix, axis_up)
    axis_forward = np.matmul(rot_matrix, axis_forward)

    rot_axis = np.cross(axis_forward, [0, -1, 0])
    cos_theta = np.sum(axis_forward * np.array([0, -1, 0])) / np.linalg.norm(axis_forward)
    rot_angle = np.arccos(np.clip(cos_theta, -1, 1))
    rot_matrix = linalg.expm(np.cross(np.eye(3), rot_axis / linalg.norm(rot_axis) * rot_angle))
    nd_data[:, 0:3] = np.matmul(rot_matrix, nd_data[:, 0:3, np.newaxis])[:, :, 0]
    nd_data[:, 3:6] = np.matmul(rot_matrix, nd_data[:, 3:6, np.newaxis])[:, :, 0]
    axis_up = np.matmul(rot_matrix, axis_forward)
    axis_forward = np.matmul(rot_matrix, axis_forward)

    return nd_data, axis_up, axis_forward


def inference(input, output, model_path='../model', rotation=True, debug_vis=False, output_json=False):
    print("=" * 60)
    print("开始执行推理过程")
    print("=" * 60)
    inf_start_time = time.time()
    ##########################################
    # Init
    ##########################################
    print("1. 初始化阶段")
    print(f"   - 加载检查点，模型路径: {model_path}")
    load_checkpoint(model_path)
    batch_size = 8
    print("   - 初始化完成")

    ##########################################
    # Data init
    ##########################################
    print("2. 数据初始化阶段")
    print(f"   - 创建推理数据集，输入文件: {input}")
    from src.data.TeethInferDataset import TeethInferDataset
    infer_set = TeethInferDataset(input)
    print("   - 数据初始化完成")

    ##########################################
    # Model
    ##########################################
    print("3. 模型设置阶段")
    print("   - 将模型移至GPU并设置为评估模式")
    model_rotation.cuda()
    model_rotation.eval()
    model_all_tooth_seg.cuda()
    model_all_tooth_seg.eval()
    model_centroid_prediction.cuda()
    model_centroid_prediction.eval()
    model_centroid_prediction_10k.cuda()
    model_centroid_prediction_10k.eval()
    model_refine_pct_16k.cuda()
    model_refine_pct_16k.eval()
    model_refine_pct_32k.cuda()
    model_refine_pct_32k.eval()
    model_cls.cuda()
    model_cls.eval()
    print("   - 模型设置完成")

    with torch.no_grad():
        start_time = time.time()

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # % Stage 0: rotate dental model
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        print(f"4. 模型旋转阶段 (启用: {rotation})")
        if rotation:
            print("   - 获取数据张量 (4k)")
            data_tensor = infer_set.get_data_tensor(False, four_k=True)
            print("   - 预测旋转轴")
            pred_axis_up, pred_axis_forward = model_rotation(data_tensor[:, :, 0:6].permute(0, 2, 1))
            pred_axis_up, pred_axis_forward = pred_axis_up[0], pred_axis_forward[0]

            standard_ax_up = torch.Tensor([0, 0, 1]).cuda()
            standard_ax_forward = torch.Tensor([0, -1, 0]).cuda()

            print("   - 计算上轴旋转矩阵")
            if torch.dot(pred_axis_up, standard_ax_up) < 0.99:
                R_ax_up = torch.nn.functional.normalize(torch.cross(pred_axis_up, standard_ax_up), p=2, dim=-1)
                R_angle_up = torch.arccos(
                    torch.dot(pred_axis_up, standard_ax_up) / (torch.norm(pred_axis_up) * torch.norm(standard_ax_up)))
                R_up = rotation_matrix_torch(R_ax_up, R_angle_up).cuda()

                pred_axis_forward = torch.matmul(R_up, pred_axis_forward.unsqueeze(1)).squeeze()
            else:
                R_up = torch.eye(3).cuda()

            # 投影到xy平面
            pred_axis_forward[-1] = 0

            print("   - 应用上轴旋转")
            data_tensor[0, :, 0:3] = torch.matmul(data_tensor[0, :, 0:3], R_up.T)
            data_tensor[0, :, 3:6] = torch.matmul(data_tensor[0, :, 3:6], R_up.T)

            print("   - 计算前轴旋转矩阵")
            if torch.dot(pred_axis_forward, standard_ax_forward) < 0.99:
                R_ax_forward = torch.nn.functional.normalize(
                    torch.cross(pred_axis_forward, standard_ax_forward), p=2, dim=-1)
                R_angle_forward = torch.arccos(torch.dot(pred_axis_forward, standard_ax_forward) / (
                        torch.norm(pred_axis_forward) * torch.norm(standard_ax_forward)))
                R_forward = rotation_matrix_torch(R_ax_forward, R_angle_forward).cuda()
            else:
                R_forward = torch.eye(3).cuda()

            print("   - 应用前轴旋转")
            data_tensor[0, :, 0:3] = torch.matmul(data_tensor[0, :, 0:3], R_forward.T)
            data_tensor[0, :, 3:6] = torch.matmul(data_tensor[0, :, 3:6], R_forward.T)

            if debug_vis:
                print("   - 保存旋转后的数据用于调试")
                np.savetxt(
                    f'<REDACTED>',
                    data_tensor.data.cpu().numpy()[0, :, 0:6])

            rot_matrix = torch.matmul(R_forward, R_up)

            print("   - 设置旋转矩阵")
            infer_set.set_rotate_matrix_inplace(rot_matrix.detach().cpu().numpy())

            logger.info('Stage 0: "rotation" done, spent {}s', time.time() - start_time)
            print(f"   - 模型旋转阶段完成，耗时: {time.time() - start_time:.2f}秒")
        else:
            print("   - 跳过模型旋转")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # % Stage 1: all tooth seg
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        print("5. 全牙分割阶段")
        print("   - 获取数据张量")
        data_tensor = infer_set.get_data_tensor(False)
        start_time = time.time()
        print("   - 执行全牙分割预测")
        pred_seg = torch.argmax(model_all_tooth_seg(data_tensor[:, :, 0:6]), dim=1)
        pred_seg = pred_seg.detach().cpu().numpy()[0]

        if debug_vis:
            print("   - 保存分割结果用于调试")
            np.savetxt(
                f'<REDACTED>',
                vis_teeth_seg(data_tensor[0, :, 0:6].data.cpu().numpy(), pred_seg))

        print("   - 移除牙齿上的曲率")
        infer_set.remove_curvatures_on_tooth(pred_seg)
        print("   - 重新获取数据张量")
        data_tensor = infer_set.get_data_tensor(False)

        print("   - 后处理分割结果")
        seg = np.array(pred_seg)
        seg[seg > 0.5] = 1
        seg[seg < 1] = 0
        if np.max(seg) < 0.5:
            seg[seg < 0.5] = 1
        else:
            # 扩大一圈预测
            nd_data = data_tensor[0].data.cpu().numpy()[:, 0:3]
            tree = KDTree(nd_data)
            neighbours = tree.query_radius(nd_data[seg > 0], 0.15, return_distance=False)
            for n in neighbours:
                seg[n] = 1

        # 0714: 二次归一化来预测质心
        # data_tensor, positive_dists_centroid, positive_dists_max = infer_set.normalize_after_pred_seg(data_tensor, seg)

        logger.info('Stage 1: "all tooth seg" done, spent {}s', time.time() - start_time)
        print(f"   - 全牙分割阶段完成，耗时: {time.time() - start_time:.2f}秒")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # % Stage 2: centroid prediction
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        print("6. 质心预测阶段")
        start_time = time.time()
        print("   - 执行质心预测")
        # kp_reg1, kp_score, seed_xyz = model_centroid_prediction(data_tensor, False, False)
        kp_reg, kp_score, seed_xyz = model_centroid_prediction(data_tensor[:, seg > 0.5, :], False, False)
        kp_reg2, kp_score2, seed_xyz = model_centroid_prediction_10k(data_tensor[:, seg > 0.5, :], False, False)

        kp_reg = torch.cat([kp_reg, kp_reg2], dim=1)
        kp_score = torch.cat([kp_score, kp_score2], dim=1)

        kp_reg = kp_reg.detach().cpu().numpy()
        kp_score = kp_score.detach().cpu().numpy()

        print("   - 过滤低置信度质心")
        kp_reg = kp_reg[0, kp_score[0] < 0.2, :]
        if debug_vis:
            print("   - 保存质心预测结果用于调试")
            np.savetxt(f'<REDACTED>',
                       kp_reg)

        print("   - 聚类质心")
        kp_reg = get_clustered_centroids(np.asarray(kp_reg, dtype=np.float64))

        if len(kp_reg) == 0:
            print("   - 未检测到质心，创建空标签")
            final_labels = np.zeros((pred_seg.shape[0],))
        else:
            if debug_vis:
                print("   - 保存聚类质心用于调试")
                np.savetxt(f'<REDACTED>',
                           vis_teeth_centroids(data_tensor[0, :, 0:6].data.cpu().numpy(), kp_reg))

            logger.info('Stage 2: "centroid prediction" done, spent {}s', time.time() - start_time)
            print(f"   - 质心预测阶段完成，耗时: {time.time() - start_time:.2f}秒")

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # % Stage 3: patches refine
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            print("7. 补丁精炼阶段")
            start_time = time.time()

            # Stage 2 preprocess: split patches
            print("   - 创建质心补丁 (_16k=True)")
            infer_set.make_patches_centroids(kp_reg, _16k=True)
            print("   - 创建质心补丁 (_16k=False)")
            infer_set.make_patches_centroids(kp_reg, _16k=False)

            # Stage 2: patch inference
            print("   - 获取补丁张量")
            patches_tensor_16k = infer_set.get_patches_tensor(True)
            patches_tensor_32k = infer_set.get_patches_tensor(False)

            print("   - 执行补丁推理")
            all_pred_seg = np.array([])
            all_pred_seg_16k = np.array([])
            batch_count = math.ceil(len(patches_tensor_32k) / batch_size)
            print(f"   - 总共 {batch_count} 个批次，批次大小: {batch_size}")
            for i in range(batch_count):
                print(f"     处理批次 {i+1}/{batch_count}")
                if i * batch_size == len(patches_tensor_32k) - 1:
                    print("       - 处理16k补丁 (重复批次)")
                    pred_seg1 = model_refine_pct_16k(
                        patches_tensor_16k[i * batch_size:(i + 1) * batch_size].repeat(2, 1, 1).transpose(2,
                                                                                                          1))  # [:, 0, :]
                    pred_seg1 = torch.argmax(pred_seg1, dim=1)
                    pred_seg1 = pred_seg1[0:1]

                    print("       - 处理32k补丁 (重复批次)")
                    pred_seg2 = model_refine_pct_32k(
                        patches_tensor_32k[i * batch_size:(i + 1) * batch_size].repeat(2, 1, 1).transpose(2,
                                                                                                          1))  # [:, 0, :]
                    pred_seg2 = torch.argmax(pred_seg2, dim=1)
                    pred_seg2 = pred_seg2[0:1]
                else:
                    print("       - 处理16k补丁")
                    pred_seg1 = model_refine_pct_16k(
                        patches_tensor_16k[i * batch_size:(i + 1) * batch_size].transpose(2, 1))  # [:, 0, :]
                    pred_seg1 = torch.argmax(pred_seg1, dim=1)

                    print("       - 处理32k补丁")
                    pred_seg2 = model_refine_pct_32k(
                        patches_tensor_32k[i * batch_size:(i + 1) * batch_size].transpose(2, 1))  # [:, 0, :]
                    pred_seg2 = torch.argmax(pred_seg2, dim=1)

                # pred_seg1 +=> pred_seg2
                print("       - 后处理预测结果")
                pred_seg1 = pred_seg1.detach().cpu().numpy()
                pred_seg2 = pred_seg2.detach().cpu().numpy()

                patch_idx_16k = infer_set.patch_indices_16k[i * batch_size:(i + 1) * batch_size]
                patch_idx_32k = infer_set.patch_indices_32k[i * batch_size:(i + 1) * batch_size]
                points_16k = infer_set.__getitem__(0)[patch_idx_16k][:, :, 0:3]
                points_32k = infer_set.__getitem__(0)[patch_idx_32k][:, :, 0:3]
                feats_16k_32k = batch_point_labels_interpolation(points_32k, points_16k, pred_seg1)
                feats_16k_32k[feats_16k_32k > 0.8] = 1
                feats_16k_32k[feats_16k_32k < 1] = 0
                pred_seg2 += feats_16k_32k.astype(np.int64)

                pred_seg2 = np.clip(pred_seg2, 0, 1)
                all_pred_seg = np.array([*all_pred_seg, *pred_seg2])
                all_pred_seg_16k = np.array([*all_pred_seg_16k, *pred_seg1])

            print("   - 后处理所有预测结果")
            all_pred_seg_16k[all_pred_seg_16k > 0.5] = 1
            all_pred_seg_16k[all_pred_seg_16k < 1] = 0
            pred_seg = all_pred_seg
            pred_seg[pred_seg > 0.5] = 1
            pred_seg[pred_seg < 1] = 0

            print("   - 执行标签去噪")
            pred_seg = postprocessing.infer_labels_denoise(infer_set.patches_32k, pred_seg)
            logger.info('Stage 3: "patch inference" done, spent {}s', time.time() - start_time)
            print(f"   - 补丁精炼阶段完成，耗时: {time.time() - start_time:.2f}秒")

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # % Stage 4: patches classification
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            print("8. 补丁分类阶段")
            start_time = time.time()

            print("   - 获取分类补丁张量")
            patches_tensor_16k, resamples_tensor_16k = infer_set.get_cls_patches_tensor(all_pred_seg_16k, True)

            print("   - 执行补丁分类")
            all_pred_cls = np.array([])
            # all_pred_quad = np.array([])
            batch_count = math.ceil(len(patches_tensor_16k) / batch_size)
            print(f"   - 总共 {batch_count} 个批次，批次大小: {batch_size}")
            for i in range(batch_count):
                print(f"     处理批次 {i+1}/{batch_count}")
                if i * batch_size == len(patches_tensor_16k) - 1:
                    print("       - 处理补丁分类 (重复批次)")
                    pred_cls = model_cls(patches_tensor_16k[i * batch_size:(i + 1) * batch_size].repeat(2, 1, 1),
                                         resamples_tensor_16k[i * batch_size:(i + 1) * batch_size].repeat(2, 1, 1))
                    pred_cls = pred_cls[0:1]
                else:
                    print("       - 处理补丁分类")
                    pred_cls = model_cls(patches_tensor_16k[i * batch_size:(i + 1) * batch_size],
                                         resamples_tensor_16k[i * batch_size:(i + 1) * batch_size])
                pred_cls = pred_cls.detach().cpu().numpy()
                pred_cls = np.argmax(pred_cls, axis=1)

                # 0-32还原为0-48
                pred_cls[pred_cls > 0] = np.ceil(pred_cls[pred_cls > 0] / 8) * 10 + (pred_cls[pred_cls > 0] - 1) % 8 + 1
                all_pred_cls = np.array([*all_pred_cls, *pred_cls])

            if debug_vis:
                print("   - 保存预测补丁用于调试")
                #########################################################
                # 保存预测的Patch
                out_dir = f'<REDACTED>'
                os.makedirs(out_dir, exist_ok=True)
                for patch_id in range(pred_seg.shape[0]):
                    nd_out = vis_teeth_heatmap(infer_set.patches_16k[patch_id, :, 0:6],
                                               infer_set.patch_heatmap_16k[patch_id, :, 0])
                    np.savetxt(os.path.join(out_dir, f'heatmap_{patch_id}.txt'), nd_out)
                    nd_out = vis_teeth_seg_cls(infer_set.patches_16k[patch_id, :, 0:6], all_pred_seg_16k[patch_id],
                                               all_pred_cls[patch_id],
                                               infer_set.patch_centroids_16k[patch_id])
                    # nd_out = vis_teeth_seg_cls(patches_tensor[patch_id, :, 0:6], patches_tensor[patch_id, :, 6],
                    #                            all_pred_cls[patch_id], None)
                    np.savetxt(os.path.join(out_dir, f'{patch_id}.txt'), nd_out)
                    # np.savetxt(os.path.join(out_dir, f'mask_{patch_id}.txt'), infer_set.patches_32k[patch_id, pred_seg[patch_id] > 0.5, :])
                #########################################################

            logger.info('Stage 4: "classification" done, spent {}s', time.time() - start_time)
            print(f"   - 补丁分类阶段完成，耗时: {time.time() - start_time:.2f}秒")

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # % Stage 5: post processing
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            print("9. 后处理阶段")
            print("   - 获取分类补丁张量用于后处理")
            patches_tensor, _ = infer_set.get_cls_patches_tensor(pred_seg, False)
            print("   - 重新排列标签")
            final_labels = postprocessing.rearrange_labels3_backup(patches_tensor[0, :, 0:3].data.cpu().numpy(),
                                                            patches_tensor[:, :, 6].data.cpu().numpy(),
                                                            all_pred_cls)

        print("   - 返回插值结果")
        infer_set.return_back_interpolation(final_labels)
        logger.info('Inference spent {}s', time.time() - inf_start_time)
        print(f"   - 后处理阶段完成")

        print("=" * 60)
        print(f"推理过程完成，总耗时: {time.time() - inf_start_time:.2f}秒")
        print("=" * 60)

        # Output
        print("10. 输出结果阶段")
        if output is not None:
            print(f"   - 保存输出结果到: {output}")
            infer_set.save_output(output)
            with open(output.replace('.obj', '.json'), 'w') as fp:
                json.dump({
                    "labels": infer_set.class_results.tolist()
                }, fp)
                fp.close()
            print("   - 输出结果保存完成")



if __name__ == '__main__':
    print("程序启动")
    init_logger()
    parser = make_arg_parser()
    ARGS = parser.parse_args()
    print(f"解析参数: {ARGS}")
    check_and_prepare_files()
    print("开始执行推理")
    inference(ARGS.input, ARGS.output, ARGS.model_path, debug_vis=True, output_json=True, rotation=True)
    print("程序执行完成")