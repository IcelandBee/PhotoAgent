# 统一 Depth + Point Cloud 通路

## 数据流

正式 workflow 的 Planner 后固定执行 estimate_reference_depth → render_target_sketch → validate_sketch。
任何 framing / rotation / translation / identity 请求均执行深度节点；不按运动类型分流。
Integrated workflow 继续使用 TargetPackage 交接，下游验证接受同一套 camera intent。

深度默认使用现有 Depth Anything V2 Small；现有转换将相对逆深度映射到正 Z，再中位归一化。
这不是度量深度恢复。Depth Pro 与 precomputed 输入仍可用，metric Z 只在渲染副本中除以中位数，原始深度保留。

camera_rendering/depth_renderer.py 默认采用深度感知双线性点云渲染：
源 K 反投影出彩色点 → R(X-C) → K_target 投影 → 浮点坐标的 2×2 邻域加权投射。
每个目标像素先求最近深度，再按 abs(z-z_min) <= max(1e-6, 0.01*z_min) 过滤贡献并计算加权 RGB。
nearest-Z 保留为显式 regression/debug 选项（--point-renderer nearest_z），其 3×3 candidates 只生成一次、供三遍计算复用。
geometry.py 提取了原 rotation_matrix 和 mesh transform 的共享数学，不再依赖旧 rasterizer。
移除了单 K 快路、identity 图像复制和 Homography dispatch，因此任意 target K 和目标尺寸均走同一投影实现。

PreparedPointCloud 的 ids/vertices/colors 数组只读。批量每图构建一次，CameraRenderer 按图像内容缓存一次深度，
source K 不变时复用点云；正式每次目标生成构建一次。调用 prepare_point_cloud 的 depth 必须已中位归一化。
直接传入 geometry 时，它是权威源几何，调用方负责将它与原图保持配对。

## 坐标与构图

相机坐标：x 向右、y 向下、z 向前。对外位移采用 +tx 右、+ty 上、+tz 前，故 C=(tx,-ty,tz)。
R = Rz(-roll) Rx(-pitch) Ry(-yaw)，外参平移为 -R C。
旋转 +yaw 右转、+pitch 仰拍、+roll 顺时针；单位度。平移为场景中位深度比例，范围 ±0.2，默认小幅 ±0.05。
角度范围 ±15°，常用 ±8°。viewpoint.mode=camera 才允许非零平移。

source K 来自配置水平 FOV（默认 60°），保持原实验约定，不自动以估计焦距替换。
framing.reference_viewport 使用原 aspect-fit 规则，转换为目标 K 的焦距/主点和画布尺寸。
focal_scale 再围绕目标画布中心缩放 K，范围 [0.5,2]。默认 viewport 与 focal_scale 不改变取景。
整个过程没有渲染后 2D crop、resize 或人物贴图。
viewport/focal 变化不自动推断物理位移；真实前进/后退必须显式设置 translation_z。

## 输出与验证

target_sketch.png 为无文字 PNG；target_sketch_valid_mask.png 为二值 mask；target_sketch.json 为元数据。
result.json 记录 workflow 状态；reference_depth.npy/json 保存本次估计，debug 可输出可视化和 projected_depth.npy。
未知几何区域采用 fill_color，默认灰色，无 inpainting、无复制边缘。
valid/hole 比例来自 mask；padding 字段为历史交接保留，固定 0，不描述非矩形遮挡区域。
校验包括请求与实际 R/t/K、图像/mask 尺寸、二值性、coverage 和留白像素；没有 target ground truth，不评价摄影真实性。

## 迁移

旧 subject=reposition 或 bbox 目标已不支持，需由调用方重新明确相机运动，不做静默换算。
旧 backend=homography/depth_mesh、mesh/yolo CLI 参数删除；只保留 depth_3d renderer。
旧 rotation/none 输入仍接受，但深度成为必需依赖，输出由 JPEG 改为 PNG；从返回路径读取目标，勿硬编码文件名。
每张参考图必须有有效深度；模型不可用时明确失败。
Guidance prompt 与校验接受 translation_x/y/z，禁止独立人物动作；已有 session 目标不能原地覆盖，请创建新 session。
Guidance 的参考图特征匹配仍可用 homography 评估对齐，它不生成草图，不是被移除的渲染分支。

## 已知局限

单图不能恢复遮挡区和原视野外几何。人物轮廓、细杆、头发、透明物体和深度跃变处可能有孔洞或拉伸。
nearest-Z 半径 1 的 3×3 splat 在 identity 时也可能选择邻近较浅像素；半径 0 可用于严格 identity 对照。
双线性投射的 footprint 固定为 2×2，扩大输出尺寸时仍可能产生采样空洞。
模型推理可用 CUDA，点云实现目前仍为 CPU；没有在本次迁移中引入补全或质量优化。
