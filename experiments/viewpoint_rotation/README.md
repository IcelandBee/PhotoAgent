# Small Viewpoint Rotation

## 范围与管线

仅实现光心固定、translation=0 的纯旋转，用 CPU OpenCV warpPerspective，不增加模型依赖。输入输出尺寸相同，不扩画布。

- follow_reference：原图（包含人物）→ rotation → reference_viewport → 输出。
- reposition：原图先抠出人物并修补背景 → 背景 rotation → reference_viewport → 合成原人物按 subject.bbox 缩放/移动的图层。
- 独立人物保持原始视角，是允许的构图近似；不旋转人物图层。旋转非零时不执行 subject no-op 优化。
- viewpoint 缺省、mode=none 或所有角度为0，走原有渲染路径；不增加插值。
- viewport 坐标相对于旋转后同尺寸画布；subject bbox 仍相对于最终画布。source_subject_bbox 仍为原图坐标，natural_subject_bbox 为旋转四角包围框再经过 viewport 的估计。
- RenderMeta.viewpoint_warp 包含 H、K、R、参数和旋转阶段有效像素比例。原 padding 仅代表 viewport 矩形填充，不描述旋转的不规则空边。

## 数学与方向

像素/相机坐标：x 向右、y 向下、z 向前。使用列向量。

K = [[f,0,W/2],[0,f,H/2],[0,0,1]]，f=(W/2)/tan(horizontal_fov_deg/2)，默认60°，fx=fy。

对标准右手轴旋转矩阵：

```text
Ry(a) = [[cos(a),0,sin(a)], [0,1,0], [-sin(a),0,cos(a)]]
Rx(a) = [[1,0,0], [0,cos(a),-sin(a)], [0,sin(a),cos(a)]]
Rz(a) = [[cos(a),-sin(a),0], [sin(a),cos(a),0], [0,0,1]]

R_yaw   = Ry(-yaw)
R_pitch = Rx(-pitch)
R_roll  = Rz(-roll)
R = R_roll @ R_pitch @ R_yaw
H = K @ R @ inv(K)
p_target ~ H @ p_source
```

R 表达原相机光线到目标相机坐标的被动变换，不是把正向相机姿态直接乘到图像。固定按 yaw、pitch、roll 顺序作用于光线；混合轴旋转不可交换，不另行采用其他 Euler 约定。warpPerspective 使用正向 H，不设置 WARP_INVERSE_MAP。

| 参数正方向 | 中央附近场景运动 |
| --- | --- |
| yaw 正：相机右转 | 内容向左 |
| pitch 正：相机抬头 | 内容向下 |
| roll 正：相机沿前方视线看顺时针转 | 内容逆时针 |

负角度反向。方向通过亮点实际像素位移测试以及真实图片网格/roll 对照检查，不仅检查矩阵符号。

## 参数与边界

TargetState 增加可选 viewpoint；mode 为 none/rotation。角度有限、各轴限制 ±15°，FoV 限制10～150°，默认60°。mode=none 不接受非零角度，防止静默忽略。若宽高比、FoV、角度组合使正向或逆向投影跨过相机平面，会明确拒绝。

border_mode：constant（默认，使用既有 Renderer fill_color）或 replicate（复制边缘，会拉伸边缘纹理）。本轮不自动裁有效区域、不补全不可见内容。

## CLI

在仓库根目录执行：

```powershell
.\.venv311\Scripts\python experiments/viewpoint_rotation/run.py `
  --input D:/Data/videoagent/test_data/PhotoAgent/agent_test_data/case_001/reference.jpg `
  --output workdir/viewpoint_rotation/single.jpg `
  --yaw 5 --pitch -3 --roll 0 --horizontal-fov 60 --grid
```

单张测试省略 --grid。可选 `--border-mode replicate`、`--fill-color 128 128 128`。

--grid 按列 yaw、按行 pitch，各取 [-10,-5,0,5,10]；roll 使用命令指定值。输出 viewpoint_grid.jpg、25张独立图片、grid_metadata.json；单张旁保存 single.warp.json。CLI 会覆盖指定输出及同目录同名网格产物。PIL 读取时应用 EXIF 方向。

TargetState 示例见：
- ../../examples/40_viewpoint_only__yaw_right_5deg.json
- ../../examples/41_combined__rotation__zoom_in__subject_right.json

静态拟合工具 fit_target_state 本轮没有增加旋转搜索维度；Workflow 没有改动。

## 本机真实验收

素材 case_001/reference.jpg（590×786）。输出根目录 `D:/Project/PhotoAgent/workdir/viewpoint_rotation/`：

- single.jpg：yaw +5°、pitch -3°；single_replicate.jpg：同角度复制边缘。
- viewpoint_grid.jpg：25组 yaw/pitch。
- roll_comparison.jpg：roll -5/0/+5°。
- combined/target_sketch.jpg：旋转 + viewport [0.1,0.1,0.9,0.9] + 人物 bbox [0.65,0.35,0.85,0.9]。人物 mask 复用此前真实CPU检测结果，不额外引入模型。
- compatibility/old_target.jpg：旧 TargetState 重跑结果，与本次修改之前保存的 case_001_acceptance/recommended/best_sketch.jpg **字节完全一致**；记录在 compatibility/result.json。

人工检查：雪山、树木、坡面运动连续，方向符合定义，±10°网格没有异常翻折。±5°较适合常规编辑，默认60°FoV下±10°在此案例稳定，但双轴10°时空边和投影变形更明显。该结论不推广到所有FoV/长宽比；±15°仅是参数限制，不代表画质保证。

截图中的字幕、UI、网格也随整图变形；本轮不识别并固定屏幕 overlay。组合图的人物模糊、原位置修补痕迹来自已有分割与 inpaint，不代表真实新视角照片。

测试：`python -m pytest tests experiments/fit_target_state/tests -q`。覆盖精确identity、正负三轴像素方向、矩阵顺序、连续性、填充模式、无效输入、旧配置与none/零角度一致、组合渲染。

本次完整测试结果：152 passed（CPU）。
