# 静态前后图拟合 TargetState（独立实验）

输入 reference（调整前）和 ground truth（调整后），搜索背景视窗及人物位置/尺度，输出标准 TargetState，再复用正式 `TargetSketchRenderer` 导出草图。

**全部代码、配置、测试仅新增在本目录。未修改 workflow、Graph、Planner、Selector、正式 Schema 或 Renderer，也没有新增依赖。** 此工具不作为正式 Agent 节点。

## 运行

使用仓库现有 Python 3.11 环境，在项目根目录执行：

```powershell
.\.venv311\Scripts\python experiments/fit_target_state/run.py --reference "D:\data\reference.jpg" --ground-truth "D:\data\ground_truth.jpg" --output-dir workdir/fit-case-01 --mode reposition --device cpu --config experiments/fit_target_state/case_config.json
```

可选追加：

```text
--ignore-mask D:\data\ignore_mask.png
--output-width 1280 --output-height 720
--seed 42
```

不传输出尺寸时，**使用 EXIF 方向纠正后的 reference 原尺寸**。同时提供 width/height 才覆盖尺寸。GT 不同宽高比时等比留边，GT bbox 和 ignore mask 同步映射；留边不参加损失，日志记录映射。

纯取景拟合可传 `--mode follow_reference`，不会加载 YOLO；此时人物 loss 权重为零，其余权重重新归一化。若有人物运动需排除，请提供 ignore mask / ignore_rects，或 `gt_subject_bbox` 作为 GT 人物排除框。

输出目录须为空或不存在，脚本不会覆盖旧结果。退出码 0 表示拟合与导出完成，并不代表存在低于某个业务阈值的“正确答案”；1 表示输入、检测或搜索失败。日志输出异常详情。

## 配置

`case_config.json` 是完整可运行的基础配置。CLI 显式参数覆盖 JSON，未显式指定的 CLI 参数不会覆盖 JSON。

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `subject_mode` | reposition | follow_reference / reposition |
| `device` | cpu | YOLO 设备 |
| `yolo_model` | yolo11n-seg.pt | 使用原检测器的标准权重加载方式 |
| `person_confidence` | 0.4 | 检测阈值 |
| `population_size` | 200 | 全局粗搜索随机候选数 |
| `num_rounds` | 4 | 局部细化轮数 |
| `samples_per_round` | 200 | 每轮局部随机候选数，另含坐标下降候选 |
| `seed` | 42 | NumPy 局部随机生成器种子 |
| `preview_max_side` | 320 | 搜索评分的最长边，原图较小时不放大 |
| `zoom_bounds` | [0.4, 3.0] | zoom 搜索范围 |
| `center_bounds` | [-0.25, 1.25] | viewport 中心 x/y 范围 |
| `min_background_fraction` | 0.15 | 剩余可比较背景的最低比例 |
| `gt_subject_bbox` | null | GT 检测失败时的手动回退 bbox |
| `ignore_rects` | [] | GT 原始方向纠正后图像中的归一化忽略矩形 |

`initial_guess` 支持 `zoom_scale`、`viewport_center_x`、`viewport_center_y`，以及可选的 `subject_bottom_center_x`、`subject_bottom_y`、`subject_height`。后面三项均属于**输出画布**归一化坐标。不传人物初值时用 GT 检测的底部中心和高度初始化。主体模式为 reposition 时，reference 检测始终必需，失败直接报错。

GT 的手动回退框示例（在 JSON 中添加）：

```json
{
  "gt_subject_bbox": [0.60, 0.25, 0.78, 0.90],
  "ignore_rects": [[0.0, 0.0, 1.0, 0.10], [0.0, 0.92, 1.0, 1.0]]
}
```

`gt_subject_bbox` 是 **GT 原图**归一化 `[xmin,ymin,xmax,ymax]`。reposition 模式仍先尝试 YOLO，仅失败后使用此值，回退原因写入日志。框应选同一主体；两张图各自采用原检测器的“最大面积、近似面积时靠中心”规则，不含身份匹配。

ignore mask 必须和方向纠正后的 GT 原图尺寸一致。值 0 表示参与，任何非零值（包括 1 / 255）表示忽略；与 `ignore_rects` 取并集。它们描述的是 GT/目标画布中的 UI，不会移除 reference 中的字幕或生成干净背景。输入图片尽量先裁去大面积 UI。

## 几何参数化

- 背景只优化中心 x/y 和 `log(zoom_scale)`。viewport 归一化宽为 `1/zoom_scale`，高为 `reference_width/reference_height × canvas_height/canvas_width / zoom_scale`，因此源像素宽高比与输出画布一致。zoom > 1 放大，zoom < 1 扩面；zoom=1 保留原图全宽（输出比例不同时不保证全高）。
- 人物只优化底部中心 x、bottom y、高度；用 reference mask 的源像素宽高比推导宽度。参数会裁到合法画布范围，保持完整主体和等比约束。标准 TargetState 中不增加内部参数或 source bbox。
- GT 姿态/宽高比与 reference 不一致时，无法同时吻合全部边界；本实验优先底部中心和高度。

## 搜索与复用

先评估 identity + GT 人物种子、用户 initial_guess 及其轴向邻域，避免随机全局候选把搜索过早带到重复背景的错误位置。再全局随机探索，随后每轮坐标下降 + 围绕当前最优的局部随机搜索，步长逐轮减半。

准备阶段 reference/GT 各检测一次，reference 只 inpaint 一次。搜索预览复用正式 `ViewportRenderer`、人物提取、仿射变换、composite 和 no-op 工具，并缓存源图层；**最终输出必须重新调用正式 TargetSketchRenderer**。没有复制模型或改动正式渲染算法。

相同图片、模型结果、配置、依赖和 seed 下搜索可复现。日志的耗时/绝对路径不保证一致，GPU 检测也可能存在非确定性。有限随机搜索不保证全局最优。

## Loss 定义

均越小越好，几何单位为归一化坐标，灰度/边缘归一化到 `[0,1]`。

```text
L_subject = mean(abs(rendered_bottom_center_x - gt_bottom_center_x),
                 abs(rendered_bottom_y - gt_bottom_y),
                 abs(rendered_height - gt_height))

L_background = 0.7 × gray_L1 + 0.3 × Sobel_magnitude_L1
L_global = gray_L1（不排除人物，只排除 UI / GT 留边）

L_total = 0.6 × L_subject + 0.4 × L_background + 0.0 × L_global
```

权重来自 `weights.subject/background/global`，运行时归一化。灰度轻量高斯滤波后计算 Sobel 梯度幅值。背景 loss 排除 GT 主体框、渲染主体框、原人物映射区域（避免填补伪影主导）、ignore mask/rects 和 GT 留边；在预览上额外扩张约 3 像素，避免滤波把字幕/人物边缘泄漏进有效区域。

如果有效背景不足未忽略 GT 区域的 15% 或少于 16 像素，该候选额外加 10 的惩罚；若所有最优候选都不足则明确失败。这个简单保护减少“把背景全部遮掉来降 loss”，但不能完全消除动态掩码的评分偏差。

每个候选都记录各 loss 项、背景有效比例和参数。选优基于低分辨率预览；导出 JPEG 后再次评分，分别记录 `best_loss` 与 `exported_sketch_loss`，不隐藏栅格采样/压缩差异。

## 输出

```text
output_dir/
  best_target_state.json
  best_sketch.jpg
  comparison_triptych.jpg          # reference | sketch | ground truth，带标题
  search_log.json                  # 配置、输入、检测/回退、最佳参数、每轮与每次评估、loss 定义
  debug/
    reference.png                  # 方向纠正后的输入
    gt_native.png
    reference_mask.png             # reposition
    gt_mask.png                    # GT 检测成功时
    effective_ignore_mask.png
    background_valid_mask.png      # 白色表示有效背景，与 ignore mask 语义相反
    masked_gt.jpg
    masked_sketch.jpg
```

正式 Renderer 开启 debug，实际独立编辑时还会在输出根目录保存 background/subject 图层，用于核查 inpaint、分割和几何。

## 验证与本轮 smoke test

```powershell
.\.venv311\Scripts\python -m pytest tests experiments/fit_target_state/tests -q
```

实验测试包含物理比例、配置错误、ignore mask 的 1/255 语义、越界 bbox、GT letterbox、字幕忽略、GT 检测失败回退、搜索复现与改善、默认原图尺寸、实际 Renderer 导出、reposition 和 CLI 覆盖。无需在线权重即可运行这些测试。

本轮尚未收到独立拍摄的真实前后图对，使用滑雪参考帧与**之前 Renderer 生成的已知变换图**作为 synthetic GT，额外添加模拟 UI 字幕及 ignore mask。CPU 自动检测两张图成功，输出 1920×1080。80 个全局样本、4 轮各 80 个局部样本，初始化 loss 约 0.05973，最优预览 0.00974，导出图复算约 0.00902。此结果仅验证管线和已知几何恢复，不代表真实摄影师移动数据上的泛化效果。

本机复现命令（需保留 workdir 中的现有素材，输出目录换新）：

```powershell
.\.venv311\Scripts\python experiments/fit_target_state/run.py --reference workdir/modes-reposition/reference_frame.jpg --ground-truth workdir/fit-smoke-input/gt_with_ui.png --ignore-mask workdir/fit-smoke-input/ignore_mask.png --config workdir/fit-smoke-input/case.json --output-dir workdir/fit-smoke-reposition-rerun --device cpu
```

## 结果不理想时

1. 先看检测与 debug mask：确认两图是同一主体，必要时修正 GT 回退框；错误 reference 分割应换图或模型。
2. 补充 ignore_rects，排除字幕、对焦框、其他运动人物，检查可比较背景是否足够。
3. 调整 zoom/center 搜索范围和 initial_guess，再增加种群或轮数；纹理重复、背景平坦时解可能不唯一。
4. 分别查看 subject/background loss，按实际目标调权重；可提高预览分辨率减少低分辨率局部最优。
5. 真实视差、透视、姿态变化和场景露出不能由本工具的 2D 缩放平移表达。不要把残差都归因于搜索失败；本轮不增加 homography、depth 或生成模型。

文件职责：`config.py` 配置，`geometry.py` 参数化，`images.py` 输入/掩码/三联图，`preview.py` 缓存预览，`loss.py` 评分，`search.py` 搜索，`run.py` CLI 与导出，`tests/test_fit.py` 独立测试。
