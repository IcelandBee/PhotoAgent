# 案例索引

按文件名前缀选取：`00_baseline` 原图基准、`10–14_frame_only` 仅调整取景、`20–23_subject_only` 仅编辑人物、`30–31_combined` 组合编辑。所有 JSON 都能直接作为 `--target-state` 输入。

## 案例清单

| 文件 | 编辑效果 | 人物模式 |
| --- | --- | --- |
| [00_baseline__full_frame_follow.json](00_baseline__full_frame_follow.json) | 完整原图，作为其他案例的对照 | follow_reference |
| [10_frame_only__zoom_in_center.json](10_frame_only__zoom_in_center.json) | 中心裁剪，人物与背景一起放大 | follow_reference |
| [11_frame_only__zoom_out_center.json](11_frame_only__zoom_out_center.json) | 整图缩小，四周灰色扩面 | follow_reference |
| [12_frame_only__pan_viewport_right.json](12_frame_only__pan_viewport_right.json) | 视窗右移：内容向左移动，右边出现填充，不缩放 | follow_reference |
| [13_frame_only__expand_left.json](13_frame_only__expand_left.json) | 向左扩面；比例适配同时补充上下空间 | follow_reference |
| [14_frame_only__crop_right_and_expand_left.json](14_frame_only__crop_right_and_expand_left.json) | 裁掉原图右侧，同时向左扩面 | follow_reference |
| [20_subject_only__move_left_keep_size.json](20_subject_only__move_left_keep_size.json) | 人物左移 25% 画布宽度，保持原显示大小 | reposition |
| [21_subject_only__move_right_keep_size.json](21_subject_only__move_right_keep_size.json) | 人物右移 25% 画布宽度，保持原显示大小 | reposition |
| [22_subject_only__shrink_keep_bottom_center.json](22_subject_only__shrink_keep_bottom_center.json) | 人物宽高缩小到原来的 60%，底部中心不变 | reposition |
| [23_subject_only__enlarge_keep_bottom_center.json](23_subject_only__enlarge_keep_bottom_center.json) | 人物宽高放大到原来的 150%，底部中心不变 | reposition |
| [30_combined__frame_zoom_in__subject_left_larger.json](30_combined__frame_zoom_in__subject_left_larger.json) | 背景中心放大，人物独立左移并放大到 150% | reposition |
| [31_combined__frame_zoom_out__subject_right_smaller.json](31_combined__frame_zoom_out__subject_right_smaller.json) | 背景中心扩面，人物独立右移并缩小到 60% | reposition |

`frame_only` 中人物、滑雪板和背景保持原始关系，不加载 YOLO。`subject_only` 中 viewport 恒为完整原图，但移除原人物时仍会对局部背景做 inpaint。`combined` 中人物与背景使用各自的目标参数。

案例 14 的请求 viewport 为 `[-0.2, 0.1, 0.9, 0.9]`。在本演示的 16:9 输入和输出下，为保持比例，实际 viewport 为 `[-0.2, -0.05, 0.9, 1.05]`：右侧裁剪保留，上下转为少量扩面。请以 `result.json` 的 `rendered_viewport` 为准。

## 人物案例的校准基准

文件名中的“保持大小”“缩小”“放大”相对于 `00_baseline`，使用以下固定输入：

- 视频：`mixkit-children-skiing-on-the-plain-of-a-pine-forest-3349-full-hd.mp4`。
- 采样间隔 `30`，随机 seed `42`，选择第 `30` 帧。
- 输出 `1280 × 720`；模型 `yolo11n-seg.pt`、置信度 `0.4`、CPU。
- 当前检测的紧 mask bbox 源像素为 `[941, 564, 1101, 932]`，原图大小 `1920 × 1080`。

这些数值仅用于离线编写**目标 bbox**。运行时仍自动检测人物，不向 TargetState 注入 source bbox，也不人工替代检测。

**TargetState 使用绝对目标坐标，不是“向左移动 25%”这种相对指令。** 换视频、参考帧、模型或输出比例后，JSON 仍然可运行，但文件名所描述的相对幅度不一定成立，需要重设目标 bbox。人物缩放以底部中心固定，所以缩小时头部会下移，这是预期对齐方式。

## 运行一个案例

在项目根目录执行；本开发机使用 `.venv311`，其他环境可替换为自己的 Python：

```powershell
$video = "D:\Data\videoagent\test_data\PhotoAgent\mixkit-children-skiing-on-the-plain-of-a-pine-forest-3349-full-hd.mp4"
$case = "31_combined__frame_zoom_out__subject_right_smaller"
.\.venv311\Scripts\python app.py --video $video --target-state "examples/$case.json" --work-dir "workdir/examples-demo/$case" --frame-sample-interval 30 --random-seed 42 --target-width 1280 --target-height 720 --device cpu --debug
```

修改 `$case` 即可选择其他案例。每次运行使用新的输出目录；查看 `target_sketch.jpg` 和 `result.json`。独立编辑仍可能出现人物白边、填补痕迹及滑雪板留在原地的现象，几何 PASS 不代表修图质量通过。

## 辅助脚本与旧案例迁移

保留 `create_demo_video.py`，可生成不依赖下载的合成视频，适合测试 `frame_only`。它画出的简化人物不保证被 YOLO 识别，因此人物编辑示例请使用真实视频。

旧 `manual_target_state.json`、`follow_reference.json`、`reposition.json`、`viewport_*.json`、`subject_move_left.json`、`subject_zoom_out_right.json` 已由上表案例替代并删除。无需维护多份名字含糊或效果重复的参数文件。

## 小视角旋转

- [40_viewpoint_only__yaw_right_5deg.json](40_viewpoint_only__yaw_right_5deg.json)：相机右转5°，整图内容向左，人物随图旋转。
- [41_combined__rotation__zoom_in__subject_right.json](41_combined__rotation__zoom_in__subject_right.json)：yaw +5°、pitch -3°，再裁剪放大，人物独立放到右侧。bbox 为绝对目标位置，不使用前述滑雪人物的相对大小校准。

旋转先于 viewport；此时 viewport 坐标相对于同尺寸的旋转后画布。详细方向与静态图片 CLI 见 [旋转实验说明](../experiments/viewpoint_rotation/README.md)。

## 深度机位编辑

| 案例 | 作用 |
| --- | --- |
| [50_depth3d__identity.json](50_depth3d__identity.json) | 深度backend零运动，对照原图 |
| [51_depth3d__translate_right_small.json](51_depth3d__translate_right_small.json) | 相机右移0.03，近景向左移动更多 |
| [52_depth3d__translate_forward_small.json](52_depth3d__translate_forward_small.json) | 相机前移0.03 |
| [53_depth3d__translate_backward_small.json](53_depth3d__translate_backward_small.json) | 相机后移0.03 |
| [54_combined__depth3d__zoom_in__subject_right.json](54_combined__depth3d__zoom_in__subject_right.json) | 3D旋转平移、显式裁剪、独立人物右置 |
| [55_depth3d__translate_up_small.json](55_depth3d__translate_up_small.json) | 相机上移0.03，画面向下 |

这些平移量以深度中位数1为单位，不是米，也不是画面宽高比例。全部通过 app.py 的正式工作流；运行方法和深度复用说明见 [depth_3d文档](../docs/depth3d.md)。旧00和40分别复用为none及rotation基准。

## Depth Pro + Mesh（可选GPU）

- [60_depthmesh__identity.json](60_depthmesh__identity.json)：零运动mesh基准。
- [61_depthmesh__translate_right_small.json](61_depthmesh__translate_right_small.json)：右移0.03。
- [62_depthmesh__translate_forward_small.json](62_depthmesh__translate_forward_small.json)：前移0.03。
- [63_combined__depthmesh__zoom_in__subject_right.json](63_combined__depthmesh__zoom_in__subject_right.json)：mesh旋转平移、viewport、人物独立编辑。

默认depth backend为Depth Pro；支持显式选择Depth Anything或预计算深度。GPU服务器运行和A/B步骤见 [depth_mesh说明](../docs/depth_mesh.md)。本机CPU未执行PyTorch3D GPU渲染。
