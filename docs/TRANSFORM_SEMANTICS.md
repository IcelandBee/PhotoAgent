# Transform 语义与边界

正式流程固定为 **Camera Orientation → 2D Framing → Subject Layout**。

## TargetState

```json
{
  "viewpoint": {"mode": "rotation", "yaw_deg": 5, "pitch_deg": 0, "roll_deg": 0},
  "framing": {"reference_viewport": [0.1, 0, 1.1, 1]},
  "subject": {"mode": "reposition", "bbox": [0.65, 0.25, 0.85, 0.85]}
}
```

subject、framing 必填；viewpoint 可省略，默认 none/零角度。viewpoint 仅允许 mode=none|rotation 和三个角度（有限值，±15°）；none 不允许非零角度。rotation 的三个角度全零时同样跳过视点变换。未知字段明确拒绝。

subject.mode=follow_reference 时 bbox 必须为空，整幅参考图一起变换；reposition 时必须给出 [0,1] 内的有效 bbox。bbox 是最终画布上人物的 envelope，保持原人物长宽比，以底部中心对齐 contain，因此实际 mask bbox 不一定填满 envelope。

reference_viewport 为 [x1,y1,x2,y2]，允许超出 [0,1]，宽高必须正。坐标相对于同尺寸的旋转后参考画布。沿用既有输出比例适配，以 rendered_viewport 为实际结果；不新增自动裁切。

## 唯一职责入口

| 意图 | Target 字段 | 含义 |
| --- | --- | --- |
| 整幅画面左右/上下取景调整 | framing.reference_viewport | 人物与背景一起二维变化 |
| crop、zoom in/out、expand | framing.reference_viewport | 整图尺度与采样窗口 |
| 相机原地右转/左转 | viewpoint.yaw_deg | 固定光心，正值右转 |
| 抬头/压低镜头 | viewpoint.pitch_deg | 固定光心，正值抬头 |
| 手机倾斜 | viewpoint.roll_deg | 正值沿镜头前方看顺时针 |
| 人物相对背景单独移动/缩放 | subject.bbox | 最终 canvas 人物布局 |
| 相机物理平移 | 无正式字段 | 仅底层研究保留 |

“画面右移”“右边多留空间”等结果描述优先映射二维 framing；如指人物相对背景独立移动则使用 subject.bbox。不能自动推导 camera translation。只有明确朝向描述才映射 yaw/pitch/roll。

**方向区别**：[0.1,0,1.1,1] 是采样视窗向右 0.1，已有图像内容在输出中向左；如果要求已有内容向右移动，窗口需要向左。这是二维操作，没有平移视差，也不能凭空生成画外内容。

## 渲染与路由

| 角度是否非零 | AgentConfig.viewpoint_backend | Viewpoint renderer | 深度节点 |
| --- | --- | --- | --- |
| 否 | 任意 | 不调用，metadata backend=none | 不执行 |
| 是 | homography（默认） | RotationViewpointWarper | 不执行 |
| 是 | depth_3d | PointCloudViewpointWarper，translation=0 | 执行 |
| 是 | depth_mesh | MeshViewpointWarper，translation=0 | 执行 |

只有 `rotation_active AND backend in {depth_3d,depth_mesh}` 才进入深度节点。framing-only、subject-only、framing+subject 不加载深度模型、不做 mesh GPU 预检。

配置字段为 viewpoint_backend、viewpoint_horizontal_fov_deg（默认60）、viewpoint_border_mode（默认constant）。mesh 只允许 constant。CLI 对应 --viewpoint-backend、--viewpoint-horizontal-fov-deg、--viewpoint-border-mode。这些均不由 Planner 的 TargetState 决定。

```sh
python app.py --video /data/ski.mp4 --target-state examples/viewpoint_yaw_right.json --work-dir workdir/yaw
python app.py --video /data/ski.mp4 --target-state examples/viewpoint_yaw_right.json --viewpoint-backend depth_3d --work-dir workdir/point_yaw
```

reposition 保留已有实现：从原图提取人物和修补背景 → 背景旋转 → 背景 viewport → 将原人物放进最终 bbox → 合成。人物放好后不会再旋转。自然位置的辅助投影只用于 metadata 和 no-op 判断，不改变最终布局。点云、mesh 几何与光栅化算法未修改；不添加 safe crop 或生成式补图。

## API 隔离与迁移

旧 translation_x/y/z（包括值0）、mode=depth_3d/depth_mesh、horizontal_fov_deg、border_mode 在正式 viewpoint 中均报错。不会忽略、不会自动换算 viewport。旧调用方必须重新明确是二维构图、朝向还是人物布局；渲染选项迁至 AgentConfig。

`renderer/camera_parameters.py` 中 CameraWarpParameters 是低层参数容器，仅供 renderer/research 使用。正式 rotation_parameters adapter 只接收公共朝向和 renderer 配置，显式构造 translation_x/y/z=0。Planner、graph、集成 handoff、Guidance 输入均不能携带这些平移字段。

旧50–55、60–63案例已迁至 [camera_translation 研究目录](../experiments/camera_translation/README.md)。该独立 runner 直接调用低层 warper，不进入 TargetState、Planner 或集成 workflow。

## Guidance 与调试

Guidance 使用独立 dataclass 校验同样的四个 viewpoint 字段，graph 和直接 backend 调用均拒绝旧输入。Prompt 将 viewport 解释为 desired 2D framing、朝向为 explicit camera orientation、bbox 为 desired subject layout；明确禁止仅凭 viewport shift 推断 move_right 等物理平移。真实导航仍可依据独立图像证据建议移动。

RenderMeta 记录 viewpoint_applied、实际 viewpoint_backend（未应用为none）、viewpoint_rotation、camera_translation=[0,0,0]、requested/rendered_viewport、subject_mode、subject_transform_applied。viewpoint_warp 保留底层诊断，包括固定为零的 translation 参数；这不是公开语义输入。

## 验证范围

tests/test_transform_semantics.py 验证无旋转跳过所有 warper、像素级旋转先于 viewport、最终 bbox、三类组合、三 backend 的 depth 路由、旧字段拒绝、metadata 篡改拒绝及研究 runner 真正保留平移。Guidance/集成测试验证边界拒绝发生在模型调用之前。低层 point/mesh 的 translation/R+t 测试继续保留；CPU mesh oracle 仅用于测试，不是正式 fallback。GPU测试跳过不代表GPU验证通过。
