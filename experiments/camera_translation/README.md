# Camera translation — research only

此目录保留旧 depth_3d/depth_mesh 平移示例与低层实验入口。它们不是正式 Composition TargetState，不能传给 app.py、Planner 或 integrated workflow。

```sh
python -m experiments.camera_translation.run --reference /data/reference.png --depth /data/reference_depth.npy --parameters experiments/camera_translation/examples/51_depth3d__translate_right_small.json --output-dir workdir/research_translation
```

支持单独 CameraWarpParameters JSON 或归档示例中的 viewpoint 对象。归档 framing/subject 字段只为保存历史记录，runner 不执行它们。仅输出 research_warp.png、valid_mask.png、research_metadata.json。深度必须与 reference 对应；可读取同名 JSON sidecar，mesh 仍需真实 CUDA/PyTorch3D。

低层参数和方向、尺度沿用原实现：正 translation_x/y/z 表示相机右/上/前移；归一化位移按 scene_reference_depth 换算。点云与 mesh 的 R+t、投影、遮挡能力及测试均保留，没有新增正式 camera translation API。

正式旋转与组合编辑请使用 [Transform 语义](../../docs/TRANSFORM_SEMANTICS.md) 和根 examples 的七个新案例。
