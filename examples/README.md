# 当前 TargetState 示例

- 00_baseline__full_frame_follow.json：全景、零位移；仍估计深度并进行点云投影。
- camera_up_pitch_down.json：相机上移 0.05，俯拍 8°。
- camera_down_pitch_up.json：相机下移 0.05，仰拍 8°。
- camera_forward_zoom_in.json：相机前进 0.05，焦距乘 1.1。
- camera_backward_wide.json：相机后退 0.05，焦距乘 0.9。
- create_demo_video.py：本地测试视频生成器。

平移单位为场景中位深度比例，所有人物和背景共同投影。
10 个批量 preset 的唯一配置在 experiments/depth_pointcloud_sketch/presets.json。
旧人物编辑与旋转分流示例已移入仓库外归档。

```sh
python app.py --video sample.mp4 --target-state examples/camera_up_pitch_down.json --work-dir workdir/up-001 --debug
```
