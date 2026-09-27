# 多人眼部监测与测试

2026-09-27：打瞌睡模板扩展为最多 5 张人脸。仍使用已安装的 MediaPipe Face Landmarker，无需新增权重或 API Key。视频上的 ID 是一次连续跟踪的编号，不代表人员真实身份。

## 使用

1. 刷新网页，上传下表视频。
2. 选择“离线模板 → 打瞌睡监测 → 生成场景配置”。
3. 保留连续闭眼 2 秒、稳定睁眼 0.5 秒、闭眼 EAR 0.20、睁眼 EAR 0.24 等默认设置。
4. 点击“应用配置 → 开始分析”。每个人都有自己的框、ID、眼部点和计时；事件表、截图、CSV/JSON 同样携带 ID。
5. “规则过程与未报警原因”可查看某个视频时刻所有人的双眼 EAR 和计时状态。ID 顺序由首次检测决定，不固定代表左右位置。

| 视频（位于 assets/demo_videos） | 内容 | 尺寸 / 时长 | 本机默认配置实测 |
| --- | --- | --- | --- |
| multi_person_clear_1080p.mp4 | 真实双人近景，包含睁眼和持续闭眼动作 | 1920×1012 / 20 秒 / 25 FPS | 500 帧均检测到 2 张脸，两个 ID 全程稳定；ID 1（画面右侧）7.96 秒触发闭眼提示 |
| multi_face_staggered_composite.mp4 | 同一演员的两路错时拼接，右路提前 3 秒，末尾定格补齐 | 1080×1008 / 10.15 秒 / 20 FPS | 203 帧均检测到 2 张脸；ID 2 在 3.40 秒报警、6.60 秒恢复；ID 1 在 6.40 秒报警、9.60 秒恢复 |

第二个文件是工程测试片，画面顶部已标记 COMPOSITE TEST；不是两个人同场拍摄。原始左右人脸像素尺寸保持不变，时间错开用于核验独立计时与恢复。
真实双人片中，另一人的 EAR 并未连续满足默认双眼阈值，因此没有第二条报警。清晰画面不意味着统一 EAR 阈值适合每个人；可通过规则过程查看测量并按机位调整，但不能为追求报警直接放宽到把睁眼也算成闭眼。
这些是眼部动作演示，不是睡眠医学标注素材。

## 实现

- `src/models/face_landmarker_adapter.py`：模型允许最多 5 张脸；逐脸使用对应关键点和朝向矩阵，输出 `observations`。
- `src/tracking/face_tracker.py`：用相邻帧的人脸框匹配。IoU 至少 0.2 且双方候选唯一才延续 ID；缺帧、时间间隔超过 0.5 秒、交叉重叠导致匹配不明确时重新编号。不会按模型返回列表的顺序当作身份。
- `src/rules/drowsiness_rules.py`：每个 ID 一套闭眼、睁眼恢复与报警状态。离开画面的状态立即清理；无效观测清空该人的连续计时。两人同时报警时各自产生事件，一人恢复不会清除另一人的提示。
- `src/pipeline/analysis_pipeline.py`：每帧传递 `faces`，逐人计数并将观测写入 `rule_progress.jsonl` 的 `faces`；旧 `face` 字段仅保留单脸兼容用途，多人消费者应读取 `faces`。
- `src/visualization/scene_overlay.py`：按 ID 匹配框颜色、计时和恢复事件，底部最多显示三条，完整列表在网页的规则过程里。
- `src/llm/prompts.py`、`requirement_guard.py`：DeepSeek 可以生成多人持续闭眼需求；当前一组阈值共享给所有人，不支持指定真实身份或每个人不同阈值。

遮挡、快速换位及相同位置无缝换人仍可能导致丢失或错误关联；该跟踪器不做人脸身份识别。多于 5 张脸不会全部处理。结果应在自己的机位、人员与光线条件下继续验证。

## 验证和素材来源

自动测试覆盖检测顺序变化、独立报警/恢复、同时报警、单人失踪/无效不干扰他人、交叉匹配重置、长时间间隔和各脸独立朝向计算。完整测试：231 项通过。

完整视频推理并转码验证了原单人片和上述两个新样片，共 906 帧，结果均可完整解码，事件截图与 CSV/JSON 已生成。单人片仍在 6.40 秒报警、9.65 秒恢复。验收报告：`runs/multi_person_source/clear_verification.json`；结果目录：`runs/verified-multiface-<文件名去扩展名>/`。

真实双人片由 cottonbro studio 发布于 [Pexels](https://www.pexels.com/video/two-people-looking-at-the-camera-6559938/)，按 [Pexels License](https://www.pexels.com/license/) 下载使用。从 4096×2160 原文件截取前 20 秒，保持宽高比缩小并移除音轨。原始文件保留在 `runs/multi_person_source/pexels_6559938.mp4`。
拼接片来源于 [OpenVino-Driver-Behaviour 的 video2.mp4](https://github.com/incluit/OpenVino-Driver-Behaviour/blob/master/data/video2.mp4)，项目 Apache-2.0 许可已保留在示例目录。没有修改眼部动作或生成虚构人脸。

下载/处理可重现命令：`.venv\Scripts\python.exe scripts\prepare_multiface_samples.py`。文件尺寸、来源、处理方式、SHA-256 和完整解码记录见 `assets/demo_videos/monitoring_samples.json`。
