# 使用 evdetmav_detector.py 从 ZIP 内 H5 重新检测

输入仅为 FRED/8.zip、20.zip、51.zip、65.zip、93.zip 内的 Event/events.hdf5。整个任务不读取 RGB 图片或 RGB 视频。

H5 使用 ECF 压缩，先由官方解码器无损转为普通 H5（字段 x、y、p、t 不变），随后通过独立命令行进程直接运行仓库 evdetmav_detector.py。没有复用之前的检测坐标，也没有以直接调用 process_window 代替该入口。

参数：1280×720，时间单位 us，30 ms 窗口、30 ms 步长；其他算法参数保持命令行默认值，包括显著性、周期性和精细定位。与上一轮自定义循环相比，这次遵循 CLI 对窗口末端事件的包含规则以及最少事件数/最短窗口条件，因此边界窗口可能产生差异，不保证与旧结果逐条相同。

每段目录保存：
- command.json：实际执行命令及原 ZIP 来源。
- detector.log：原入口的标准输出和错误日志。
- evdetmav_detections_batch.csv、per_file/：原入口写出的检测结果。
- source_manifest.csv：解码后的临时输入路径；视频生成后会清理临时 H5。
- Detection.mp4：使用本轮 CSV 重绘的检测视频，1280×720、3 像素框、22 像素 Times New Roman 标签。
- video_windows.csv：视频各帧的事件窗口。
- status.json：完成状态。

保存 PNG 的开关关闭以避免大量重复图像；视频从同一 H5 事件与本轮检测 CSV 生成。低于 CLI 最少事件数的窗口不产生检测框，仍保留视频画面和播放时间。

五段全部完成后拼接为 Detection_all.mp4，保持序列号顺序。code_sha256.json 记录本次代码版本哈希。当前完整视频已按五段核验；单段视频副本已清理，逐段 CSV 保留。

