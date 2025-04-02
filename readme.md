# VideoTranscriptWidget

## 功能简介
VideoTranscriptWidget 是一个为 ShellAgent 设计的自定义 widget，用于从 YouTube 视频中提取对话/字幕内容。当视频没有可用字幕时，它会自动切换到使用语音识别技术来提取对话内容。

## 核心功能
- 支持从 YouTube 视频 URL 提取字幕
- 当视频没有字幕时，自动下载音频并使用语音识别(Whisper)
- 支持多语言字幕提取和识别
- 可选时间戳标记
- 提供详细的输出信息，包括视频标题、文本来源和语言

## 使用方法
1. 将 YouTube 视频 URL 作为输入
2. 可选择指定字幕语言或使用自动检测
3. 根据需要开启时间戳显示
4. 若需强制使用语音转文本，可开启 force_stt 选项

### 独立使用方法
此版本的 widget 已经修改为可以在没有 ShellAgent 环境的情况下独立运行：

```python
from VideoTranscriptWidget import VideoTranscriptWidget

widget = VideoTranscriptWidget()
config = {
    "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "language": "auto",
    "include_timestamps": True
}

result = widget({}, config)
print(result["transcript"])
```

## 输入参数
- `video_url`: YouTube 视频 URL (必需)
- `language`: 字幕语言代码，如'en'或'zh-cn'，默认'auto'
- `include_timestamps`: 是否在输出中包含时间戳，默认 False
- `force_stt`: 是否强制使用语音识别，即使有可用字幕，默认 False

## 输出结果
- `transcript`: 提取的文本内容
- `source`: 文本来源，'subtitles'或'speech_recognition'
- `video_title`: 视频标题
- `language`: 检测到的语言
- `error`: 错误信息 (如果有)

## 依赖项
- youtube-transcript-api: 获取 YouTube 字幕
- pytube: 下载 YouTube 视频音频 (主要方法)
- yt-dlp: 下载 YouTube 视频音频 (备选方法，更加稳定)
- whisper: 语音识别功能
- torch: Whisper 深度学习后端
- ffmpeg-python: 音频处理
- pydub: 音频处理工具
- pydantic: 数据验证

## 注意事项
- 使用语音识别转换功能需要安装 FFmpeg
- 如果遇到 `HTTP Error 400: Bad Request` 错误，程序将自动切换到使用 yt-dlp 尝试下载
- 语音识别部分可能需要较长时间处理，特别是对于长视频
- 若有 GPU 支持，将自动使用 CUDA 加速转写过程
- 此版本已经适配为不依赖 ShellAgent 环境也可独立使用
