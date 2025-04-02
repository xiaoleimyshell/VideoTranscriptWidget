import os
import re
import tempfile
import time
from typing import Optional, List, Dict, Any, Union
from pydantic import Field, BaseModel
import logging
from urllib.parse import urlparse, parse_qs

# 由于缺少proconfig模块，创建模拟的基础结构
class BaseWidget:
    """Base widget class for custom implementations"""
    
    class InputsSchema(BaseModel):
        """Base input schema class"""
        pass
    
    class OutputsSchema(BaseModel):
        """Base output schema class"""
        pass
        
    def __call__(self, environ, config):
        """Call method to execute the widget"""
        return self.execute(environ, config)

# 模拟WIDGETS注册装饰器
class WidgetRegistry:
    """Widget registry to collect custom widgets"""
    def register_module(self):
        """Decorator to register widgets"""
        def decorator(cls):
            return cls
        return decorator

WIDGETS = WidgetRegistry()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@WIDGETS.register_module()
class VideoTranscriptWidget(BaseWidget):
    """
    Widget for extracting transcripts from YouTube videos.
    Falls back to audio extraction and speech-to-text if no transcript is available.
    """
    CATEGORY = "Custom Widgets/Media Processing"
    NAME = "Video Transcript Extractor"
    
    class InputsSchema(BaseWidget.InputsSchema):
        video_url: str = Field(
            default="", 
            description="YouTube视频URL"
        )
        language: str = Field(
            default="auto", 
            description="字幕语言代码 (例如 'en', 'zh-cn', 'auto'表示自动检测)"
        )
        include_timestamps: bool = Field(
            default=False, 
            description="是否在输出中包含时间戳"
        )
        force_stt: bool = Field(
            default=False, 
            description="强制使用语音识别转写，即使已有字幕"
        )
    
    class OutputsSchema(BaseWidget.OutputsSchema):
        transcript: str = Field(description="提取的文本内容")
        source: str = Field(description="文本来源: 'subtitles' 或 'speech_recognition'")
        video_title: str = Field(description="视频标题")
        language: str = Field(description="检测到的语言")
        error: Optional[str] = Field(None, description="错误信息")
    
    def extract_video_id(self, video_url: str) -> str:
        """从YouTube URL中提取视频ID"""
        # 处理常规YouTube URL
        if "youtube.com/watch" in video_url:
            parsed_url = urlparse(video_url)
            return parse_qs(parsed_url.query).get('v', [''])[0]
        # 处理短链接
        elif "youtu.be/" in video_url:
            return video_url.split("youtu.be/")[1].split("?")[0]
        # 如果不是有效格式，返回空
        return ""
    
    def get_youtube_transcript(self, video_id: str, language: str = "auto") -> Dict[str, Any]:
        """尝试获取YouTube字幕"""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
            
            if language == "auto":
                # 自动获取可用的字幕
                try:
                    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                    # 尝试获取自动生成的字幕，首先尝试英文或中文
                    try:
                        transcript = transcript_list.find_transcript(['en', 'zh-Hans'])
                        transcript_data = transcript.fetch()
                        detected_language = transcript.language_code
                    except Exception:
                        # 如果没有找到指定语言，尝试获取任何可用字幕
                        transcript = transcript_list.find_generated_transcript(['en'])
                        if not transcript:
                            # 如果没有自动生成的字幕，获取任何可用的字幕
                            for transcript in transcript_list:
                                break
                        transcript_data = transcript.fetch()
                        detected_language = transcript.language_code
                except Exception as e:
                    logger.warning(f"无法自动获取字幕: {str(e)}")
                    # 尝试直接获取英文字幕
                    try:
                        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
                        detected_language = 'en'  # 假设为英文
                    except Exception as direct_error:
                        return {"success": False, "error": f"无法获取字幕: {str(direct_error)}"}
            else:
                # 获取指定语言的字幕
                try:
                    transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
                    detected_language = language
                except (NoTranscriptFound, TranscriptsDisabled) as e:
                    logger.warning(f"指定语言的字幕不可用: {str(e)}")
                    return {"success": False, "error": f"指定语言的字幕不可用: {str(e)}"}
            
            return {
                "success": True,
                "transcript_data": transcript_data,
                "language": detected_language
            }
        except Exception as e:
            logger.error(f"获取YouTube字幕时出错: {str(e)}")
            return {"success": False, "error": f"获取YouTube字幕失败: {str(e)}"}
    
    def get_video_info(self, video_url: str) -> Dict[str, Any]:
        """获取视频信息，包括标题"""
        try:
            from pytube import YouTube
            # 简化初始化方式，避免不兼容的参数
            yt = YouTube(video_url)
            return {"success": True, "title": yt.title, "yt_object": yt}
        except Exception as e:
            logger.error(f"获取视频信息时出错: {str(e)}")
            
            # 尝试使用替代方法从视频ID获取标题
            try:
                video_id = self.extract_video_id(video_url)
                if video_id:
                    # 使用 youtube-transcript-api 获取相关信息
                    from youtube_transcript_api import YouTubeTranscriptApi
                    # 由于无法直接获取标题，我们至少可以得到视频ID
                    dummy_title = f"YouTube Video (ID: {video_id})"
                    
                    # 创建一个简单的 yt_object 替代品
                    class DummyYouTubeObject:
                        def __init__(self, video_id):
                            self.video_id = video_id
                    
                    return {
                        "success": True, 
                        "title": dummy_title, 
                        "yt_object": DummyYouTubeObject(video_id)
                    }
            except Exception as alt_error:
                logger.error(f"替代方法获取视频信息也失败: {str(alt_error)}")
            
            return {"success": False, "error": f"获取视频信息失败: {str(e)}"}
            
    def download_audio(self, yt_object, output_dir: str = None) -> Dict[str, Any]:
        """从YouTube视频下载音频"""
        try:
            if output_dir is None:
                output_dir = tempfile.mkdtemp()
            
            # 检查是否为模拟对象
            if hasattr(yt_object, 'video_id') and not hasattr(yt_object, 'streams'):
                # 这是我们的备选对象，使用另一种方法下载音频
                video_id = yt_object.video_id
                
                # 使用 youtube-dl 或其他方法下载
                try:
                    import subprocess
                    output_file = os.path.join(output_dir, f"{video_id}.mp3")
                    
                    # 尝试使用 yt-dlp (youtube-dl 的更好替代)
                    cmd = [
                        "yt-dlp",
                        "-x",  # 提取音频
                        "--audio-format", "mp3",
                        "-o", output_file,
                        f"https://www.youtube.com/watch?v={video_id}"
                    ]
                    
                    logger.info(f"执行命令: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True)
                    
                    if os.path.exists(output_file):
                        return {"success": True, "audio_path": output_file}
                except Exception as e:
                    logger.error(f"使用替代方法下载音频失败: {str(e)}")
                    return {"success": False, "error": f"下载音频失败: {str(e)}"}
            
            # 常规 pytube 方式
            # 获取仅包含音频的流
            audio_stream = yt_object.streams.filter(only_audio=True).first()
            if not audio_stream:
                return {"success": False, "error": "无法找到音频流"}
            
            # 下载音频文件到临时目录
            audio_file = audio_stream.download(output_path=output_dir)
            
            # 将文件转换为更常用的格式 (如果需要)
            audio_path = audio_file
            logger.info(f"音频已下载到: {audio_path}")
            
            return {"success": True, "audio_path": audio_path}
        except Exception as e:
            logger.error(f"下载音频时出错: {str(e)}")
            
            # 尝试使用 yt-dlp 作为备用方法
            if hasattr(yt_object, 'video_id'):
                try:
                    import subprocess
                    video_id = self.extract_video_id(f"https://www.youtube.com/watch?v={yt_object.video_id}")
                    output_file = os.path.join(output_dir, f"{video_id}.mp3")
                    
                    cmd = [
                        "yt-dlp",
                        "-x",  # 提取音频
                        "--audio-format", "mp3",
                        "-o", output_file,
                        f"https://www.youtube.com/watch?v={video_id}"
                    ]
                    
                    logger.info(f"使用备用方法下载音频，执行命令: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True)
                    
                    if os.path.exists(output_file):
                        return {"success": True, "audio_path": output_file}
                except Exception as backup_error:
                    logger.error(f"备用方法下载音频也失败: {str(backup_error)}")
            
            return {"success": False, "error": f"下载音频失败: {str(e)}"}
    
    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """使用语音识别转写音频内容"""
        try:
            import whisper
            import torch
            
            # 检查CUDA可用性以选择设备
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"使用 {device} 进行转写")
            
            # 加载模型 (选择合适的大小)
            model = whisper.load_model("medium", device=device)
            
            # 执行转写
            result = model.transcribe(audio_path)
            
            return {
                "success": True, 
                "transcript_data": result["segments"],
                "language": result["language"]
            }
        except Exception as e:
            logger.error(f"转写音频时出错: {str(e)}")
            return {"success": False, "error": f"音频转写失败: {str(e)}"}
    
    def format_transcript(self, transcript_data: List[Dict], include_timestamps: bool = False) -> str:
        """将字幕数据格式化为易读的文本"""
        if not transcript_data:
            return ""
        
        formatted_text = ""
        
        # 检查是否需要转换transcript_data类型
        from youtube_transcript_api.formatters import Formatter
        if hasattr(transcript_data, "__iter__") and not isinstance(transcript_data, list):
            try:
                # 转换为列表格式
                transcript_data = list(transcript_data)
            except Exception as e:
                logger.warning(f"转换字幕格式时出错: {str(e)}")
                # 尝试直接获取字幕文本
                try:
                    transcript_text = ""
                    for snippet in transcript_data:
                        if hasattr(snippet, "text"):
                            if include_timestamps and hasattr(snippet, "start"):
                                minutes = int(snippet.start) // 60
                                seconds = int(snippet.start) % 60
                                timestamp = f"[{minutes:02d}:{seconds:02d}] "
                                transcript_text += f"{timestamp}{snippet.text}\n"
                            else:
                                transcript_text += f"{snippet.text}\n"
                    return transcript_text.strip()
                except Exception as text_error:
                    logger.error(f"直接获取字幕文本时出错: {str(text_error)}")
                    return "无法解析字幕数据"
        
        for item in transcript_data:
            try:
                # 尝试不同方式访问字幕文本和时间戳
                if isinstance(item, dict) and 'text' in item and 'start' in item:
                    if include_timestamps:
                        # 格式化时间戳 (秒转换为 MM:SS 格式)
                        minutes = int(item['start']) // 60
                        seconds = int(item['start']) % 60
                        timestamp = f"[{minutes:02d}:{seconds:02d}] "
                        formatted_text += f"{timestamp}{item['text']}\n"
                    else:
                        formatted_text += f"{item['text']}\n"
                elif hasattr(item, 'text') and hasattr(item, 'start'):
                    if include_timestamps:
                        minutes = int(item.start) // 60
                        seconds = int(item.start) % 60
                        timestamp = f"[{minutes:02d}:{seconds:02d}] "
                        formatted_text += f"{timestamp}{item.text}\n"
                    else:
                        formatted_text += f"{item.text}\n"
            except Exception as e:
                logger.warning(f"格式化字幕项时出错: {str(e)}")
                continue
        
        return formatted_text.strip()
    
    def execute(self, environ, config):
        # 开始计时
        start_time = time.time()
        stage_times = {}
        
        # 确保配置对象正确访问
        try:
            # 尝试以 easydict 方式访问配置
            video_url = config.video_url
            language = config.language
            include_timestamps = config.include_timestamps
            force_stt = config.force_stt
        except AttributeError:
            # 如果失败，尝试以字典方式访问
            video_url = config.get("video_url", "")
            language = config.get("language", "auto")
            include_timestamps = config.get("include_timestamps", False)
            force_stt = config.get("force_stt", False)
        
        # 结果初始化
        result = {
            "transcript": "",
            "source": "",
            "video_title": "",
            "language": "",
            "error": None,
            "timing_stats": {}  # 添加时间统计字段
        }
        
        if not video_url:
            result["error"] = "请提供有效的视频URL"
            return result
        
        # 提取视频ID
        video_id = self.extract_video_id(video_url)
        if not video_id:
            result["error"] = "无效的YouTube URL格式"
            return result
        
        # 获取视频信息
        info_start_time = time.time()
        video_info = self.get_video_info(video_url)
        stage_times["get_video_info"] = time.time() - info_start_time
        
        if not video_info["success"]:
            result["error"] = video_info["error"]
            # 记录执行时间
            result["timing_stats"] = stage_times
            result["timing_stats"]["total_time"] = time.time() - start_time
            return result
        
        result["video_title"] = video_info["title"]
        
        # 首先尝试获取字幕 (除非强制使用STT)
        transcript_result = None
        if not force_stt:
            transcript_start_time = time.time()
            transcript_result = self.get_youtube_transcript(video_id, language)
            stage_times["get_transcript"] = time.time() - transcript_start_time
        
        # 如果成功获取字幕
        if transcript_result and transcript_result["success"]:
            format_start_time = time.time()
            result["transcript"] = self.format_transcript(
                transcript_result["transcript_data"], 
                include_timestamps
            )
            stage_times["format_transcript"] = time.time() - format_start_time
            
            result["source"] = "subtitles"
            result["language"] = transcript_result["language"]
            
            # 记录执行时间
            result["timing_stats"] = stage_times
            result["timing_stats"]["total_time"] = time.time() - start_time
            return result
        
        # 否则，使用语音识别
        logger.info("字幕不可用或被强制跳过，尝试使用语音识别...")
        
        # 下载音频
        download_start_time = time.time()
        audio_result = self.download_audio(video_info["yt_object"])
        stage_times["download_audio"] = time.time() - download_start_time
        
        if not audio_result["success"]:
            result["error"] = audio_result["error"]
            # 记录执行时间
            result["timing_stats"] = stage_times
            result["timing_stats"]["total_time"] = time.time() - start_time
            return result
        
        # 转写音频
        transcribe_start_time = time.time()
        stt_result = self.transcribe_audio(audio_result["audio_path"])
        stage_times["transcribe_audio"] = time.time() - transcribe_start_time
        
        if not stt_result["success"]:
            result["error"] = stt_result["error"]
            # 记录执行时间
            result["timing_stats"] = stage_times
            result["timing_stats"]["total_time"] = time.time() - start_time
            return result
        
        # 返回最终结果
        format_start_time = time.time()
        result["transcript"] = self.format_transcript(
            stt_result["transcript_data"], 
            include_timestamps
        )
        stage_times["format_transcript"] = time.time() - format_start_time
        
        result["source"] = "speech_recognition"
        result["language"] = stt_result["language"]
        
        # 清理临时文件 (如果需要)
        cleanup_start_time = time.time()
        try:
            if os.path.exists(audio_result["audio_path"]):
                os.remove(audio_result["audio_path"])
                logger.info(f"临时音频文件已删除: {audio_result['audio_path']}")
        except Exception as e:
            logger.warning(f"清理临时文件时出错: {str(e)}")
        stage_times["cleanup"] = time.time() - cleanup_start_time
        
        # 记录执行时间
        result["timing_stats"] = stage_times
        result["timing_stats"]["total_time"] = time.time() - start_time
        
        return result

# 测试代码
if __name__ == "__main__":
    widget = VideoTranscriptWidget()
    config = {
        "video_url": "https://www.youtube.com/watch?v=FHAvjCe86Ig",
        "language": "auto",
        "include_timestamps": True,
        "force_stt": False
    }
    
    try:
        from easydict import EasyDict
        config = EasyDict(config)
    except ImportError:
        print("注意: easydict 未安装，使用普通字典")
    
    print("正在处理视频:", config.get("video_url", config["video_url"]) if isinstance(config, dict) else config.video_url)
    
    # 开始整体计时
    total_start_time = time.time()
    output = widget({}, config)
    total_execution_time = time.time() - total_start_time
    
    # 打印结果
    print("输出结果:", output)
    
    # 打印时间统计信息
    print("\n===== 执行时间统计 =====")
    print(f"总执行时间: {total_execution_time:.2f} 秒")
    
    if "timing_stats" in output:
        for stage, duration in output["timing_stats"].items():
            if stage != "total_time":  # 单独处理总时间
                print(f"  - {stage}: {duration:.2f} 秒")
        
        # 显示内部总时间
        if "total_time" in output["timing_stats"]:
            print(f"内部计时总时间: {output['timing_stats']['total_time']:.2f} 秒") 