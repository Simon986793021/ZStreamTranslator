import argparse
import os
import signal
import sys
import subprocess
import threading
from datetime import datetime

import ffmpeg
import numpy as np
from whisper.audio import SAMPLE_RATE
import whisper
from vad import VAD


class RingBuffer:

    def __init__(self, size):
        self.size = size
        self.data = []
        self.full = False
        self.cur = 0

    def append(self, x):
        if self.size <= 0:
            return
        if self.full:
            self.data[self.cur] = x
            self.cur = (self.cur + 1) % self.size
        else:
            self.data.append(x)
            if len(self.data) == self.size:
                self.full = True

    def get_all(self):
        """ Get all elements in chronological order from oldest to newest. """
        all_data = []
        for i in range(len(self.data)):
            idx = (i + self.cur) % self.size
            all_data.append(self.data[idx])
        return all_data

    def has_repetition(self):
        prev = None
        for elem in self.data:
            if elem == prev:
                return True
            prev = elem
        return False

    def clear(self):
        self.data = []
        self.full = False
        self.cur = 0

def main(url, **decode_options):
    print ("zego translator start")
   
    model = whisper.load_model("small")

    parser = argparse.ArgumentParser(description="Parameters for translator.py")
    parser.add_argument('URL',type=str,help='Stream website and channel name, e.g. twitch.tv/forsen')
    args = parser.parse_args().__dict__
    url = args.pop("URL")
    frame_duration = 0.1
    continuous_no_speech_threshold = 0.8
    prefix_retention_length=0.8
    min_audio_length = 3.0
    max_audio_length = 30.0
    vad_threshold = 0.5 
    history_audio_buffer = RingBuffer(1)
    history_text_buffer = RingBuffer(0)
    n_bytes = round(frame_duration * SAMPLE_RATE *2) 
    #初始化流切片
    stream_slicer = StreamSlicer(frame_duration=frame_duration,
                                 continuous_no_speech_threshold=continuous_no_speech_threshold,
                                 min_audio_length=min_audio_length,
                                 max_audio_length=max_audio_length,
                                 prefix_retention_length=prefix_retention_length,
                                 vad_threshold=vad_threshold,
                                 sampling_rate=SAMPLE_RATE)
    print("open stream")
    ffmpeg_process, ytdlp_process = open_stream(url)

    def handler(signum, frame):
        ffmpeg_process.kill()
        if ytdlp_process:
            ytdlp_process.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)

    while ffmpeg_process.poll() is None:
        # Read audio from ffmpeg stream
        in_bytes = ffmpeg_process.stdout.read(n_bytes)
        if not in_bytes:
            break
        #将int16类型的数据转为float32类型的数据，并进行标准化，将数据缩放至 [-1.0, 1.0] 之间
        audio = np.frombuffer(in_bytes, np.int16).flatten().astype(np.float32) / 32768.0
        stream_slicer.put(audio)
        if stream_slicer.should_slice():
            # Decode the audio
            sliced_audio ,time_range= stream_slicer.slice()
            history_audio_buffer.append(sliced_audio)
            result = model.transcribe(np.concatenate(history_audio_buffer.get_all()),
                                          prefix="".join(history_text_buffer.get_all()),
                                          language="Chinese",
                                          without_timestamps=True,
                                          **decode_options)
            decoded_text = result.get("text")
            print(decoded_text)

def open_stream(stream):
    print(stream)
    def writer(ytdlp_proc, ffmpeg_proc):
        while (ytdlp_proc.poll() is None) and (ffmpeg_proc.poll() is None):
            try:
                chunk = ytdlp_proc.stdout.read(1024)
                ffmpeg_proc.stdin.write(chunk)
            except (BrokenPipeError, OSError):
                pass
        ytdlp_proc.kill()
        ffmpeg_proc.kill()

    cmd = ['yt-dlp', stream, '-f', "wa*", '-o', '-', '-q']
    ytdlp_process = subprocess.Popen(cmd, stdout=subprocess.PIPE)   

    try:
        ffmpeg_process = (ffmpeg.input("pipe:", loglevel="panic").output("pipe:",
                                                                         format="s16le",
                                                                         acodec="pcm_s16le",
                                                                         ac=1,
                                                                         ar=SAMPLE_RATE).run_async(
                                                                             pipe_stdin=True,
                                                                             pipe_stdout=True))
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    thread = threading.Thread(target=writer, args=(ytdlp_process, ffmpeg_process))
    thread.start()
    return ffmpeg_process, ytdlp_process

class StreamSlicer:

    def __init__(self, frame_duration, continuous_no_speech_threshold, min_audio_length,
                 max_audio_length, prefix_retention_length, vad_threshold, sampling_rate):
        self.vad = VAD()
        self.continuous_no_speech_threshold = round(continuous_no_speech_threshold / frame_duration)
        self.min_audio_length = round(min_audio_length / frame_duration)
        self.max_audio_length = round(max_audio_length / frame_duration)
        self.prefix_retention_length = round(prefix_retention_length / frame_duration)
        self.vad_threshold = vad_threshold
        self.sampling_rate = sampling_rate
        self.audio_buffer = []
        self.prefix_audio_buffer = []
        self.speech_count = 0
        self.no_speech_count = 0
        self.continuous_no_speech_count = 0
        self.frame_duration = frame_duration
        self.counter = 0
        self.last_slice_second = 0.0

    def put(self, audio):
        self.counter += 1
        if self.vad.is_speech(audio, self.vad_threshold, self.sampling_rate):
            self.audio_buffer.append(audio)
            self.speech_count += 1
            self.continuous_no_speech_count = 0
        else:
            if self.speech_count == 0 and self.no_speech_count == 1:
                self.slice()
            self.audio_buffer.append(audio)
            self.no_speech_count += 1
            self.continuous_no_speech_count += 1
        if self.speech_count and self.no_speech_count / 4 > self.speech_count:
            self.slice()

    def should_slice(self):
        audio_len = len(self.audio_buffer)
        if audio_len < self.min_audio_length:
            return False
        if audio_len > self.max_audio_length:
            return True
        if self.continuous_no_speech_count >= self.continuous_no_speech_threshold:
            return True
        return False

    def slice(self):
        concatenate_buffer = self.prefix_audio_buffer + self.audio_buffer
        concatenate_audio = np.concatenate(concatenate_buffer)
        self.audio_buffer = []
        self.prefix_audio_buffer = concatenate_buffer[-self.prefix_retention_length:]
        self.speech_count = 0
        self.no_speech_count = 0
        self.continuous_no_speech_count = 0
        # self.vad.reset_states()
        slice_second = self.counter * self.frame_duration
        last_slice_second = self.last_slice_second
        self.last_slice_second = slice_second
        return concatenate_audio, (last_slice_second, slice_second)


def cli():
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"
    parser = argparse.ArgumentParser(description="Parameters for translator.py")
    parser.add_argument('URL',
                        type=str,
                        help='set the stream url')
   

    args = parser.parse_args().__dict__
    url = args.pop("URL")
    # Remove yt-dlp cache
    for file in os.listdir('./'):
        if file.startswith('--Frag'):
            os.remove(file)

    main(url, **args)

if __name__ == '__main__':
    cli()
    

