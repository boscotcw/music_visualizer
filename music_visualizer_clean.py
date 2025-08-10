"""
Clean Music Visualizer - Main Application
Modular design with separated concerns
"""
import pydub
import pygame
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
import numpy as np
import pyaudio
import struct
import math
import colorsys
import threading
import wave
import os
import cv2
import datetime
# from pydub import AudioSegment
# AudioSegment.converter = "C:/Program Files/FFMPEG/bin/ffmpeg.exe"
# AudioSegment.ffmpeg = "C:/Program Files/FFMPEG/bin/ffmpeg.exe"
# AudioSegment.ffprobe = "C:/Program Files/FFMPEG/bin/ffprobe.exe"
import tkinter as tk
from tkinter import filedialog
import time
import tempfile


# Import our modular components
from audio_loader import AudioLoader
from visualization import VisualizationRenderer

class MusicVisualizerClean:
    def __init__(self, width=1200, height=1000):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Music Visualizer - Clean Version")
        
        # Initialize modular components
        self.audio_loader = AudioLoader(self)
        self.renderer = VisualizationRenderer(self)
        
        # Audio settings
        self.chunk = 2048
        self.sample_rate = 44100
        self.channels = 1
        
        # Colors
        self.bg_color = (0, 0, 0)
        
        # Audio data
        self.audio_data = np.zeros(self.chunk)
        
        # Animation
        self.clock = pygame.time.Clock()
        
        # Audio stream
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Video recording
        self.is_recording = False
        self.video_writer = None
        self.video_filename = None
        self.frame_count = 0
        
        # Audio file playback
        self.audio_file = None
        self.audio_data_file = None
        self.file_position = 0
        self.is_playing_file = False
        self.file_sample_rate = 44100
        self.pygame_mixer_initialized = False
        
        # Audio processing settings
        self.gain_multiplier = 1.0  # Adjustable gain/sensitivity
        slider = Slider(self.screen, 100, 100, 800, 40, min=0, max=100, step=1)
        self.gain_multiplier_constant = 1e-4
        
        # Frequency range settings
        self.min_freq = 20
        self.max_freq = 20000
        
        # Frequency distribution mode
        self.freq_distribution_mode = "logarithmic"  # "logarithmic", "linear", "mel", "hybrid"
        
        # Text input system
        self.active_textbox = None  # 'min' or 'max' or None
        self.min_freq_text = str(self.min_freq)
        self.max_freq_text = str(self.max_freq)
        self.font = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
    def check_ffmpeg_availability(self):
        """Check if FFmpeg is available on the system"""
        try:
            import subprocess
            # Try to run ffmpeg -version
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                print(f"✅ FFmpeg found: {version_line}")
                return True
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"❌ FFmpeg not found or not working: {e}")
            
        # Try alternative locations
        alternative_paths = [
            'ffmpeg.exe',
            r'C:\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
            r'C:\ProgramData\chocolatey\bin\ffmpeg.exe'
        ]
        
        for path in alternative_paths:
            try:
                result = subprocess.run([path, '-version'], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=5)
                if result.returncode == 0:
                    print(f"✅ FFmpeg found at: {path}")
                    return True
            except:
                continue
                
        return False
    
    def diagnose_audio_file(self, file_path):
        """Diagnose why an audio file might not be loading"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        print(f"\n🔍 DIAGNOSING FILE: {os.path.basename(file_path)}")
        print(f"Extension: {file_extension}")
        print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
        
        # Check file header to determine actual format
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)
                
            print(f"File header: {header[:12].hex()}")
                
            if header.startswith(b'RIFF') and b'WAVE' in header:
                print("✅ Valid WAV file detected")
                return "wav"
            if header.startswith(b'ID3') or header[0:2] == b'\xff\xfb' or header[0:2] == b'\xff\xfa':
                print("🎵 MP3 file detected")
                
                # Additional MP3 diagnostics
                if header.startswith(b'ID3'):
                    id3_version = f"v2.{header[3]}.{header[4]}"
                    print(f"   📋 ID3 tag version: {id3_version}")
                    
                    # Try to find the actual audio start
                    try:
                        with open(file_path, 'rb') as f:
                            data = f.read(1024)
                            for i in range(len(data) - 1):
                                if data[i:i+2] in [b'\xff\xfb', b'\xff\xfa', b'\xff\xf3', b'\xff\xf2']:
                                    frame_header = data[i:i+4]
                                    # Parse MP3 frame header
                                    if len(frame_header) >= 4:
                                        sync = (frame_header[0] << 4) | (frame_header[1] >> 4)
                                        version = (frame_header[1] >> 3) & 0x03
                                        layer = (frame_header[1] >> 1) & 0x03
                                        bitrate_index = (frame_header[2] >> 4) & 0x0F
                                        sample_rate_index = (frame_header[2] >> 2) & 0x03
                                        
                                        versions = {1: "MPEG-2.5", 2: "Reserved", 3: "MPEG-2", 0: "MPEG-1"}
                                        layers = {1: "Layer III", 2: "Layer II", 3: "Layer I", 0: "Reserved"}
                                        
                                        print(f"   🎵 MPEG version: {versions.get(version, 'Unknown')}")
                                        print(f"   🎵 Layer: {layers.get(layer, 'Unknown')}")
                                        print(f"   🎵 Bitrate index: {bitrate_index}")
                                        print(f"   🎵 Sample rate index: {sample_rate_index}")
                                    break
                    except Exception as mp3_parse_error:
                        print(f"   ⚠️ Could not parse MP3 details: {mp3_parse_error}")
                
                return "mp3"
            if header[4:8] == b'ftyp':
                print("🎵 MP4/M4A file detected (requires FFmpeg)")
                return "m4a"
            if header.startswith(b'fLaC'):
                print("🎵 FLAC file detected (requires FFmpeg)")
                return "flac"
            if header.startswith(b'OggS'):
                print("🎵 OGG file detected (requires FFmpeg)")
                return "ogg"
            
            print(f"❓ Unknown format. Header: {header.hex()}")
            
            # Try to detect if it's actually an MP3 without proper header
            try:
                with open(file_path, 'rb') as f:
                    first_kb = f.read(1024)
                    if b'\xff\xfb' in first_kb or b'\xff\xfa' in first_kb:
                        print("🎵 Possible MP3 file (frame sync found)")
                        return "mp3"
            except:
                pass
                
            return "unknown"
                
        except Exception as e:
            print(f"❌ Cannot read file header: {e}")
            return "error"
    
    def load_audio_file(self):
        """Load an audio file for visualization - delegates to AudioLoader"""
        return self.audio_loader.load_audio_file()
    
    def start_video_recording(self):
        """Start recording the visualization as MP4 video"""
        if not self.is_recording:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_filename = f"music_visualizer_{timestamp}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.video_filename, fourcc, 60.0, (self.width, self.height)
            )
            
            self.is_recording = True
            self.frame_count = 0
            print(f"Started recording video: {self.video_filename}")
    
    def stop_video_recording(self):
        """Stop recording the video"""
        if self.is_recording and self.video_writer:
            self.video_writer.release()
            self.is_recording = False
            print(f"Video saved: {self.video_filename} ({self.frame_count} frames)")
            self.video_writer = None
            self.video_filename = None
            self.frame_count = 0
    
    def _create_temp_audio_file(self):
        """Create a temporary audio file with unique name"""
        import uuid
        
        # Create a unique temporary file name
        temp_dir = tempfile.gettempdir()
        unique_id = str(uuid.uuid4())[:8]
        temp_filename = f"music_visualizer_{unique_id}.wav"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        return temp_path
    
    def start_file_playback(self, filepath=None):
        """Start playing the loaded audio file"""
        if not self.audio_file:
            print("No audio file loaded")
            return
            
        if self.is_playing_file:
            print("Audio is already playing")
            return
            
        try:
            # Stop any existing music and unload it
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            
            # Create a unique temporary file
            temp_wav = self._create_temp_audio_file()
            
            # Export audio to temp file
            self.audio_file.export(temp_wav, format="wav")
            
            if not os.path.exists(temp_wav):
                print("Error: Failed to create temporary audio file")
                return
            
            pygame.mixer.music.load(temp_wav)
            pygame.mixer.music.play()
            
            self.is_playing_file = True
            self.file_position = 0
            self.current_temp_file = temp_wav  # Store reference for cleanup
            
            print("Started playing audio file")
            
            def cleanup_temp_file():
                # Wait for music to finish or be stopped
                while pygame.mixer.music.get_busy() and self.is_playing_file:
                    time.sleep(0.5)
                
                # Additional wait to ensure file is released
                time.sleep(1.0)
                
                try:
                    if os.path.exists(temp_wav):
                        # Make sure pygame has released the file
                        pygame.mixer.music.unload()
                        time.sleep(0.5)
                        os.remove(temp_wav)
                        print(f"Temporary audio file cleaned up: {os.path.basename(temp_wav)}")
                except Exception as e:
                    print(f"Could not remove temp file: {e}")
            
            cleanup_thread = threading.Thread(target=cleanup_temp_file)
            cleanup_thread.daemon = True
            cleanup_thread.start()
            
        except Exception as e:
            print(f"Error starting file playback: {e}")
    
    def stop_file_playback(self):
        """Stop playing the audio file"""
        if self.is_playing_file:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()  # Properly unload to release file handle
            self.is_playing_file = False
            self.file_position = 0
            print("Stopped audio file playback")
            
            # Clean up temp file immediately when manually stopped
            if hasattr(self, 'current_temp_file') and os.path.exists(self.current_temp_file):
                try:
                    time.sleep(0.5)  # Brief wait for file handle to be released
                    os.remove(self.current_temp_file)
                    print(f"Immediate cleanup of temp file: {os.path.basename(self.current_temp_file)}")
                except Exception as e:
                    print(f"Could not immediately clean temp file: {e}")
    
    def get_file_audio_data(self):
        """Get current audio data from file for visualization"""
        if not self.audio_data_file is None and self.is_playing_file:
            current_time_ms = pygame.mixer.music.get_pos()
            if current_time_ms == -1:
                self.is_playing_file = False
                return np.zeros(self.chunk)
            
            samples_per_ms = self.file_sample_rate / 1000
            sample_position = int(current_time_ms * samples_per_ms)
            
            start_pos = sample_position
            end_pos = start_pos + self.chunk
            
            if end_pos > len(self.audio_data_file):
                self.is_playing_file = False
                return np.zeros(self.chunk)
            
            return self.audio_data_file[start_pos:end_pos]
        
        return np.zeros(self.chunk)

    def capture_frame(self):
        """Capture current pygame screen and add to video"""
        if self.is_recording and self.video_writer:
            frame = pygame.surfarray.array3d(self.screen)
            frame = frame.swapaxes(0, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame)
            self.frame_count += 1

    def start_microphone_input(self):
        """Start capturing audio from microphone"""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk,
                stream_callback=self.audio_callback
            )
            self.stream.start_stream()
            print("Microphone input started. Speak or play music near your microphone!")
        except Exception as e:
            print(f"Error starting microphone: {e}")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        if in_data:
            audio_array = struct.unpack(f'{frame_count}h', in_data)
            self.audio_data = np.array(audio_array, dtype=np.float32)
        return (None, pyaudio.paContinue)
    
    def create_frequency_bins(self, num_bars, min_freq, max_freq, mode="logarithmic"):
        """Create frequency bin edges based on different distribution modes"""
        if mode == "linear":
            # Linear distribution - equal Hz spacing
            return np.linspace(min_freq, max_freq, num_bars + 1)
            
        elif mode == "logarithmic":
            # Logarithmic distribution - equal ratio spacing
            log_min = np.log10(min_freq)
            log_max = np.log10(max_freq)
            return np.logspace(log_min, log_max, num_bars + 1, base=10)
            
        elif mode == "mel":
            # Mel scale - perceptually uniform
            def hz_to_mel(hz):
                return 2595 * np.log10(1 + hz / 700)
            
            def mel_to_hz(mel):
                return 700 * (10**(mel / 2595) - 1)
            
            mel_min = hz_to_mel(min_freq)
            mel_max = hz_to_mel(max_freq)
            mel_points = np.linspace(mel_min, mel_max, num_bars + 1)
            return np.array([mel_to_hz(mel) for mel in mel_points])
            
        elif mode == "hybrid":
            # Hybrid: Linear for low frequencies, log for high frequencies
            crossover_freq = 1000  # Hz
            
            if max_freq <= crossover_freq:
                return np.linspace(min_freq, max_freq, num_bars + 1)
            elif min_freq >= crossover_freq:
                log_min = np.log10(min_freq)
                log_max = np.log10(max_freq)
                return np.logspace(log_min, log_max, num_bars + 1, base=10)
            else:
                # Split bars between linear and log sections
                linear_bars = int(num_bars * 0.4)  # 40% for linear section
                log_bars = num_bars - linear_bars   # 60% for log section
                
                linear_edges = np.linspace(min_freq, crossover_freq, linear_bars + 1)
                log_edges = np.logspace(np.log10(crossover_freq), np.log10(max_freq), log_bars + 1, base=10)[1:]  # Skip first point to avoid duplication
                
                return np.concatenate([linear_edges, log_edges])
        
        # Default to logarithmic
        log_min = np.log10(min_freq)
        log_max = np.log10(max_freq)
        return np.logspace(log_min, log_max, num_bars + 1, base=10)
    
    def process_audio(self):
        """Process audio data and extract frequencies with enhanced resolution"""
        if self.is_playing_file:
            current_audio_data = self.get_file_audio_data()
        else:
            current_audio_data = self.audio_data
        
        if len(current_audio_data) == 0:
            return
        
        # Zero-pad for better frequency resolution if needed
        padded_length = max(len(current_audio_data), 16384)  # Minimum 16k for good resolution
        padded_data = np.zeros(padded_length)
        padded_data[:len(current_audio_data)] = current_audio_data
        
        # Apply windowing to reduce spectral leakage
        window = np.hanning(len(padded_data))
        windowed_data = padded_data * window
        
        # Apply FFT to get frequency domain
        fft = np.fft.fft(windowed_data)
        magnitude = np.abs(fft)
        
        # Take only the first half (positive frequencies)
        magnitude = magnitude[:len(magnitude)//2]
        
        # Create frequency bins for FFT
        freq_bins = np.fft.fftfreq(len(padded_data), 1 / self.sample_rate)
        freq_bins = freq_bins[:len(magnitude)]
        
        # Create frequency edges based on selected distribution mode
        frequency_edges = self.create_frequency_bins(
            self.renderer.bars, self.min_freq, self.max_freq, self.freq_distribution_mode
        )
        
        # Create frequency array to pass to renderer
        frequencies = np.zeros(self.renderer.bars)
        
        for i in range(self.renderer.bars):
            start_freq = frequency_edges[i]
            end_freq = frequency_edges[i + 1]
            
            # Find indices in FFT corresponding to this frequency range
            idx = np.where((freq_bins >= start_freq) & (freq_bins < end_freq))[0]
            
            if len(idx) > 0:
                # Use RMS for better dynamic response
                avg_magnitude = np.sqrt(np.mean(magnitude[idx] ** 2))
                
                # Dynamic scaling based on frequency
                freq_center = (start_freq + end_freq) / 2
                scale_factor = int(30000 / np.power(np.log(freq_center + 1), 4))
                
                normalized_magnitude = (avg_magnitude / scale_factor) * self.gain_multiplier * self.gain_multiplier_constant
                frequencies[i] = min(np.log1p(normalized_magnitude * 10) / np.log1p(10), 1.0)
            else:
                frequencies[i] = 0.0
        
        # Update renderer with new frequency data
        self.renderer.update_frequencies(frequencies)
    
    def handle_text_input(self, event):
        """Handle text input for frequency textboxes"""
        if self.active_textbox == 'min':
            if event.key == pygame.K_BACKSPACE:
                self.min_freq_text = self.min_freq_text[:-1]
            elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                try:
                    new_freq = float(self.min_freq_text)
                    if 20 <= new_freq < self.max_freq:
                        self.min_freq = new_freq
                        print(f"Min frequency set to {self.min_freq} Hz")
                    else:
                        print(f"Invalid min frequency. Must be between 20 and {self.max_freq}")
                        self.min_freq_text = str(self.min_freq)
                except ValueError:
                    print("Invalid number format")
                    self.min_freq_text = str(self.min_freq)
                self.active_textbox = None
            elif event.unicode.isdigit() or event.unicode == '.':
                self.min_freq_text += event.unicode
                
        elif self.active_textbox == 'max':
            if event.key == pygame.K_BACKSPACE:
                self.max_freq_text = self.max_freq_text[:-1]
            elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                try:
                    new_freq = float(self.max_freq_text)
                    if self.min_freq < new_freq <= 22050:
                        self.max_freq = new_freq
                        print(f"Max frequency set to {self.max_freq} Hz")
                    else:
                        print(f"Invalid max frequency. Must be between {self.min_freq} and 22050")
                        self.max_freq_text = str(self.max_freq)
                except ValueError:
                    print("Invalid number format")
                    self.max_freq_text = str(self.max_freq)
                self.active_textbox = None
            elif event.unicode.isdigit() or event.unicode == '.':
                self.max_freq_text += event.unicode

    def handle_textbox_click(self, mouse_pos):
        """Handle mouse clicks on textboxes"""
        min_freq_rect = pygame.Rect(150, 150, 100, 25)
        max_freq_rect = pygame.Rect(150, 180, 100, 25)
        
        if min_freq_rect.collidepoint(mouse_pos):
            self.active_textbox = 'min'
            self.min_freq_text = ""
        elif max_freq_rect.collidepoint(mouse_pos):
            self.active_textbox = 'max'
            self.max_freq_text = ""
        else:
            self.active_textbox = None
    
    def run(self):
        """Main game loop"""
        self.start_microphone_input()
        running = True
        visualization_mode = 0  # 0: bars, 1: circle
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.handle_textbox_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if self.active_textbox:
                        self.handle_text_input(event)
                    else:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_SPACE:
                            visualization_mode = (visualization_mode + 1) % 2
                        elif event.key == pygame.K_r:
                            if self.is_recording:
                                self.stop_video_recording()
                            else:
                                self.start_video_recording()
                        elif event.key == pygame.K_l:
                            self.load_audio_file()
                        elif event.key == pygame.K_p:
                            if self.audio_file:
                                if self.is_playing_file:
                                    self.stop_file_playback()
                                else:
                                    self.start_file_playback()
                        elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                            self.gain_multiplier = min(self.gain_multiplier + 0.1, 5.0)
                            print(f"Gain increased to {self.gain_multiplier:.1f}x")
                        elif event.key == pygame.K_MINUS:
                            self.gain_multiplier = max(self.gain_multiplier - 0.1, 0.1)
                            print(f"Gain decreased to {self.gain_multiplier:.1f}x")
                        elif event.key == pygame.K_f:
                            # Cycle through frequency distribution modes
                            modes = ["logarithmic", "linear", "mel", "hybrid"]
                            current_index = modes.index(self.freq_distribution_mode)
                            self.freq_distribution_mode = modes[(current_index + 1) % len(modes)]
                            print(f"Frequency distribution mode: {self.freq_distribution_mode}")
                        elif event.key == pygame.K_b:
                            # Increase number of bars for more detail
                            self.renderer.bars = min(self.renderer.bars + 32, 512)
                            self.renderer.frequencies = np.zeros(self.renderer.bars)
                            self.renderer.smoothed_frequencies = np.zeros(self.renderer.bars)
                            self.renderer.generate_colors()
                            print(f"Bars increased to {self.renderer.bars}")
                        elif event.key == pygame.K_v:
                            # Decrease number of bars
                            self.renderer.bars = max(self.renderer.bars - 32, 64)
                            self.renderer.frequencies = np.zeros(self.renderer.bars)
                            self.renderer.smoothed_frequencies = np.zeros(self.renderer.bars)
                            self.renderer.generate_colors()
                            print(f"Bars decreased to {self.renderer.bars}")
            
            self.process_audio()
            
            # Update renderer animation time
            self.renderer.update_time(0.1)
            
            self.screen.fill(self.bg_color)
            
            if visualization_mode == 0:
                self.renderer.draw_frequency_bars()
            else:
                self.renderer.draw_circle_visualizer()
            
            self.renderer.draw_waveform()
            self.renderer.draw_info()
            
            if self.is_recording:
                self.capture_frame()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        # Cleanup
        if self.is_recording:
            self.stop_video_recording()
        if self.is_playing_file:
            self.stop_file_playback()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        if self.pygame_mixer_initialized:
            pygame.mixer.quit()
        
        # Clean up any remaining temp files (including unique ones)
        temp_dir = tempfile.gettempdir()
        try:
            for filename in os.listdir(temp_dir):
                if filename.startswith("music_visualizer_") and filename.endswith(".wav"):
                    temp_file_path = os.path.join(temp_dir, filename)
                    try:
                        os.remove(temp_file_path)
                        print(f"Cleaned up temp file: {filename}")
                    except Exception as e:
                        print(f"Could not remove temp file {filename}: {e}")
        except Exception as e:
            print(f"Error during temp file cleanup: {e}")
        
        pygame.quit()

def main():
    """Main function to run the clean visualizer"""
    try:
        visualizer = MusicVisualizerClean()
        visualizer.run()
    except Exception as e:
        print(f"Error running visualizer: {e}")
        print("Make sure you have the required packages installed:")
        print("pip install pygame numpy pyaudio")

if __name__ == "__main__":
    main()
