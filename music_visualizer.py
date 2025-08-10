from re import match
from turtle import width
import pydub
import pygame
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
from pydub import AudioSegment
AudioSegment.converter = "C:/Program Files/FFMPEG/bin/ffmpeg.exe"
AudioSegment.ffmpeg = "C:/Program Files/FFMPEG/bin/ffmpeg.exe"
AudioSegment.ffprobe = "C:/Program Files/FFMPEG/bin/ffprobe.exe"
from pydub.playback import play
import tkinter as tk
from tkinter import filedialog
import time

class MusicVisualizer:
    def __init__(self, width=1200, height=1000):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Music Visualizer")
        
        # Audio settings
        self.chunk = 4096
        self.sample_rate = 44100
        self.channels = 1
        
        # Visualization settings
        self.bars = 256
        self.bar_layers = 3  # Number of layers for bar visualization
        self.bar_width = min(width * self.bar_layers / self.bars, 1)
        self.max_height = height - 100
        
        # Colors
        self.bg_color = (0, 0, 0)
        self.bar_colors = []
        self.generate_colors()
        
        # Audio data
        self.audio_data = np.zeros(self.chunk)
        self.frequencies = np.zeros(self.bars)
        self.smoothed_frequencies = np.zeros(self.bars)
        
        # Animation
        self.time = 0
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
        self.gain_multiplier_constant = 1e-3
        
        # Frequency range settings
        self.min_freq = 20
        self.max_freq = 20000
        
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
            elif header.startswith(b'ID3') or header[0:2] == b'\xff\xfb' or header[0:2] == b'\xff\xfa':
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
            elif header[4:8] == b'ftyp':
                print("🎵 MP4/M4A file detected (requires FFmpeg)")
                return "m4a"
            elif header.startswith(b'fLaC'):
                print("🎵 FLAC file detected (requires FFmpeg)")
                return "flac"
            elif header.startswith(b'OggS'):
                print("🎵 OGG file detected (requires FFmpeg)")
                return "ogg"
            else:
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
        
    def start_video_recording(self):
        """Start recording the visualization as MP4 video"""
        if not self.is_recording:
            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_filename = f"music_visualizer_{timestamp}.mp4"
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.video_filename, 
                fourcc, 
                60.0,  # FPS
                (self.width, self.height)
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
    

    def load_audio_file(self, file_path, file_extension):
        try:
            match file_extension:
                case '.mp3':
                    print("Attempting MP3 loading with multiple methods...")

                    # Method 1: Standard pydub MP3 loading
                    try:
                        audio = AudioSegment.from_mp3(file_path)
                        print("✅ MP3 loaded with standard method")
                        return audio
                    except Exception as mp3_error:
                        print(f"Standard MP3 loading failed: {mp3_error}")
                    
                    # Method 2: Force format specification
                    try:
                        audio = AudioSegment.from_file(file_path, format="mp3")
                        print("✅ MP3 loaded with explicit format")
                        return audio
                    except Exception as mp3_error:
                        print(f"Explicit format loading failed: {mp3_error}")
                        
                    # Method 3: Try without format (let ffmpeg auto-detect)
                    try:
                        audio = AudioSegment.from_file(file_path)
                        print("✅ MP3 loaded with auto-detection")
                        return audio
                    except Exception as mp3_error:
                        print(f"Auto-detection failed: {mp3_error}")
                            
                    # Method 4: Force parameters using pydub
                    try:
                        audio = AudioSegment.from_file(
                            file_path,
                            format="mp3",
                            parameters=["-ac", "1", "-ar", "44100"]
                        )
                        print("✅ MP3 loaded with forced parameters")
                        return audio
                    except Exception as mp3_error:
                        print(f"Forced parameters failed: {mp3_error}")
                        
                    # Method 5: Try direct ffmpeg command
                    try:
                        import subprocess
                        import tempfile
                        
                        print("Trying direct FFmpeg conversion...")
                        
                        # Create temporary WAV file
                        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                        temp_wav.close()
                        
                        # Run ffmpeg directly
                        ffmpeg_cmd = [
                            "C:/Program Files/FFMPEG/bin/ffmpeg.exe",
                            "-i", file_path,
                            "-ac", "1",  # mono
                            "-ar", "44100",  # sample rate
                            "-y",  # overwrite
                            temp_wav.name
                        ]
                        
                        result = subprocess.run(ffmpeg_cmd, 
                                                capture_output=True, 
                                                text=True, 
                                                timeout=30)
                        
                        if result.returncode == 0:
                            # Load the converted WAV file
                            audio = AudioSegment.from_wav(temp_wav.name)
                            print("✅ MP3 loaded via direct FFmpeg conversion")
                            os.unlink(temp_wav.name)        # Clean up temp file
                            return audio
                        
                        os.unlink(temp_wav.name)        # Clean up temp file
                        raise Exception("Direct FFmpeg conversion failed: " + result.stderr)
                            
                    except Exception as mp3_error:
                        print(f"Direct FFmpeg conversion failed: {mp3_error}")
                        raise Exception("All MP3 loading methods failed")
                                
                case '.wav':
                    # For WAV files, try the native wave library first
                    try:
                        import wave
                        import struct
                        with wave.open(file_path, 'rb') as wav_file:
                            frames = wav_file.readframes(-1)
                            sample_rate = wav_file.getframerate()
                            channels = wav_file.getnchannels()
                            sample_width = wav_file.getsampwidth()

                        # Create AudioSegment from raw data
                        audio = AudioSegment(
                            data=frames,
                            sample_width=sample_width,
                            frame_rate=sample_rate,
                            channels=channels
                        )
                        print("Successfully loaded WAV using native wave library")
                    except Exception as wav_error:
                        print(f"Native wave library failed: {wav_error}, trying pydub...")
                        audio = AudioSegment.from_wav(file_path)
                case '.m4a':
                    audio = AudioSegment.from_file(file_path, format="m4a")
                case '.flac':
                    audio = AudioSegment.from_file(file_path, format="flac")
                case '.ogg':
                    audio = AudioSegment.from_ogg(file_path)
                case _:
                    # Fallback to auto-detection
                    print("Unknown file type, trying auto-detection...")
                    audio = AudioSegment.from_file(file_path)
            
            print(f"Successfully loaded audio file")
            
        except Exception as codec_error:
            print(f"Error with codec/format: {codec_error}")
            print("Trying alternative loading method...")
            
            # Try using alternative method with different parameters
            try:
                audio = AudioSegment.from_file(file_path, format="auto")
            except Exception as alt_error:
                print(f"Alternative method failed: {alt_error}")
                
                # Last resort: try with wave library for WAV files
                if file_extension == '.wav':
                    try:
                        import wave
                        import struct
                        
                        print("Trying direct WAV loading without pydub...")
                        
                        # Load WAV file directly
                        with wave.open(file_path, 'rb') as wav_file:
                            frames = wav_file.readframes(-1)
                            sample_rate = wav_file.getframerate()
                            channels = wav_file.getnchannels()
                            sample_width = wav_file.getsampwidth()
                        
                        # Convert to numpy array directly (skip pydub)
                        if sample_width == 1:  # 8-bit
                            audio_data = np.frombuffer(frames, dtype=np.uint8)
                            audio_data = (audio_data.astype(np.float32) - 128) / 128.0
                        elif sample_width == 2:  # 16-bit
                            audio_data = np.frombuffer(frames, dtype=np.int16)
                            audio_data = audio_data.astype(np.float32) / 32768.0
                        elif sample_width == 3:  # 24-bit
                            audio_data = np.frombuffer(frames, dtype=np.int8).reshape(-1, 3)
                            audio_data = ((audio_data[:, 0] << 16) | (audio_data[:, 1] << 8) | audio_data[:, 2]).astype(np.float32) / 8388608.0
                        elif sample_width == 4:  # 32-bit
                            audio_data = np.frombuffer(frames, dtype=np.int32)
                            audio_data = audio_data.astype(np.float32) / 2147483648.0
                        
                        # Handle stereo to mono conversion
                        if channels == 2:
                            audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                        
                        # Resample to 44100 if needed (simple decimation/interpolation)
                        if sample_rate != 44100:
                            from scipy import signal
                            try:
                                audio_data = signal.resample(audio_data, int(len(audio_data) * 44100 / sample_rate))
                            except ImportError:
                                print("Warning: scipy not available for resampling, using original sample rate")
                                self.file_sample_rate = sample_rate
                        
                        # Store the audio data directly
                        self.audio_data_file = audio_data * 16384  # Scale to appropriate level
                        self.file_sample_rate = 44100 if sample_rate != 44100 else sample_rate
                        
                        # Create a dummy audio object for compatibility
                        class DummyAudio:
                            def __init__(self, duration_ms):
                                self.duration_ms = duration_ms
                                self.frame_rate = 44100
                                self.channels = 1
                            
                            def __len__(self):
                                return self.duration_ms
                            
                            def export(self, filename, format):
                                # Export the audio data as WAV
                                import wave
                                with wave.open(filename, 'wb') as out_wav:
                                    out_wav.setnchannels(1)
                                    out_wav.setsampwidth(2)
                                    out_wav.setframerate(44100)
                                    out_wav.writeframes((self.audio_data_file * 32767).astype(np.int16).tobytes())
                        
                        duration_ms = len(audio_data) / 44100 * 1000
                        audio = DummyAudio(duration_ms)
                        audio.audio_data_file = audio_data
                        
                        print("Successfully loaded WAV using direct method (no pydub)")
                        
                        # Skip the pydub processing since we already have the data
                        self.audio_file = audio
                        self.file_position = 0
                        self.is_playing_file = False
                        
                        # Initialize pygame mixer for playback
                        if not self.pygame_mixer_initialized:
                            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=1024)
                            self.pygame_mixer_initialized = True
                        
                        print(f"Audio file loaded successfully!")
                        print(f"Duration: {duration_ms/1000:.2f} seconds")
                        print(f"Sample rate: {self.file_sample_rate} Hz")
                        print(f"Channels: 1 (mono)")
                        
                        return True
                        
                    except Exception as wav_error:
                        print(f"Direct WAV loading failed: {wav_error}")
                        raise Exception(f"Could not load audio file. Please ensure the file is a valid audio file.")
                else:
                    raise Exception(f"Could not load audio file. For non-WAV files, please install FFmpeg or convert to WAV format.")


    def load_audio_file(self):
        """Load an audio file for visualization"""
        # Hide the main window temporarily
        root = tk.Tk()
        root.withdraw()
        
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.mp3 *.wav *.m4a *.flac *.ogg"),
                ("MP3 Files", "*.mp3"),
                ("WAV Files", "*.wav"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path and os.path.exists(file_path):
            try:
                # Check if file exists and is readable
                if not os.path.isfile(file_path):
                    print(f"Error: File does not exist: {file_path}")
                    return False
                
                # Get file info for diagnostics
                file_size = os.path.getsize(file_path)
                print(f"Loading audio file: {file_path}")
                print(f"File size: {file_size / (1024*1024):.2f} MB")
                
                # Try to determine file format
                file_extension = os.path.splitext(file_path)[1].lower()
                print(f"File extension: {file_extension}")
                
                # Check if file is readable
                try:
                    with open(file_path, 'rb') as f:
                        first_bytes = f.read(16)
                        print(f"File header (first 16 bytes): {first_bytes.hex()}")
                except Exception as read_error:
                    print(f"Cannot read file: {read_error}")
                    return False
                
                # Check FFmpeg availability for non-WAV files
                ffmpeg_available = self.check_ffmpeg_availability()
                if not ffmpeg_available and file_extension != '.wav':
                    print(f"\n⚠️  WARNING: FFmpeg not found!")
                    print(f"File type {file_extension} requires FFmpeg for loading.")
                    print(f"Either install FFmpeg or convert your file to WAV format.")
                    print(f"Attempting to load anyway...\n")
                
                
                
                # Convert to mono and resample to 44100 Hz
                audio = audio.set_channels(1).set_frame_rate(44100)
                
                print(f"Audio file loaded: {file_path}")
                # Convert to numpy array
                self.audio_data_file = np.array(audio.get_array_of_samples(), dtype=np.float32)
                
                # Normalize
                if len(self.audio_data_file) > 0:
                    max_val = np.max(np.abs(self.audio_data_file))
                    if max_val > 0:
                        self.audio_data_file = self.audio_data_file / max_val * 16384  # Reduced from 32767
                
                self.audio_file = audio
                self.file_position = 0
                self.file_sample_rate = 44100
                self.is_playing_file = False
                
                # Initialize pygame mixer for playback
                if not self.pygame_mixer_initialized:
                    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=1024)
                    self.pygame_mixer_initialized = True
                
                print(f"Audio file loaded successfully!")
                print(f"Duration: {len(audio)/1000:.2f} seconds")
                print(f"Sample rate: {audio.frame_rate} Hz")
                print(f"Channels: {audio.channels}")
                
                return True
                
            except Exception as e:
                print(f"Error loading audio file: {e}")
                
                # Run diagnostics to help troubleshoot
                file_format = self.diagnose_audio_file(file_path)
                
                print("\n=== TROUBLESHOOTING GUIDE ===")
                
                if file_format == "wav":
                    print("🔧 WAV file troubleshooting:")
                    print("   - File might be corrupted")
                    print("   - Unsupported WAV encoding (try converting with a different tool)")
                    print("   - File might be too large")
                elif file_format in ["mp3", "m4a", "flac", "ogg"]:
                    print(f"🔧 {file_format.upper()} file troubleshooting:")
                    if not self.check_ffmpeg_availability():
                        print("   ❌ FFmpeg is required but not found!")
                        print("   📥 Install FFmpeg:")
                        print("      1. Download from https://ffmpeg.org/download.html")
                        print("      2. Add to system PATH")
                        print("      3. Or use: choco install ffmpeg (if you have Chocolatey)")
                        print("      4. Alternative: pip install pydub[mp3]")
                    else:
                        print("   ✅ FFmpeg is available")
                        if file_format == "mp3":
                            print("   🎵 MP3-specific issues:")
                            print("      - Variable bitrate (VBR) MP3s can be problematic")
                            print("      - Large ID3 tags might cause issues")
                            print("      - Corrupted or unusual encoding")
                            print("      - Try converting with: ffmpeg -i input.mp3 -acodec mp3 -ab 128k output.mp3")
                        print("   - File might be corrupted or use unsupported codec")
                        print("   - Try converting to WAV format first")
                else:
                    print("🔧 Unknown file format:")
                    print("   - File extension might not match actual format")
                    print("   - File might be corrupted")
                    print("   - Try converting to WAV format")
                
                print("\n💡 Quick solutions:")
                print("   1. Convert to WAV format (works without FFmpeg)")
                print("   2. Try a different audio file")
                print("   3. Use online converter: https://convertio.co/")
                print("   4. Use Audacity to convert and repair files")
                print("=============================\n")
                return False
        return False
    
    def start_file_playback(self, filepath=None):
        """Start playing the loaded audio file"""
        if self.audio_file and not self.is_playing_file:
            try:
                # Export to temporary WAV for pygame
                temp_wav = os.path.join(os.getcwd(), "temp_audio.wav")
                
                # Remove existing temp file if it exists
                if os.path.exists(temp_wav):
                    try:
                        os.remove(temp_wav)
                    except:
                        pass
                
                # Export audio to temp file
                self.audio_file.export(temp_wav, format="wav")
                
                # Verify temp file was created
                if not os.path.exists(temp_wav):
                    print("Error: Failed to create temporary audio file")
                    return
                
                # Load and play with pygame
                pygame.mixer.music.load(temp_wav)
                pygame.mixer.music.play()
                
                self.is_playing_file = True
                self.file_position = 0
                
                print("Started playing audio file")
                
                # Clean up temp file after a delay (increased delay)
                def cleanup_temp_file():
                    time.sleep(2.0)  # Wait longer before cleanup
                    try:
                        if os.path.exists(temp_wav):
                            os.remove(temp_wav)
                            print("Temporary audio file cleaned up")
                    except Exception as e:
                        print(f"Could not remove temp file: {e}")
                
                # Start cleanup in background thread
                cleanup_thread = threading.Thread(target=cleanup_temp_file)
                cleanup_thread.daemon = True
                cleanup_thread.start()
                
            except Exception as e:
                print(f"Error starting file playback: {e}")
                # Try to clean up temp file if it exists
                temp_wav = os.path.join(os.getcwd(), "temp_audio.wav")
                if os.path.exists(temp_wav):
                    try:
                        os.remove(temp_wav)
                    except:
                        pass
    
    def stop_file_playback(self):
        """Stop playing the audio file"""
        if self.is_playing_file:
            pygame.mixer.music.stop()
            self.is_playing_file = False
            self.file_position = 0
            print("Stopped audio file playback")
    
    def get_file_audio_data(self):
        """Get current audio data from file for visualization"""
        if not self.audio_data_file is None and self.is_playing_file:
            # Calculate current position based on time
            current_time_ms = pygame.mixer.music.get_pos()
            if current_time_ms == -1:  # Music stopped
                self.is_playing_file = False
                return np.zeros(self.chunk)
            
            # Convert to sample position
            samples_per_ms = self.file_sample_rate / 1000
            sample_position = int(current_time_ms * samples_per_ms)
            
            # Get chunk of data
            start_pos = sample_position
            end_pos = start_pos + self.chunk
            
            if end_pos > len(self.audio_data_file):
                # End of file reached
                self.is_playing_file = False
                return np.zeros(self.chunk)
            
            return self.audio_data_file[start_pos:end_pos]
        
        return np.zeros(self.chunk)

    def capture_frame(self):
        """Capture current pygame screen and add to video"""
        if self.is_recording and self.video_writer:
            # Get the pygame surface as a numpy array
            frame = pygame.surfarray.array3d(self.screen)
            # Transpose to get correct orientation (pygame uses different axis order)
            frame = frame.swapaxes(0, 1)
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Write frame to video
            self.video_writer.write(frame)
            self.frame_count += 1

    def generate_colors(self):
        """Generate a spectrum of colors for the bars"""
        self.bar_colors = []
        for i in range(self.bars):
            hue = i / self.bars
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            self.bar_colors.append(tuple(int(c * 255) for c in rgb))
    
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
            # Convert audio data to numpy array
            audio_array = struct.unpack(f'{frame_count}h', in_data)
            self.audio_data = np.array(audio_array, dtype=np.float32)
        return (None, pyaudio.paContinue)
    
    def process_audio(self):
        """Process audio data and extract frequencies"""
        # Get audio data from file or microphone
        if self.is_playing_file:
            current_audio_data = self.get_file_audio_data()
        else:
            current_audio_data = self.audio_data
        
        if len(current_audio_data) == 0:
            return
        
        # Apply windowing to reduce spectral leakage
        window = np.hanning(len(current_audio_data))
        windowed_data = current_audio_data * window
        
        # Apply FFT to get frequency domain
        fft = np.fft.fft(windowed_data)
        magnitude = np.abs(fft)
        
        # Take only the first half (positive frequencies)
        magnitude = magnitude[:len(magnitude)//2]
        
        # Group frequencies into exponentially wider bars (logarithmic scale)
        # min_freq = 20  # Hz
        # max_freq = self.sample_rate // 2  # Nyquist frequency
        min_freq, max_freq = self.min_freq, self.max_freq  # Hz, use instance variables
        freq_bins = np.fft.fftfreq(len(current_audio_data), 1 / self.sample_rate)
        freq_bins = freq_bins[:len(magnitude)]

        # Calculate logarithmically spaced frequency boundaries
        log_min = np.log10(min_freq)
        log_max = np.log10(max_freq)
        log_edges = np.logspace(log_min, log_max, self.bars + 1, base=10)
        self.generate_colors()
        for i in range(self.bars):
            # Find indices in FFT corresponding to this frequency range
            start_freq = log_edges[i]
            end_freq = log_edges[i + 1]
            idx = np.where((freq_bins >= start_freq) & (freq_bins < end_freq))[0]
            if len(idx) > 0:
                # Use RMS instead of mean for better dynamic response
                avg_magnitude = np.sqrt(np.mean(magnitude[idx] ** 2))
                
                # Dynamic scaling based on audio source and frequency range
                freq_center = (start_freq + end_freq) / 2
                # if self.is_playing_file:
                #     # Lower frequencies need more scaling, higher frequencies less
                #     if freq_center < 100:
                #         scale_factor = 8000  # Bass frequencies
                #     elif freq_center < 1000:
                #         scale_factor = 2000  # Mid frequencies
                #     else:
                #         scale_factor = 500  # High frequencies
                # else:
                #     # Microphone input scaling``
                #     if freq_center < 100:
                #         scale_factor = 8000  # Bass frequencies
                #     elif freq_center < 1000:
                #         scale_factor = 2000  # Mid frequencies
                #     else:
                #         scale_factor = 500  # High frequencies
                scale_factor = int(30000 / np.power(np.log(freq_center), 3))
                
                # Apply logarithmic scaling for better visual dynamics
                normalized_magnitude = (avg_magnitude / scale_factor) * self.gain_multiplier * self.gain_multiplier_constant
                self.frequencies[i] = min(np.log1p(normalized_magnitude * 10) / np.log1p(10), 1.0)
            else:
                self.frequencies[i] = 0.0
        
        # Smooth the frequencies for better visual effect
        smoothing_factor = 0.8  # Increased smoothing
        self.smoothed_frequencies = (smoothing_factor * self.smoothed_frequencies + 
                                   (1 - smoothing_factor) * self.frequencies)
    
    def draw_frequency_bars(self):
        """Draw the frequency bars visualization"""
        Y_GAP_FROM_BOTTOM = 300
        IDLE_ANIMATION_SPEED = 0.5
        bars_per_layer = self.bars // self.bar_layers + (self.bars % self.bar_layers > 0)
        for i in range(self.bars):
            # Calculate bar height with minimum height for better visuals
            base_height = max(int(self.smoothed_frequencies[i] * self.max_height), 5)
            
            # Calculate position
            x = (i % bars_per_layer) * self.bar_width
            y = self.height - base_height - Y_GAP_FROM_BOTTOM + (i // bars_per_layer * 100)  # Layer offset

            # Add some animation with sine wave (reduced amplitude)
            wave_offset = (math.sin(self.time * IDLE_ANIMATION_SPEED + i * 0.1) + 1)* 4
            bar_height = base_height + int(wave_offset)

            # Ensure minimum height
            bar_height = max(bar_height, 3)
            # Draw bar with gradient effect
            if bar_height > 0:
                color = self.bar_colors[i]
                
                # Enhanced gradient effect with multiple color zones
                for j in range(bar_height):
                    # Create a more dynamic gradient
                    height_ratio = j / bar_height 
                    
                    # Create different intensity zones
                    if height_ratio < 0.3:
                        # Bottom: darker
                        alpha = 0.4 + height_ratio * 0.4
                    elif height_ratio < 0.7:
                        # Middle: full intensity
                        alpha = 0.8 + height_ratio * 0.2
                    else:
                        # Top: bright
                        alpha = 1.0
                    
                    gradient_color = tuple(int(c * alpha) for c in color)
                    pygame.draw.rect(self.screen, gradient_color, 
                                   (x + 2, y + j, self.bar_width - 4, 1))
    
    def draw_waveform(self):
        """Draw a waveform at the bottom"""
        # Get current audio data
        if self.is_playing_file:
            current_audio_data = self.get_file_audio_data()
        else:
            current_audio_data = self.audio_data
            
        if len(current_audio_data) > 1:
            points = []
            wave_height = 80
            wave_y = self.height - 25
            
            for i in range(0, len(current_audio_data), 8):
                x = int((i / len(current_audio_data)) * self.width)
                y = wave_y + int((current_audio_data[i] / 32768) * wave_height)
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, (100, 200, 255), False, points, 2)
    
    def draw_circle_visualizer(self):
        """Draw a circular frequency visualizer"""
        center_x = self.width // 2
        center_y = self.height // 2
        inner_radius = 80
        max_radius = min(self.width, self.height) / 2.5 - inner_radius
        
        for i in range(self.bars):
            angle = (i / self.bars) * 2 * math.pi
            radius = inner_radius + (self.smoothed_frequencies[i] * max_radius)
            
            # Calculate line endpoints
            x1 = center_x + int(inner_radius * math.cos(angle))
            y1 = center_y + int(inner_radius * math.sin(angle))
            x2 = center_x + int(radius * math.cos(angle))
            y2 = center_y + int(radius * math.sin(angle))
            
            # Draw line with color
            color = self.bar_colors[i]
            pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), self.bar_width)
    
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
            else:
                print("Invalid character for frequency input")
                
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

    def draw_frequency_controls(self):
        """Draw frequency range control textboxes"""
        # Min frequency textbox
        min_freq_rect = pygame.Rect(150, 150, 100, 25)
        max_freq_rect = pygame.Rect(150, 180, 100, 25)
        
        # Draw textbox backgrounds
        min_color = (100, 100, 150) if self.active_textbox == 'min' else (60, 60, 60)
        max_color = (100, 100, 150) if self.active_textbox == 'max' else (60, 60, 60)
        
        pygame.draw.rect(self.screen, min_color, min_freq_rect)
        pygame.draw.rect(self.screen, max_color, max_freq_rect)
        pygame.draw.rect(self.screen, (200, 200, 200), min_freq_rect, 2)
        pygame.draw.rect(self.screen, (200, 200, 200), max_freq_rect, 2)
        
        # Draw labels
        min_label = self.font.render("Min Freq:", True, (255, 255, 255))
        max_label = self.font.render("Max Freq:", True, (255, 255, 255))
        self.screen.blit(min_label, (10, 155))
        self.screen.blit(max_label, (10, 185))
        
        # Draw textbox content
        min_text = self.min_freq_text if self.active_textbox == 'min' else str(self.min_freq)
        max_text = self.max_freq_text if self.active_textbox == 'max' else str(self.max_freq)
        
        min_surface = self.font.render(min_text, True, (255, 255, 255))
        max_surface = self.font.render(max_text, True, (255, 255, 255))
        
        self.screen.blit(min_surface, (min_freq_rect.x + 5, min_freq_rect.y + 5))
        self.screen.blit(max_surface, (max_freq_rect.x + 5, max_freq_rect.y + 5))
        
        # Draw cursor for active textbox
        if self.active_textbox == 'min':
            cursor_x = min_freq_rect.x + 5 + min_surface.get_width()
            pygame.draw.line(self.screen, (255, 255, 255), 
                           (cursor_x, min_freq_rect.y + 5), 
                           (cursor_x, min_freq_rect.y + 20), 2)
        elif self.active_textbox == 'max':
            cursor_x = max_freq_rect.x + 5 + max_surface.get_width()
            pygame.draw.line(self.screen, (255, 255, 255), 
                           (cursor_x, max_freq_rect.y + 5), 
                           (cursor_x, max_freq_rect.y + 20), 2)

    def draw_info(self):
        """Draw information text"""
        title = self.font_large.render("Music Visualizer", True, (255, 255, 255))
        self.screen.blit(title, (10, 10))
        
        controls = "SPACE: Toggle | R: Record | L: Load | P: Play/Stop | +/-: Gain | ESC: Quit | Click textboxes to edit freq"
        info = self.font.render(controls, True, (200, 200, 200))
        self.screen.blit(info, (10, 50))
        
        # Show gain level
        gain_text = self.font.render(f"Gain: {self.gain_multiplier:.1f}x", True, (200, 255, 200))
        self.screen.blit(gain_text, (10, 210))
        
        # Show current frequency range
        freq_range_text = self.font.render(f"Freq Range: {self.min_freq:.0f} - {self.max_freq:.0f} Hz", True, (255, 200, 100))
        self.screen.blit(freq_range_text, (260, 165))
        
        # Show audio source status
        if self.is_playing_file:
            if self.audio_file:
                duration = len(self.audio_file) / 1000  # Convert to seconds
                current_pos = pygame.mixer.music.get_pos() / 1000 if pygame.mixer.music.get_pos() != -1 else 0
                source_text = self.font.render(f"🎵 FILE: Playing ({current_pos:.1f}s / {duration:.1f}s)", True, (100, 255, 100))
            else:
                source_text = self.font.render("🎵 FILE: Loaded", True, (100, 255, 100))
        else:
            source_text = self.font.render("🎤 MICROPHONE: Active", True, (255, 200, 100))
        self.screen.blit(source_text, (10, 75))
        
        # Show recording status
        if self.is_recording:
            recording_text = self.font.render(f"🔴 RECORDING: {self.video_filename} (Frame: {self.frame_count})", True, (255, 100, 100))
            self.screen.blit(recording_text, (10, 100))
        else:
            recording_text = self.font.render("Press 'R' to start recording", True, (150, 150, 150))
            self.screen.blit(recording_text, (10, 100))
        
        # Draw frequency controls
        self.draw_frequency_controls()
    
    def run(self):
        """Main game loop"""
        self.start_microphone_input()
        running = True
        visualization_mode = 0  # 0: bars, 1: circle
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_textbox_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if self.active_textbox:
                        # Handle text input for frequency textboxes
                        self.handle_text_input(event)
                    else:
                        # Handle regular controls
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
                            # Increase gain
                            self.gain_multiplier = min(self.gain_multiplier + 0.1, 5.0)
                            print(f"Gain increased to {self.gain_multiplier:.1f}x")
                        elif event.key == pygame.K_MINUS:
                            # Decrease gain
                            self.gain_multiplier = max(self.gain_multiplier - 0.1, 0.1)
                            print(f"Gain decreased to {self.gain_multiplier:.1f}x")
            
            # Process audio
            self.process_audio()
            
            # Clear screen
            self.screen.fill(self.bg_color)
            
            # Draw visualizations
            if visualization_mode == 0:
                self.draw_frequency_bars()
            else:
                self.draw_circle_visualizer()
            
            self.draw_waveform()
            self.draw_info()
            
            # Capture frame for video if recording
            if self.is_recording:
                self.capture_frame()
            
            # Update display
            pygame.display.flip()
            self.time += 0.1
            self.clock.tick(60)  # 60 FPS
        
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
        
        # Clean up any remaining temp files
        temp_wav = os.path.join(os.getcwd(), "temp_audio.wav")
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
                print("Cleaned up temporary audio file on exit")
            except Exception as e:
                print(f"Could not remove temp file on exit: {e}")
        
        pygame.quit()

def main():
    """Main function to run the visualizer"""
    try:
        visualizer = MusicVisualizer()
        visualizer.run()
    except Exception as e:
        print(f"Error running visualizer: {e}")
        print("Make sure you have the required packages installed:")
        print("pip install pygame numpy pyaudio")

if __name__ == "__main__":
    main()
