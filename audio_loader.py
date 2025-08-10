"""
Audio loading functionality for the Music Visualizer
"""
import os
import numpy as np
import pygame
import tkinter as tk
from tkinter import filedialog
import tempfile
import subprocess
from pydub import AudioSegment
import wave

class AudioLoader:
    """Handles all audio file loading operations"""
    
    def __init__(self, visualizer):
        self.visualizer = visualizer
        self._configure_ffmpeg()
    
    def _configure_ffmpeg(self):
        """Configure FFmpeg paths for pydub"""
        AudioSegment.converter = "C:/Program Files/FFMPEG/bin/ffmpeg.exe"
        AudioSegment.ffmpeg = "C:/Program Files/FFMPEG/bin/ffmpeg.exe"
        AudioSegment.ffprobe = "C:/Program Files/FFMPEG/bin/ffprobe.exe"
    
    def select_audio_file(self) -> str:
        """Select an audio file using file dialog"""
        root = tk.Tk()
        root.withdraw()
        
        return filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.mp3 *.wav *.m4a *.flac *.ogg"),
                ("MP3 Files", "*.mp3"),
                ("WAV Files", "*.wav"),
                ("All Files", "*.*")
            ]
        )
    
    def validate_file(self, file_path) -> bool:
        """Validate and log file information"""
        if not os.path.isfile(file_path):
            print(f"Error: File does not exist: {file_path}")
            return False
        
        file_size = os.path.getsize(file_path)
        print(f"Loading audio file: {file_path}")
        print(f"File size: {file_size / (1024*1024):.2f} MB")
        
        try:
            with open(file_path, 'rb') as f:
                first_bytes = f.read(16)
                print(f"File header (first 16 bytes): {first_bytes.hex()}")
            return True
        except Exception as read_error:
            print(f"Cannot read file: {read_error}")
            return False

    def load_mp3_progressive(self, file_path) -> tuple[any, str]:
        """Try multiple methods to load MP3 files"""
        print("Attempting MP3 loading with multiple methods...")
        
        # Method 1: Standard pydub MP3 loading
        try:
            return AudioSegment.from_mp3(file_path), "standard method"
        except Exception as e:
            print(f"Standard MP3 loading failed: {e}")
        
        # Method 2: Force format specification
        try:
            return AudioSegment.from_file(file_path, format="mp3"), "explicit format"
        except Exception as e:
            print(f"Explicit format loading failed: {e}")
        
        # Method 3: Auto-detection
        try:
            return AudioSegment.from_file(file_path), "auto-detection"
        except Exception as e:
            print(f"Auto-detection failed: {e}")
        
        # Method 4: Custom paths with parameters
        # try:
        #     import pydub.utils
        #     original_which = pydub.utils.which
            
        #     def custom_which(name):
        #         paths = {
        #             "ffmpeg": "C:/Program Files/FFMPEG/bin/ffmpeg.exe",
        #             "ffprobe": "C:/Program Files/FFMPEG/bin/ffprobe.exe"
        #         }
        #         return paths.get(name, original_which(name))
            
        #     pydub.utils.which = custom_which
        #     audio = AudioSegment.from_file(file_path, format="mp3", 
        #                                  parameters=["-ac", "1", "-ar", "44100"])
        #     pydub.utils.which = original_which
        #     return audio, "forced parameters and custom paths"
        # except Exception as e:
        #     print(f"Forced parameters failed: {e}")
        
        # Method 5: Direct FFmpeg conversion
        return self.load_with_direct_ffmpeg(file_path)

    def load_with_direct_ffmpeg(self, file_path) -> any:
        """Convert audio file using direct FFmpeg command"""
        print("Trying direct FFmpeg conversion...")
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav.close()
        
        ffmpeg_cmd = [
            "C:/Program Files/FFMPEG/bin/ffmpeg.exe",
            "-i", file_path, "-ac", "1", "-ar", "44100", "-y", temp_wav.name
        ]
        
        try:
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                audio = AudioSegment.from_wav(temp_wav.name)
                os.unlink(temp_wav.name)
                return audio, "direct FFmpeg conversion"
            else:
                print(f"FFmpeg direct conversion failed: {result.stderr}")
                os.unlink(temp_wav.name)
        except Exception as e:
            print(f"Direct FFmpeg conversion failed: {e}")
            if os.path.exists(temp_wav.name):
                os.unlink(temp_wav.name)
        
        raise Exception("All MP3 loading methods failed")
    
    def load_wav_file(self, file_path) -> any:
        """Load WAV file with fallback methods"""
        try:
            # Try native wave library first
            with wave.open(file_path, 'rb') as wav_file:
                frames = wav_file.readframes(-1)
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
            
            audio = AudioSegment(data=frames, sample_width=sample_width, 
                               frame_rate=sample_rate, channels=channels)
            print("Successfully loaded WAV using native wave library")
            return audio
        except Exception as wav_error:
            print(f"Native wave library failed: {wav_error}, trying pydub...")
            return AudioSegment.from_wav(file_path)
    
    def load_by_extension(self, file_path, extension) -> any:
        """Load audio file based on extension"""
        loaders = {
            '.mp3': lambda: self.load_mp3_progressive(file_path),
            '.wav': lambda: (self.load_wav_file(file_path), "pydub"),
            '.m4a': lambda: (AudioSegment.from_file(file_path, format="m4a"), "m4a"),
            '.flac': lambda: (AudioSegment.from_file(file_path, format="flac"), "flac"),
            '.ogg': lambda: (AudioSegment.from_ogg(file_path), "ogg")
        }
        
        if extension in loaders:
            result = loaders[extension]()
            if isinstance(result, tuple):
                audio, method = result
                print(f"✅ {extension.upper()} loaded with {method}")
                return audio
            else:
                return result
        else:
            print("Unknown file type, trying auto-detection...")
            return AudioSegment.from_file(file_path)
    
    def finalize_audio_loading(self, audio, file_path) -> bool:
        """Process and store the loaded audio"""
        # Convert to stereo and resample to 44100 Hz
        # audio = audio.set_channels(2)
        audio = audio.set_frame_rate(44100)

        # Convert to numpy array and normalize
        self.visualizer.audio_data_file = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if len(self.visualizer.audio_data_file) > 0:
            max_val = np.max(np.abs(self.visualizer.audio_data_file))
            if max_val > 0:
                self.visualizer.audio_data_file = self.visualizer.audio_data_file / max_val * 16384
        
        # Set audio properties
        self.visualizer.audio_file = audio
        self.visualizer.file_position = 0
        self.visualizer.file_sample_rate = 44100
        self.visualizer.is_playing_file = False
        
        # Initialize pygame mixer
        if not self.visualizer.pygame_mixer_initialized:
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=1024)
            self.visualizer.pygame_mixer_initialized = True
        
        print(f"Audio file loaded successfully!")
        print(f"Duration: {len(audio)/1000:.2f} seconds")
        print(f"Sample rate: {audio.frame_rate} Hz")
        print(f"Channels: {audio.channels}")
        return True
    
    def show_troubleshooting_guide(self, file_path, error):
        """Display troubleshooting information"""
        file_format = self.visualizer.diagnose_audio_file(file_path)
        
        print(f"Error loading audio file: {error}")
        print("\n=== TROUBLESHOOTING GUIDE ===")
        
        if file_format == "wav":
            print("🔧 WAV file troubleshooting:")
            print("   - File might be corrupted")
            print("   - Unsupported WAV encoding (try converting with a different tool)")
            print("   - File might be too large")
        elif file_format in ["mp3", "m4a", "flac", "ogg"]:
            print(f"🔧 {file_format.upper()} file troubleshooting:")
            if not self.visualizer.check_ffmpeg_availability():
                print("   ❌ FFmpeg is required but not found!")
                print("   📥 Install FFmpeg:")
                print("      1. Download from https://ffmpeg.org/download.html")
                print("      2. Add to system PATH")
                print("      3. Or use: choco install ffmpeg (if you have Chocolatey)")
            else:
                print("   ✅ FFmpeg is available")
                if file_format == "mp3":
                    print("   🎵 MP3-specific issues:")
                    print("      - Variable bitrate (VBR) MP3s can be problematic")
                    print("      - Large ID3 tags might cause issues")
                    print("      - Try converting with: ffmpeg -i input.mp3 -acodec mp3 -ab 128k output.mp3")
        else:
            print("🔧 Unknown file format - try converting to WAV")
        
        print("\n💡 Quick solutions:")
        print("   1. Convert to WAV format (works without FFmpeg)")
        print("   2. Try a different audio file")
        print("   3. Use online converters to convert audio files")
        print("=============================\n")
    
    def load_audio_file(self):
        """Main method to load an audio file for visualization"""
        file_path = self.select_audio_file()
        
        if not file_path or not os.path.exists(file_path):
            return False
        
        if not self.validate_file(file_path):
            return False
        
        file_extension = os.path.splitext(file_path)[1].lower()
        print(f"File extension: {file_extension}")
        
        # Check FFmpeg availability for non-WAV files
        if file_extension != '.wav' and not self.visualizer.check_ffmpeg_availability():
            print(f"\n⚠️  WARNING: FFmpeg not found!")
            print(f"File type {file_extension} requires FFmpeg for loading.")
            print(f"Attempting to load anyway...\n")
        
        try:    
            audio = self.load_by_extension(file_path, file_extension)
            return self.finalize_audio_loading(audio, file_path)
            
        except Exception as e:
            self.show_troubleshooting_guide(file_path, e)
            return False
