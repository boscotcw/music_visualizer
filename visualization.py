"""
Visualization components for the Music Visualizer
"""
import pygame
import numpy as np
import math
import colorsys

class VisualizationRenderer:
    """Handles all visualization rendering operations"""
    
    def __init__(self, visualizer):
        self.visualizer = visualizer
        
        # Visualization settings
        self.bars = 256
        self.bar_layers = 1  # Number of layers for bar visualization
        self.bar_width = min(visualizer.width * self.bar_layers / self.bars, 1)
        self.max_height = visualizer.height - 100
        
        # Colors and visual state
        self.bar_colors = []
        self.frequencies = np.zeros(self.bars)
        self.smoothed_frequencies = np.zeros(self.bars)
        
        # Animation state
        self.time = 0
        
        # Generate initial colors
        self.generate_colors()
    
    def update_time(self, time_increment=0.1):
        """Update animation time"""
        self.time += time_increment
    
    def update_frequencies(self, new_frequencies):
        """Update frequency data with smoothing"""
        # Handle dynamic bar count changes
        if len(new_frequencies) != len(self.frequencies):
            self.frequencies = np.zeros(len(new_frequencies))
            self.smoothed_frequencies = np.zeros(len(new_frequencies))
            self.bars = len(new_frequencies)
            self.generate_colors()
        
        self.frequencies = new_frequencies
        
        # Smooth the frequencies for better visual effect
        smoothing_factor = 0.5
        self.smoothed_frequencies = (smoothing_factor * self.smoothed_frequencies + 
                                   (1 - smoothing_factor) * self.frequencies)
    
    def generate_colors(self):
        """Generate a spectrum of colors for the bars"""
        self.bar_colors = []
        for i in range(self.bars):
            hue = i / self.bars
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            self.bar_colors.append(tuple(int(c * 255) for c in rgb))
    
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
            y = self.visualizer.height - base_height - Y_GAP_FROM_BOTTOM + (i // bars_per_layer * 100)  # Layer offset

            # Add some animation with sine wave (reduced amplitude)
            wave_offset = (math.sin(self.time * IDLE_ANIMATION_SPEED + i * 0.1) + 1) * 4
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
                    pygame.draw.rect(self.visualizer.screen, gradient_color, 
                                   (x + 2, y + j, self.bar_width - 4, 1))
    
    def draw_waveform(self):
        """Draw a waveform at the bottom"""
        # Get current audio data
        if self.visualizer.is_playing_file:
            current_audio_data = self.visualizer.get_file_audio_data()
        else:
            current_audio_data = self.visualizer.audio_data
            
        if len(current_audio_data) > 1:
            points = []
            wave_height = 80
            wave_y = self.visualizer.height - 25
            
            for i in range(0, len(current_audio_data), 8):
                x = int((i / len(current_audio_data)) * self.visualizer.width)
                y = wave_y + int((current_audio_data[i] / 32768) * wave_height)
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(self.visualizer.screen, (100, 200, 255), False, points, 2)
    
    def draw_circle_visualizer(self):
        """Draw a circular frequency visualizer"""
        center_x = self.visualizer.width // 2
        center_y = self.visualizer.height // 2
        inner_radius = 80
        max_radius = min(self.visualizer.width, self.visualizer.height) / 2.5 - inner_radius
        
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
            pygame.draw.line(self.visualizer.screen, color, (x1, y1), (x2, y2), self.bar_width)
    
    def draw_frequency_controls(self):
        """Draw frequency range control textboxes"""
        # Min frequency textbox
        min_freq_rect = pygame.Rect(150, 150, 100, 25)
        max_freq_rect = pygame.Rect(150, 180, 100, 25)
        
        # Draw textbox backgrounds
        min_color = (100, 100, 150) if self.visualizer.active_textbox == 'min' else (60, 60, 60)
        max_color = (100, 100, 150) if self.visualizer.active_textbox == 'max' else (60, 60, 60)
        
        pygame.draw.rect(self.visualizer.screen, min_color, min_freq_rect)
        pygame.draw.rect(self.visualizer.screen, max_color, max_freq_rect)
        pygame.draw.rect(self.visualizer.screen, (200, 200, 200), min_freq_rect, 2)
        pygame.draw.rect(self.visualizer.screen, (200, 200, 200), max_freq_rect, 2)
        
        # Draw labels
        min_label = self.visualizer.font.render("Min Freq:", True, (255, 255, 255))
        max_label = self.visualizer.font.render("Max Freq:", True, (255, 255, 255))
        self.visualizer.screen.blit(min_label, (10, 155))
        self.visualizer.screen.blit(max_label, (10, 185))
        
        # Draw textbox content
        min_text = self.visualizer.min_freq_text if self.visualizer.active_textbox == 'min' else str(self.visualizer.min_freq)
        max_text = self.visualizer.max_freq_text if self.visualizer.active_textbox == 'max' else str(self.visualizer.max_freq)
        
        min_surface = self.visualizer.font.render(min_text, True, (255, 255, 255))
        max_surface = self.visualizer.font.render(max_text, True, (255, 255, 255))
        
        self.visualizer.screen.blit(min_surface, (min_freq_rect.x + 5, min_freq_rect.y + 5))
        self.visualizer.screen.blit(max_surface, (max_freq_rect.x + 5, max_freq_rect.y + 5))
        
        # Draw cursor for active textbox
        if self.visualizer.active_textbox == 'min':
            cursor_x = min_freq_rect.x + 5 + min_surface.get_width()
            pygame.draw.line(self.visualizer.screen, (255, 255, 255), 
                           (cursor_x, min_freq_rect.y + 5), 
                           (cursor_x, min_freq_rect.y + 20), 2)
        elif self.visualizer.active_textbox == 'max':
            cursor_x = max_freq_rect.x + 5 + max_surface.get_width()
            pygame.draw.line(self.visualizer.screen, (255, 255, 255), 
                           (cursor_x, max_freq_rect.y + 5), 
                           (cursor_x, max_freq_rect.y + 20), 2)

    def draw_info(self):
        """Draw information text"""
        title = self.visualizer.font_large.render("Music Visualizer", True, (255, 255, 255))
        self.visualizer.screen.blit(title, (10, 10))
        
        controls = "SPACE: Toggle | R: Record | L: Load | P: Play/Stop | +/-: Gain | F: Freq Mode | B/V: Bars | ESC: Quit"
        info = self.visualizer.font.render(controls, True, (200, 200, 200))
        self.visualizer.screen.blit(info, (10, 50))
        
        # Show current settings
        gain_text = self.visualizer.font.render(f"Gain: {self.visualizer.gain_multiplier:.1f}x", True, (200, 255, 200))
        self.visualizer.screen.blit(gain_text, (10, 210))
        
        # Show frequency distribution mode
        freq_mode_text = self.visualizer.font.render(f"Freq Mode: {self.visualizer.freq_distribution_mode}", True, (255, 200, 255))
        self.visualizer.screen.blit(freq_mode_text, (10, 235))
        
        # Show bar count
        bars_text = self.visualizer.font.render(f"Bars: {self.bars}", True, (200, 255, 255))
        self.visualizer.screen.blit(bars_text, (260, 210))
        
        # Show current frequency range
        freq_range_text = self.visualizer.font.render(f"Freq Range: {self.visualizer.min_freq:.0f} - {self.visualizer.max_freq:.0f} Hz", True, (255, 200, 100))
        self.visualizer.screen.blit(freq_range_text, (260, 165))
        
        # Show audio source status
        if self.visualizer.is_playing_file:
            if self.visualizer.audio_file:
                duration = len(self.visualizer.audio_file) / 1000  # Convert to seconds
                current_pos = pygame.mixer.music.get_pos() / 1000 if pygame.mixer.music.get_pos() != -1 else 0
                source_text = self.visualizer.font.render(f"🎵 FILE: Playing ({current_pos:.1f}s / {duration:.1f}s)", True, (100, 255, 100))
            else:
                source_text = self.visualizer.font.render("🎵 FILE: Loaded", True, (100, 255, 100))
        else:
            source_text = self.visualizer.font.render("🎤 MICROPHONE: Active", True, (255, 200, 100))
        self.visualizer.screen.blit(source_text, (10, 75))
        
        # Show recording status
        if self.visualizer.is_recording:
            recording_text = self.visualizer.font.render(f"🔴 RECORDING: {self.visualizer.video_filename} (Frame: {self.visualizer.frame_count})", True, (255, 100, 100))
            self.visualizer.screen.blit(recording_text, (10, 100))
        else:
            recording_text = self.visualizer.font.render("Press 'R' to start recording", True, (150, 150, 150))
            self.visualizer.screen.blit(recording_text, (10, 100))
        
        # Draw frequency controls
        self.draw_frequency_controls()
