"""
ระบบควบคุมการเล่นเสียงผ่านลำโพง (Audio Player)
- ใช้ pygame.mixer สำหรับเล่นไฟล์เสียง MP3 / WAV
- ทำงานแบบ Asynchronous (Non-blocking) เพื่อไม่ให้วิดีโอกล้องสะดุด
- มีระบบ Cooldown ป้องกันไม่ให้เสียงทับซ้อนกัน
"""

import os
import time
import threading

class AudioAlertPlayer:
    def __init__(self, audio_file="assets/alert.wav", volume=1.0, min_interval=8.0):
        self.audio_file = audio_file
        self.volume = max(0.0, min(1.0, volume))
        self.min_interval = min_interval
        self.last_played_time = 0.0
        self.lock = threading.Lock()
        self.is_ready = False
        
        # ตรวจสอบไฟล์เสียง ถ้าไม่มีไฟล์ ให้ลองหาทางเลือกอื่น
        if not os.path.exists(self.audio_file):
            for candidate in ["assets/alert.wav", "assets/alert.mp3"]:
                if os.path.exists(candidate):
                    self.audio_file = candidate
                    break

        self._init_mixer()

    def _init_mixer(self):
        try:
            import pygame
            if not pygame.mixer.get_init():
                # กำหนด Audio Driver สำหรับ Linux/Raspberry Pi
                if "SDL_AUDIODRIVER" not in os.environ:
                    os.environ["SDL_AUDIODRIVER"] = "alsa"
                try:
                    pygame.mixer.init()
                except Exception:
                    # กรณีค่าเริ่มต้นไม่ผ่าน ให้ลองกำหนด buffer และ sample rate สำหรับ Pi
                    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)

            if os.path.exists(self.audio_file):
                pygame.mixer.music.load(self.audio_file)
                pygame.mixer.music.set_volume(self.volume)
                self.is_ready = True
                print(f"[AUDIO] Ready: {self.audio_file} (Volume: {self.volume * 100:.0f}%)")
            else:
                print(f"[WARN] Audio file not found: {self.audio_file}")
        except Exception as e:
            print(f"[ERROR] pygame mixer init failed: {e}")
            print("[INFO] Fallback to system audio player (aplay / mpg123 / ffplay) enabled.")
            self.is_ready = False

    def play(self, force=False):
        """สั่งเล่นเสียงแจ้งเตือน (รันใน Background Thread ไม่สะดุดภาพ)"""
        now = time.time()
        with self.lock:
            # เช็คว่าอยู่ในช่วง Cooldown หรือไม่
            if not force and (now - self.last_played_time < self.min_interval):
                return False

            self.last_played_time = now

        threading.Thread(target=self._play_worker, daemon=True).start()
        return True

    def _play_worker(self):
        try:
            import pygame
            if not self.is_ready:
                self._init_mixer()

            if self.is_ready:
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play()
                    print("[ALARM] >>> Playing warning sound: EMERGENCY VEHICLE ZONE ONLY! <<<")
            else:
                # ระบบสำรองกรณี pygame เข้าถึงการ์ดเสียงใน systemd ไม่ได้ (รันคำสั่งตรงของ Linux)
                import subprocess
                if os.path.exists(self.audio_file):
                    print("[ALARM] >>> Playing warning sound via fallback CLI player <<<")
                    if self.audio_file.endswith(".wav"):
                        subprocess.Popen(["aplay", "-q", self.audio_file])
                    elif self.audio_file.endswith(".mp3"):
                        # ลอง mpg123 หรือ ffplay
                        subprocess.Popen(f"mpg123 -q '{self.audio_file}' || ffplay -nodisp -autoexit -loglevel quiet '{self.audio_file}'", shell=True)
        except Exception as e:
            print(f"[ERROR] Audio playback error: {e}")

    def stop(self):
        try:
            import pygame
            if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
        except Exception:
            pass
