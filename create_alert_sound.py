"""
สคริปต์สร้างไฟล์เสียงแจ้งเตือนภาษาไทยอัตโนมัติ
- ใช้ gTTS (Google Text-to-Speech) เพื่อสร้างเสียงพูดแจ้งเตือน (หากติดตั้ง)
- มีระบบสำรองสร้างเสียงไซเรนเตือน (Synthetic Siren Tone) ออฟไลน์อัตโนมัติ
"""

import os
import math
import wave
import struct

ASSET_DIR = "assets"
AUDIO_MP3 = os.path.join(ASSET_DIR, "alert.mp3")
AUDIO_WAV = os.path.join(ASSET_DIR, "alert.wav")

TEXT_PROMPT = "เตือน: พื้นที่นี้เป็นช่องจอดเฉพาะรถพยาบาลฉุกเฉินเท่านั้น กรุณาเคลื่อนย้ายรถทันที"

def create_synthetic_siren(filename=AUDIO_WAV, duration_sec=3.0, sample_rate=22050):
    """สร้างเสียงไซเรนเตือนแบบความถี่สลับ (Dual-tone Alarm) โดยไม่ต้องต่อเน็ตและไม่ต้องลงโมดูลเพิ่ม"""
    os.makedirs(ASSET_DIR, exist_ok=True)
    num_samples = int(duration_sec * sample_rate)
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setparams((1, 2, sample_rate, num_samples, 'NONE', 'not compressed'))
        
        for i in range(num_samples):
            t = i / sample_rate
            # ไซเรนสลับความถี่ระหว่าง 750Hz ถึง 1200Hz
            freq = 900 + 300 * math.sin(2 * math.pi * 3.0 * t)
            sample_value = int(32767 * 0.7 * math.sin(2 * math.pi * freq * t))
            data = struct.pack('<h', sample_value)
            wav_file.writeframesraw(data)
            
    print(f"[OK] สร้างเสียงเตือนสำรอง (Siren WAV) สำเร็จ: {filename}")

def generate_voice_alert():
    os.makedirs(ASSET_DIR, exist_ok=True)
    try:
        from gtts import gTTS
        print(f"[INFO] กำลังสร้างเสียงพูดแจ้งเตือนภาษาไทย...")
        tts = gTTS(text=TEXT_PROMPT, lang='th')
        tts.save(AUDIO_MP3)
        print(f"[OK] บันทึกไฟล์เสียงสำเร็จ: {AUDIO_MP3}")
        return AUDIO_MP3
    except Exception as e:
        print(f"[WARN] ไม่สามารถสร้างเสียงผ่าน gTTS ได้ ({e}) -> กำลังสร้างเสียงไซเรนสำรอง...")
        create_synthetic_siren(AUDIO_WAV)
        return AUDIO_WAV

if __name__ == "__main__":
    generate_voice_alert()
