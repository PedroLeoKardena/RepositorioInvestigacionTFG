import numpy as np
import soundfile as sf
import os

print("=== 1. Comprobando importaciones ===")
# DeepFilterNet se importa como 'df'
librerias = ["librosa", "pydub", "soundfile", "torchaudio", 
             "noisereduce", "pyannote.audio", "speechbrain", "df"]

todas_ok = True
for lib in librerias:
    try:
        __import__(lib)
        print(f"✅ {lib} importado correctamente.")
    except ImportError as e:
        print(f"❌ Error al importar {lib}: {e}")
        todas_ok = False

if not todas_ok:
    print("\n⚠️ Hay errores en las importaciones. Deteniendo la prueba.")
    exit()

print("\n=== 2. Comprobando integración de FFmpeg y procesamiento ===")
try:
    # 1. Crear un audio de prueba (un tono de 440Hz de 1 segundo)
    print("-> Generando audio de prueba (test.wav)...")
    sample_rate = 16000
    t = np.linspace(0, 1, sample_rate)
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write('test.wav', audio_data, sample_rate)

    # 2. Probar torchaudio (Carga el audio a un tensor de PyTorch)
    import torchaudio
    waveform, sr = torchaudio.load('test.wav')
    print(f"Torchaudio funciona: Tensor creado con forma {waveform.shape}")

    # 3. Probar pydub (Esto verificará que FFmpeg está accesible en tu sistema)
    from pydub import AudioSegment
    audio_pydub = AudioSegment.from_file('test.wav')
    print(f"Pydub y FFmpeg funcionan: Audio cargado, duración {len(audio_pydub)} ms")

    # 4. Probar noisereduce (Aplicar reducción de ruido al array de numpy)
    import noisereduce as nr
    reduced_noise = nr.reduce_noise(y=audio_data, sr=sample_rate)
    print(f"Noisereduce funciona: Array procesado correctamente.")

    # 5. Probar librosa
    import librosa
    y, sr_librosa = librosa.load('test.wav', sr=None)
    print(f"Librosa funciona: Frecuencia de muestreo {sr_librosa} Hz")

    print("\n¡TODO FUNCIONA PERFECTAMENTE! Tu entorno está listo.")

except Exception as e:
    print(f"\nSe produjo un error durante la prueba de procesamiento:\n{e}")
finally:
    # Limpieza del archivo de prueba
    if os.path.exists('test.wav'):
        os.remove('test.wav')