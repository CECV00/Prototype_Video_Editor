
import streamlit as st
import torch
import torchaudio
from transformers.models.wav2vec2 import Wav2Vec2ForCTC
#from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
#from moviepy.editor import VideoFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from streamlit_player import st_player
import numpy as np
import wave
import os

# Procesador de audio en vivo
class AudioProcessor:
    def __init__(self):
        self.frames = []

    def recv(self, frame):
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return frame

    def save_audio(self, filename="comando.wav"):
        if self.frames:
            audio_np = np.concatenate(self.frames, axis=0)
            audio_np = (audio_np * 32767).astype(np.int16)

            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_np.tobytes())

            return filename
        return None

# Transcribir el audio
def transcribir_audio(ruta_audio="comando.wav"):
    processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-spanish")
    model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-spanish")

    waveform, sample_rate = torchaudio.load(ruta_audio)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    input_values = processor(waveform.squeeze(), return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    texto = processor.batch_decode(predicted_ids)[0].lower()
    return texto

# Descargar video desde YouTube
def descargar_video_youtube(url, nombre_salida="video_original.mp4"):
    opciones = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': nombre_salida,
        'merge_output_format': 'mp4',
        'quiet': True,
        'no_warnings': True
    }
    with yt_dlp.YoutubeDL(opciones) as ydl:
        ydl.download([url])
    return nombre_salida

# Cortar un video entre dos minutos
def cortar_video(video_path, inicio_min, fin_min, salida="video_cortado.mp4"):
    inicio_seg = inicio_min * 60
    fin_seg = fin_min * 60
    clip = VideoFileClip(video_path).subclip(inicio_seg, fin_seg)
    clip.write_videofile(salida, codec="libx264")
    return salida

# Procesar el texto transcrito
def procesar_comando(texto):
    if "cortar" in texto:
        st.info("🟠 Se detectó el comando: **cortar video**")

        inicio = st.number_input("⏱️ Minuto inicial", min_value=0, value=0, step=1)
        fin = st.number_input("⏱️ Minuto final", min_value=inicio + 1, value=inicio + 1, step=1)

        video_file = st.file_uploader("📤 Sube tu video", type=["mp4", "mov", "avi"])
        if video_file is not None:
            with open("video_original.mp4", "wb") as f:
                f.write(video_file.getbuffer())

            st.video("video_original.mp4")

            if st.button("✂️ Cortar video"):
                salida = cortar_video("video_original.mp4", inicio, fin)
                st.success("✅ Video cortado correctamente.")
                st.video(salida)

# Interfaz principal
def main():
    st.title("🎤 Editor de Video por Voz con Streamlit")
    st.write("Sube o graba un audio con tu comando de voz.")

    st.subheader("🌐 Reproducir video desde URL")
    url_video = st.text_input("🔗 Ingresa una URL de video (YouTube, Vimeo, etc.)")

    if url_video:
        st_player(url_video)
        #descargar_video_youtube(url_video)

    opcion = st.radio("¿Cómo deseas ingresar el audio?", ["🎙️ Grabar con micrófono", "📁 Subir archivo"])
    texto = ""

    if opcion == "📁 Subir archivo":
        audio_file = st.file_uploader("Selecciona un archivo de audio", type=["wav", "mp3"])
        if audio_file is not None:
            with open("comando.wav", "wb") as f:
                f.write(audio_file.getbuffer())
            texto = transcribir_audio("comando.wav")

    elif opcion == "🎙️ Grabar con micrófono":
        processor = AudioProcessor()
        ctx = webrtc_streamer(
            key="grabacion_audio",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            video_processor_factory=None,
            audio_processor_factory=lambda: processor,
        )

        if st.button("Guardar y transcribir audio"):
            archivo = processor.save_audio("comando.wav")
            if archivo:
                st.success("✅ Audio grabado y guardado.")
                texto = transcribir_audio(archivo)
            else:
                st.error("⚠️ No se ha grabado audio.")

    if texto:
        st.write(f"📝 Texto detectado: `{texto}`")
        procesar_comando(texto)

if __name__ == "__main__":
    main()
