import streamlit as st
import requests


class AllModels:

    def run(self):

        choice = st.radio(
            "Тандоо кыл:",
            ["WAV файл жүктөө", "Голос жазуу"],
            key="dataset_audio_mode"
        )


        audio_bytes = None
        filename = "recording.wav"
        mime_type = "audio/wav"

        if choice == "WAV файл жүктөө":
            st.write("WAV файл жүктө")

            uploaded_file = st.file_uploader(
                "",
                type=["wav", "mp3"],
                key="audio_file"
            )

            if uploaded_file is not None:
                audio_bytes = uploaded_file.getvalue()
                filename = uploaded_file.name
                mime_type = uploaded_file.type or "audio/wav"

                st.audio(audio_bytes, format=mime_type)

        else:
            audio_value = st.audio_input("Голос жаз")

            if audio_value is not None:
                audio_bytes = audio_value.getvalue()
                st.audio(audio_value)

        if st.button("Predict"):
            if audio_bytes is None:
                st.warning("Алгач аудио файл жүктө же голос жаз!")
                return

            files = {
                "file": (filename, audio_bytes, mime_type)
            }

            try:
                response = requests.post(
                    "http://127.0.0.1:8000/audio1/predict/",
                    files=files
                )

                if response.status_code == 200:
                    result = response.json()

                    if isinstance(result, dict):
                        result = (
                            result.get("Класс")
                            or result.get("class")
                            or result.get("genre")
                            or result
                        )

                    elif isinstance(result, list) and result:
                        result = result[0]

                    st.success(
                        f"Модель думает, что это жанр: {result}"
                    )

                else:
                    st.error(f"Error: {response.status_code}")
                    st.write(response.text)

            except requests.exceptions.ConnectionError:
                st.error("FastAPI сервер иштеп жаткан жок!")




