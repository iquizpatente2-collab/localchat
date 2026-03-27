# Localchat: Local Conversational AI for Raspberry Pi5

The **Localchat** project was a test to see if I could build a low-latency, local (non-networked) voice assistant in Python for the Raspberry Pi. The system combines real-time speech recognition, LLM-based dialog, and high-quality TTS into a reactive system running on Raspberry Pi5.

The final device is housed in a Game 5 Pi retro arcade case. The AdaFruit arcade style LED button was integrated to provide feedback and control. USB ports are used for the camera/mic array (Playstation Eye) and speaker audio out. A USB flash drive is used for headless configuration. The original build targeted a life-size Stormtrooper prop; the codebase name is now **Localchat**.

### Rename / compatibility (existing installs)

- **Config file (Pi):** preferred path is `~/.localchat_config.json`. The loader still accepts **`~/.trooper_config.json`** if the new file is missing.
- **USB override:** `LOCALCHAT*` volume + `localchat_config.json` is preferred; legacy **`TROOPER*` + `trooper_config.json`** is still detected and copied into `~/.localchat_config.json`.
- **LED FIFO:** default stays **`/tmp/trooper_led`** for backward compatibility. To use a new path, set **`LOCALCHAT_LED_FIFO`** (same value for `main.py` and anything using `led_request()`).

<img src="docs/system_pic1.jpg" alt="system_pic1" style="zoom:33%;" />

## Conclusion

The project was a success. The streaming architecture was capable of providing low enough latency to make possible a reasonable conversation with Localchat. The `Gemma3:1b` and `Qwen2.5:0.5b` models provided acceptable performance. The Gemma3 model was able to provide a more direct, authoritarian persona, while Qwen2.5 was faster, but generally provided a more friendly interaction. The programmable `System Message` is key in tuning your desired personality. I decided on Vosk for STT, although I did extensive testing with faster-whisper. Piper gave excellent performance for TTS, and many voices are available.

## Features

- Fully integrated into headless Raspberry Pi5-8Gb
- No reliance on remote API calls or cloud providers
- WebSocket client/server architecture with full-duplex mic/speaker support
- Sentence streaming Speech-to-Text (STT) using lightweight Vosk model.  Support for any Vosk voice. Strong character voice achieved using stock Piper voice `en_US-danny-low.onnx`. Additional support for add-on voice effects.
- Sentence-by-sentence streaming Text-to-Speech (TTS) using Piper
- LLM inference is achieved locally using Ollama. Tested with two lightweight models: `gemma3:1b` and `qwen2.5:0.5b`
- Configurable mic-mute mode for setup with a speaker and separate mic
- JSON-based configuration file: `.localchat_config.json` (legacy `.trooper_config.json` still supported on device)
- Configurable device names (mic and speaker)
- Arcade style lighted button for visual feedback and control. The large LED provides feedback (listening / speaking / thinking) and a push button to start or stop sessions as an alternative to gesture detection mode.
- Detection and elimination of false low-energy utterances
- System can be triggered via push button or gesture detection (camera + MediaPipe Hands model)

## Performance

Packing a low-latency voice system onto a raspberry pi device was a challenge. The Pi5 made this project possible. I opted not to include the AI kit or SSD, so the system runs on a stock Pi5 8Gb RAM and a 32Gb MicroSD card running stock Pi OS.

During Vosk STT, Inference via Ollama, and Piper TTS, the CPU on the Pi5 is completely maxed out at 100%.

The active cooler fan was installed as well as an additional case fan integrated in the Pi52 retro arcade case.

Over a large number of dialog samples, the following average timings were recorded:

- Vosk STT ~10ms
- LLM ~3–15 sec depending on prompt
- Piper TTS ~2–5 sec per response
- All speech was streamed sentence-by-sentence for responsiveness

Note that neither the Vosk STT (input) nor Piper TTS (output) were designed for true token by token streaming. I had to modify the system to detect sentence breaks via punctuation and silence boundaries to trigger the stream. The allows for long responses from the LLM to be read back without waiting for the entire response, making the system seem much more responsive. The system is able to respond with long elaborate stories, especially using the `gemma3:1b` model without issue.

I experimented with Faster-Whisper projects as an alternative to Vosk. In the end, I stayed with Vosk. It was lighter and worked well. I observed the even Whisper STT was not designed for true streaming and while it was responsive on the Pi5, it still would require modifications to keep sentences together. The small Vosk model, while lower performing, was satisfactory for this assistant. If you are building a therapist, for example, or application where greater accuracy is required, you may need to pursue Faster-Whisper.

## Python File Overview

| File        | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| `main.py`   | **Main system entry point**. Manages session lifecycle (start/stop), LED state, and gesture-based or button-based activation. Handles Piper playback for greetings and timeouts. Pre-warms the LLM model. |
| `client.py` | **Audio interface and WebSocket client**. Captures audio from the mic, sends it to the server, and plays back streamed TTS audio. Handles volume control, fade-in/out, and mic muting to prevent feedback. |
| `server.py` | **Streaming WebSocket server**. Receives audio, performs real-time speech-to-text (Vosk), queries the LLM via Ollama, and streams TTS responses (Piper). Sends playback audio back in chunks for smooth UX. |
| `utils.py`  | **Shared utilities**. Includes configuration loading (USB override), audio device detection, LED control via FIFO pipe, and fade-in/out DSP for playback audio. |

## Core Architecture

The primary goal of the project to was to create a local voice solution, small enough to install in a compact build (e.g. a life-size prop), with acceptable latency to allow for a natural conversation with Localchat.

#### Speech Input

The system captures audio through a Playstation PS-Eye mic array connected to the Raspberry Pi5 via USB-A port. The PS-Eye has a 4 mic array that is sensitive enough to allow users at a distance to be able to speak to the assistant.

Audio-in highlights:

- Uses `PyAudio` to capture live mic input.
- Optional voice activity detection (VAD) gates LED feedback.
- Audio is streamed to the server in 16kHz mono PCM format.

#### Vosk Speech Recognition (STT)

- Vosk is used in batch mode.
- Each utterance is sent to the LLM only after a silence break.

#### Ollama LLM

For inference, the system uses a local Ollama install to provide API for the chosen LLM model. Multiple models supported using Ollama. Two models have been tested expensively with the project, `gemma2:2b` and `qwen2.5:0.5b`.  You can pull models onto the Pi5 as long as you have free RAM to store them. 

```
$ ollama list
NAME                   ID              SIZE      MODIFIED     
qwen2.5:0.5b           a8b0c5157701    397 MB    10 days ago        
gemma3:1b              8ccf136fdd52    815 MB    6 weeks ago          
```

To keep the system responsive, you need to choose a lightweight model, otherwise the token rate out of Ollama will be insufficient to provide a comfortable conversation. The system uses `Ollama` to stream JSON token-by-token responses. Each sentence-ending token triggers real-time TTS.

Choose your model in the JSON configuration file:

```
"model_name": "gemma3:1b",
```

The system also implements configurable System Prompt to give the assistant its personality. The default System Prompt is stored in the JSON configuration file:

```
"system_prompt":   "You are a loyal Imperial Stormtrooper. 
					You need to keep order. 
					Your weapon is a lightsabre. 
					Dont ask to help or assist.",
```

#### Piper Text-to-Speech (TTS)

The system uses the Piper Text-to-Speech engine for natural voice synthesis.

- Piper generates 16kHz mono audio.
- SoX upsamples to 48kHz stereo.
- Optional Retro Voice FX filtering (SoX high-pass, low-pass, compand, and noise mix) can be applied using SoX high-pass, low-pass, and noise effects.
- Audio is streamed back to the client in ~2048 byte chunks.

#### Audio Output

Audio is implemented using a low-cost USB speaker.

- Audio is played in a background thread using `PyAudio`.
- ~50ms of silence is prepended to each sentence to avoid clipping.
- A playback queue ensures smooth streaming.
- Fade-in and Fade-out effects are applied to voice outputs for smoother audio.

#### LED/Switch/Camera Integration

The system integrates an LED / Switch combination. The LED is used to communicate status of the system. The AdaFruit 30mm illuminated arcade style button can be used to start/stop a session with Localchat.

- LED modes reflect states: `listen`, `blink`, `speak`, `solid`.
- Controlled via FIFO pipe (`/tmp/trooper_led` by default; override with `LOCALCHAT_LED_FIFO`) and interpreted by `main.py`.

The switch is wired into GPIO pins of the Raspberry Pi5.

The Playstation Eye USB camera / microphone is used for camera and audio input. The device provides a sensitive 4-array microphone. The camera is used for gesture detection to initiate sessions automatically.

## Project Structure

```
Localchat/   # or your clone directory name
├── client.py             # Audio I/O, mic, speaker, LED
├── server.py             # Streaming server: LLM, STT, TTS
├── main.py               # Launches client on gesture/button
├── utils.py              # Shared helpers (e.g. led_request)
├── voices/               # Piper voice models
├── vosk-model/           # Vosk STT models
├── localchat_config.json # Example JSON config (copy to Pi home)
├── requirements.txt      # Dependencies file
├── client.log            # Log output for client debug
```

## Project Requirements

#### Python Dependencies

Install all required Python packages via:

```
pip install -r requirements.txt
```

**`requirements.txt`**

```
aiofiles==23.2.1
aiohttp==3.9.3
asyncio
numpy==1.26.4
pyaudio==0.2.13
python-dotenv==1.0.1
soxr==0.3.7
soundfile==0.12.1
websockets==12.0
vosk==0.3.45
gpiozero==2.0
lgpio==0.0.4
opencv-python==4.9.0.80
mediapipe==0.10.9
```

> `pyaudio` may require `portaudio19-dev` to build correctly on some systems.

#### System Dependencies

These are **not** installed via pip and must be installed via your OS package manager or manually.

##### APT Install (Debian / Ubuntu)

```
sudo apt update && sudo apt install -y \
    sox \
    pulseaudio \
    ffmpeg \
    python3-pyaudio \
    libasound-dev \
    portaudio19-dev
```

##### Piper (Text-to-Speech Engine)

Used for fast local speech synthesis.

```
# Build from source (requires Rust)
cargo install piper

# OR download a prebuilt binary from:
# https://github.com/rhasspy/piper/releases
```

> Place the binary at `~/.local/bin/piper` or update the path in `server.py`.

##### Ollama (LLM Backend)

Ollama runs your local language models like `gemma` or `qwen2.5`.

```
curl -fsSL https://ollama.com/install.sh | sh
```

Start and load your preferred model:

```
ollama serve &
ollama pull gemma3:1b
```

##### Audio System

Ensure `PulseAudio` is running:

```
pulseaudio --start
```

Make sure your user is in the audio group:

```
sudo usermod -aG audio $USER
```

Then log out or reboot.

## WebSocket Architecture

Localchat uses a bidirectional WebSocket connection between the **client** (audio I/O and playback on device) and the **server** (speech recognition, LLM inference, and TTS).

#### Message Flow Overview

```
[ Mic Audio ] ──► client.py ── send ─► server.py ──► 
STT ─► LLM ─► TTS ──► 
client.py ──► [ Audio Output ]
```

#### Message Loop

##### 1. **Client → Server**

- The microphone stream is continuously captured.
- It is resampled (if needed) and sent as **binary audio chunks** via WebSocket.
- These chunks are 16kHz mono PCM in `int16` format.

##### 2. **Server**

- Uses `Vosk` for real-time speech recognition.
- Once a full utterance is detected:
  - The transcript is sent to the LLM (via Ollama).
  - The response is synthesized using `Piper`.
  - Audio is optionally processed with SoX for retro voice effects.

##### 3. **Server → Client**

- The TTS audio is streamed back in small binary chunks.
- When playback is complete, the server sends the string message `"__END__"`.

##### 4. **Client Playback**

- On receiving audio, the client:
  - Optionally mutes the mic to prevent feedback.
  - Plays the audio stream in real time.
  - Sends `"__done__"` to the server to indicate playback is finished.

#### Message Types

| Direction       | Type         | Description                                       |
| --------------- | ------------ | ------------------------------------------------- |
| Client → Server | `bytes`      | 16-bit PCM audio input                            |
| Server → Client | `bytes`      | 16-bit PCM TTS output                             |
| Server → Client | `"__END__"`  | Signals end of TTS segment                        |
| Client → Server | `"__done__"` | Signals playback complete (used for LED feedback) |

## Configuration

The system is configured via a JSON file named `.localchat_config.json` on the device (`/home/<user>/...`). A sample ships in the repo as `localchat_config.json`. Legacy **`~/.trooper_config.json`** is still read if the new file is absent.

#### USB-Based Configuration Override

To support headless operation, configuration updates can be applied via a USB flash drive:

- Format the drive with the volume name: **`Localchat`** (preferred) or legacy **`Trooper`**
- Place **`localchat_config.json`** (preferred) or legacy **`trooper_config.json`** in the root of the USB
- On boot or restart, if the USB file is detected, it will:
  - Be **loaded immediately**
  - Be **copied** to `~/.localchat_config.json`, making it the new default

This allows users to easily update persona (e.g. voice, model, prompt) without SSH access.

#### Sample Configuration (`localchat_config.json`)

```
{
  "volume": 95,
  "mic_name": "USB Camera-B4.09.24.1: Audio",
  "audio_output_device": "USB PnP Sound Device: Audio",
  "model_name": "gemma3:1b",
  "voice": "danny-low.onnx",
  "mute_mic_during_playback": true,
  "fade_duration_ms": 100,
  "retro_voice_fx": false,
  "history_length": 6,
  "system_prompt": "You are a loyal Imperial Stormtrooper.",
  "greeting_message": "Civilian detected!",
  "closing_message": "Mission completed. Carry on with your civilian duties.",
  "timeout_message": "Communication terminated. Returning to base.",
  "session_timeout": 500,
  "vision_wake": false
}
```

#### Parameter Descriptions

| Key                        | Description                                                  |
| -------------------------- | ------------------------------------------------------------ |
| `volume`                   | Initial system audio level (0–100) applied at boot.          |
| `mic_name`                 | Partial or exact match string for the microphone input device. |
| `audio_output_device`      | Partial or exact match string for audio output device.       |
| `model_name`               | Local LLM to use via Ollama (e.g., `gemma3:1b`, `qwen2.5:0.5b`). |
| `voice`                    | Piper voice model filename (must exist in `voices/` directory). |
| `mute_mic_during_playback` | Prevents audio feedback by muting mic during TTS playback (recommended: `true`). |
| `fade_duration_ms`         | Fade-in/out duration in milliseconds for smoother playback transitions. Set to `0` to disable. |
| `retro_voice_fx`           | Enables SoX filters for vintage radio effect (high-pass, compression, etc.). |
| `history_length`           | Number of previous user/system messages retained for context-aware LLM replies. |
| `system_prompt`            | Role-based instruction injected into LLM at start of each session (sets persona and tone). |
| `greeting_message`         | Spoken at session start, using the configured voice.         |
| `closing_message`          | Spoken at session end.                                       |
| `timeout_message`          | Spoken if session times out with no user input.              |
| `session_timeout`          | Session timeout in seconds. If no activity, session will auto-close. |
| `vision_wake`              | Reserved for future use (e.g., camera-based wake triggers). Set to `false`. |

## Vision-Based Wake (Gesture Detection)

Localchat supports **gesture-based activation** as an alternative to the physical button.

Using a webcam and the MediaPipe library, the system continuously monitors for a raised open hand gesture using real-time hand landmark detection. When five fingers are detected extended for a brief streak, Localchat toggles its session (start/stop).

#### Activation Logic

- Uses **MediaPipe Hands** for landmark tracking
- Requires 5 fingers to be up
- Requires a **streak** of consistent detection (e.g. 5 frames in a row)
- Cooldown enforced between gesture activations (default: 10 seconds)

#### Requirements

This feature requires:

- `opencv-python`
- `mediapipe`

These are included in the `requirements.txt`.

#### Enable Gesture Detection

Gesture detection is optional and controlled via config:

```
{
  "vision_wake": true
}
```

Set this flag in your `.localchat_config.json` or `localchat_config.json` on the USB (legacy `trooper_config.json` still works on USB).

## GPIO Connections

The Localchat system uses the Raspberry Pi 5’s GPIO header to connect:

- A **30mm Adafruit arcade-style LED pushbutton**
- A **case cooling fan**
- The **official Pi5 active cooler** (connected separately via fan header)

#### GPIO Pinout Table

| Component              | GPIO Pin | Physical Pin | Function                     |
| ---------------------- | -------- | ------------ | ---------------------------- |
| **Arcade Button**      | GPIO 17  | Pin 11       | Input (detect button press)  |
| **Button LED**         | GPIO 18  | Pin 12       | Output (blink status LED)    |
| **Button Power (+5V)** | —        | Pin 2        | +5V power for LED ring       |
| **Button Ground**      | —        | Pin 6        | Ground for button + LED      |
| **Fan Power (+5V)**    | —        | Pin 4        | +5V for external case fan    |
| **Fan Ground**         | —        | Pin 34       | Ground for external case fan |

#### Notes on Button Logic

- The arcade button uses **internal pull-up resistors**, which is why its switch contact is connected to **+5V**.
- The logic is **active-low**: pressing the button pulls GPIO 17 **low**, triggering an event.
- The button is **debounced in software** and configured with `hold_time=0.75` seconds in `main.py`, so it only activates Localchat on a **long press**.
- Short taps are ignored and logged as `"Ignored short press"`.

> This debounce and long-press detection helps avoid accidental session toggles due to noise or brief contact.

## Starting and Systemd Integration

#### Services:

- `localchat-server.service`: runs the LLM + TTS backend (`server.py`)
- `localchat-main.service`: launches the LED/session manager (`main.py`)

#### Starting the System

To test the system, start the `server.py` and `main.py`. If you don't wont the button control, you can start `client.py` directly instead of starting `main.py`:

```
# Start the server
cd /path/to/Localchat && python3 server.py

# Start the main, which controls the initial and closing greetings, 
# the arcade button, and launches the client
cd /path/to/Localchat && python3 main.py

# Start the client directly
cd /path/to/Localchat && python3 client.py
```

#### Automatic Operation

For automatic operation, the client and server can be started via `Systemd`

##### Example: `/etc/systemd/system/localchat-server.service`

```
[Unit]
Description=Localchat Voice Server (LLM + TTS)
After=network.target sound.target

[Service]
ExecStart=/usr/bin/python3 /home/mjw/Trooper/server.py
WorkingDirectory=/home/mjw/Trooper
Restart=always
User=mjw

[Install]
WantedBy=multi-user.target
```

##### Example: `/etc/systemd/system/localchat-main.service`

```
[Unit]
Description=Localchat Main Controller (LED + Session Launcher)
After=localchat-server.service

[Service]
ExecStart=/usr/bin/python3 /home/mjw/Trooper/main.py
WorkingDirectory=/home/mjw/Trooper
Restart=always
User=mjw

[Install]
WantedBy=multi-user.target
```

#### Setup:

```
sudo systemctl enable localchat-server.service
sudo systemctl enable localchat-main.service
sudo systemctl start localchat-server.service
sudo systemctl start localchat-main.service
```

To verify:

```
systemctl status localchat-server
systemctl status localchat-main
```

Use `systemctl list-unit-files | grep localchat` to confirm they are enabled. (Rename unit files if you still use `trooper-*.service` — or keep old names and only update `Description` text.)

## References

Localchat stands on the shoulders of giants. I could not have built this system without the brilliant work shared by these open-source pioneers and educators:

- [Vosk STT](https://alphacephel.com/vosk/) – Lightweight, off-line-capable speech recognition engine.
- [Piper TTS](https://github.com/rhasspy/piper) – High-quality local text-to-speech engine developed by the Rhasspy team.
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) – Optimized Whisper inference using CTranslate2.
- [Whisper Streaming by UFAL](https://github.com/ufal/whisper_streaming) – Real-time whisper implementation.
- YouTube Inspirations:
  - [“Build a Voice Assistant with Whisper + TTS”](https://www.youtube.com/watch?v=3yLFWpKKbe8)
  - [“Offline AI Assistant on Pi”](https://www.youtube.com/watch?v=_spinzpEeFM)

> Open source makes this possible. If you're building a similar system, go give these projects a star 🌟 and support them however you can.

## License

MIT 2.0

## *May The Force Be With You Always*