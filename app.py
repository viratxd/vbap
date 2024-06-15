import gradio as gr
import requests
import random
import os
import zipfile # built in module for unzipping files (thank god)
import librosa
import time
from infer_rvc_python import BaseLoader
from pydub import AudioSegment
from tts_voice import tts_order_voice
import edge_tts
import tempfile
import anyio
from audio_separator.separator import Separator


language_dict = tts_order_voice

# ilaria tts implementation :rofl:
async def text_to_speech_edge(text, language_code):
    voice = language_dict[language_code]
    communicate = edge_tts.Communicate(text, voice)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_path = tmp_file.name

    await communicate.save(tmp_path)

    return tmp_path

# fucking dogshit toggle
try:
    import spaces
    spaces_status = True
except ImportError:
    spaces_status = False

separator = Separator()
converter = BaseLoader(only_cpu=False, hubert_path=None, rmvpe_path=None) # <- yeah so like this handles rvc

global pth_file
global index_file

pth_file = "model.pth"
index_file = "model.index"

#CONFIGS
TEMP_DIR = "temp"
MODEL_PREFIX = "model"
PITCH_ALGO_OPT = [
    "pm",
    "harvest",
    "crepe",
    "rmvpe",
    "rmvpe+",
]
UVR_5_MODELS = [
    {"model_name": "BS-Roformer-Viperx-1297", "checkpoint": "model_bs_roformer_ep_317_sdr_12.9755.ckpt"},
    {"model_name": "MDX23C-InstVoc HQ 2", "checkpoint": "MDX23C-8KFFT-InstVoc_HQ_2.ckpt"},
    {"model_name": "Kim Vocal 2", "checkpoint": "Kim_Vocal_2.onnx"},
    {"model_name": "5_HP-Karaoke", "checkpoint": "5_HP-Karaoke-UVR.pth"},
    {"model_name": "UVR-DeNoise by FoxJoy", "checkpoint": "UVR-DeNoise.pth"},
    {"model_name": "UVR-DeEcho-DeReverb by FoxJoy", "checkpoint": "UVR-DeEcho-DeReverb.pth"},
]

os.makedirs(TEMP_DIR, exist_ok=True)

def unzip_file(file):
    filename = os.path.basename(file).split(".")[0] # converts "model.zip" to "model" so we can do things
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(TEMP_DIR, filename)) # might not be very ram efficient...
    return True
    

def progress_bar(total, current): # best progress bar ever trust me sunglasses emoji 😎 
    return "[" + "=" * int(current / total * 20) + ">" + " " * (20 - int(current / total * 20)) + "] " + str(int(current / total * 100)) + "%"

def download_from_url(url, filename=None):
    if "/blob/" in url:
        url = url.replace("/blob/", "/resolve/") # made it delik proof 😎
    if "huggingface" not in url:
        return ["The URL must be from huggingface", "Failed", "Failed"]
    if filename is None:
        filename = os.path.join(TEMP_DIR, MODEL_PREFIX + str(random.randint(1, 1000)) + ".zip")
    response = requests.get(url)
    total = int(response.headers.get('content-length', 0)) # bytes to download (length of the file)
    if total > 500000000:

        return ["The file is too large. You can only download files up to 500 MB in size.", "Failed", "Failed"]
    current = 0
    with open(filename, "wb") as f:
        for data in response.iter_content(chunk_size=4096): # download in chunks of 4096 bytes (4kb - helps with memory usage and speed)
            f.write(data)
            current += len(data)
            print(progress_bar(total, current), end="\r") # \r is a carriage return, it moves the cursor to the start of the line so its like tqdm sunglasses emoji 😎
    
    # unzip because the model is in a zip file lel

    try:
        unzip_file(filename)
    except Exception as e:
        return ["Failed to unzip the file", "Failed", "Failed"] # return early if it fails and like tell the user but its dogshit hahahahahahaha 😎 According to all known laws aviation, there is no way a bee should be able to fly.
    unzipped_dir = os.path.join(TEMP_DIR, os.path.basename(filename).split(".")[0]) # just do what we did in unzip_file because we need the directory
    pth_files = []
    index_files = []
    for root, dirs, files in os.walk(unzipped_dir): # could be done more efficiently because nobody stores models in subdirectories but like who cares (it's a futureproofing thing lel)
        for file in files:
            if file.endswith(".pth"):
                pth_files.append(os.path.join(root, file))
            elif file.endswith(".index"):
                index_files.append(os.path.join(root, file))
    
    print(pth_files, index_files) # debug print because im fucking stupid and i need to see what is going on
    global pth_file
    global index_file
    pth_file = pth_files[0]
    index_file = index_files[0]

    pth_file_ui.value = pth_file
    index_file_ui.value = index_file
    print(pth_file_ui.value)
    print(index_file_ui.value)
    return ["Downloaded as " + filename, pth_files[0], index_files[0]]

def inference(audio, model_name):
        output_data = inf_handler(audio, model_name)
        vocals = output_data[0]
        inst = output_data[1]

        return vocals, inst

if spaces_status:
    @spaces.GPU()
    def convert_now(audio_files, random_tag, converter):
        return converter(
            audio_files,
            random_tag,
            overwrite=False,
            parallel_workers=8
        )

        
else:
    def convert_now(audio_files, random_tag, converter):
        return converter(
            audio_files,
            random_tag,
            overwrite=False,
            parallel_workers=8
        )

def calculate_remaining_time(epochs, seconds_per_epoch):
    total_seconds = epochs * seconds_per_epoch

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    if hours == 0:
        return f"{int(minutes)} minutes"
    elif hours == 1:
        return f"{int(hours)} hour and {int(minutes)} minutes"
    else:
        return f"{int(hours)} hours and {int(minutes)} minutes"

def inf_handler(audio, model_name): # its a shame that zerogpu just WONT cooperate with us
    model_found = False
    for model_info in UVR_5_MODELS:
        if model_info["model_name"] == model_name:
            separator.load_model(model_info["checkpoint"])
            model_found = True
            break
    if not model_found:
        separator.load_model()
    output_files = separator.separate(audio)
    vocals = output_files[0]
    inst = output_files[1]
    return vocals, inst

    
def run(
    audio_files,
    pitch_alg,
    pitch_lvl,
    index_inf,
    r_m_f,
    e_r,
    c_b_p,
):
    if not audio_files:
        raise ValueError("The audio pls")

    if isinstance(audio_files, str):
        audio_files = [audio_files]

    try:
        duration_base = librosa.get_duration(filename=audio_files[0])
        print("Duration:", duration_base)
    except Exception as e:
        print(e)

    random_tag = "USER_"+str(random.randint(10000000, 99999999))

    file_m = pth_file_ui.value
    file_index = index_file_ui.value

    print("Random tag:", random_tag)
    print("File model:", file_m)
    print("Pitch algorithm:", pitch_alg)
    print("Pitch level:", pitch_lvl)
    print("File index:", file_index)
    print("Index influence:", index_inf)
    print("Respiration median filtering:", r_m_f)
    print("Envelope ratio:", e_r)

    converter.apply_conf(
        tag=random_tag,
        file_model=file_m,
        pitch_algo=pitch_alg,
        pitch_lvl=pitch_lvl,
        file_index=file_index,
        index_influence=index_inf,
        respiration_median_filtering=r_m_f,
        envelope_ratio=e_r,
        consonant_breath_protection=c_b_p,
        resample_sr=44100 if audio_files[0].endswith('.mp3') else 0, 
    )
    time.sleep(0.1)

    result = convert_now(audio_files, random_tag, converter)
    print("Result:", result)

    return result[0]

def upload_model(index_file, pth_file):
    pth_file = pth_file.name
    index_file = index_file.name
    pth_file_ui.value = pth_file
    index_file_ui.value = index_file
    return "Uploaded!"  

with gr.Blocks(theme="Ilaria RVC") as demo:
    gr.Markdown("## Ilaria RVC 💖")
    with gr.Tab("Inference"):
        sound_gui = gr.Audio(value=None,type="filepath",autoplay=False,visible=True,)
        pth_file_ui = gr.Textbox(label="Model pth file",value=pth_file,visible=False,interactive=False,)
        index_file_ui = gr.Textbox(label="Index pth file",value=index_file,visible=False,interactive=False,)

        with gr.Accordion("Settings", open=False):
            pitch_algo_conf = gr.Dropdown(PITCH_ALGO_OPT,value=PITCH_ALGO_OPT[4],label="Pitch algorithm",visible=True,interactive=True,)
            pitch_lvl_conf = gr.Slider(label="Pitch level (lower -> 'male' while higher -> 'female')",minimum=-24,maximum=24,step=1,value=0,visible=True,interactive=True,)
            index_inf_conf =  gr.Slider(minimum=0,maximum=1,label="Index influence -> How much accent is applied",value=0.75,)
            respiration_filter_conf = gr.Slider(minimum=0,maximum=7,label="Respiration median filtering",value=3,step=1,interactive=True,)
            envelope_ratio_conf = gr.Slider(minimum=0,maximum=1,label="Envelope ratio",value=0.25,interactive=True,)
            consonant_protec_conf = gr.Slider(minimum=0,maximum=0.5,label="Consonant breath protection",value=0.5,interactive=True,)

        button_conf = gr.Button("Convert",variant="primary",)
        output_conf = gr.Audio(type="filepath",label="Output",)
    	
        button_conf.click(lambda :None, None, output_conf)
        button_conf.click(
            run,
            inputs=[
                sound_gui,
                pitch_algo_conf,
                pitch_lvl_conf,
                index_inf_conf,
                respiration_filter_conf,
                envelope_ratio_conf,
                consonant_protec_conf,
            ],
            outputs=[output_conf],
        )
    
    with gr.Tab("Ilaria TTS"):
            text_tts = gr.Textbox(label="Text", placeholder="Hello!", lines=3, interactive=True,)
            dropdown_tts = gr.Dropdown(label="Language and Model",choices=list(language_dict.keys()),interactive=True, value=list(language_dict.keys())[0])

            button_tts = gr.Button("Speak", variant="primary",)

            output_tts = gr.Audio(type="filepath", label="Output",)

            button_tts.click(text_to_speech_edge, inputs=[text_tts, dropdown_tts], outputs=[output_tts])


    with gr.Tab("Model Loader (Download and Upload)"):
        with gr.Accordion("Model Downloader", open=False):
            gr.Markdown(
                "Download the model from the following URL and upload it here. (Hugginface RVC model)"
            )
            model = gr.Textbox(lines=1, label="Model URL")
            download_button = gr.Button("Download Model")
            status = gr.Textbox(lines=1, label="Status", placeholder="Waiting....", interactive=False)
            model_pth = gr.Textbox(lines=1, label="Model pth file", placeholder="Waiting....", interactive=False)
            index_pth = gr.Textbox(lines=1, label="Index pth file", placeholder="Waiting....", interactive=False)
            download_button.click(download_from_url, model, outputs=[status, model_pth, index_pth])
        with gr.Accordion("Upload A Model", open=False):
            index_file_upload = gr.File(label="Index File (.index)")
            pth_file_upload = gr.File(label="Model File (.pth)")
            upload_button = gr.Button("Upload Model")
            upload_status = gr.Textbox(lines=1, label="Status", placeholder="Waiting....", interactive=False)

            upload_button.click(upload_model, [index_file_upload, pth_file_upload], upload_status)
    

    with gr.Tab("Vocal Separator (UVR)"):
        gr.Markdown("Separate vocals and instruments from an audio file using UVR models. - This is only on CPU due to ZeroGPU being ZeroGPU :(")
        uvr5_audio_file = gr.Audio(label="Audio File",type="filepath")

        with gr.Row():
            uvr5_model = gr.Dropdown(label="Model", choices=[model["model_name"] for model in UVR_5_MODELS])
            uvr5_button = gr.Button("Separate Vocals", variant="primary",)

        uvr5_output_voc = gr.Audio(type="filepath", label="Output 1",) # UVR models sometimes output it in a weird way where it's like the positions swap randomly, so let's just call them Outputs lol
        uvr5_output_inst = gr.Audio(type="filepath", label="Output 2",)

        uvr5_button.click(inference, [uvr5_audio_file, uvr5_model], [uvr5_output_voc, uvr5_output_inst])
    
    with gr.Tab("Extra"):
        with gr.Accordion("Training Time Calculator", open=False):
            with gr.Column():
                epochs_input = gr.Number(label="Number of Epochs")
                seconds_input = gr.Number(label="Seconds per Epoch")
                calculate_button = gr.Button("Calculate Time Remaining")
                remaining_time_output = gr.Textbox(label="Remaining Time", interactive=False)
                
                calculate_button.click(
                    fn=calculate_remaining_time,
                    inputs=[epochs_input, seconds_input],
                    outputs=[remaining_time_output]
                )

        with gr.Accordion("Model Fusion", open=False):
            gr.Markdown(value="Fusion of two models to create a new model - coming soon! 😎")

        with gr.Accordion("Model Quantization", open=False):
            gr.Markdown(value="Quantization of a model to reduce its size - coming soon! 😎")
        
        with gr.Accordion("Training Helper", open=False):
            gr.Markdown(value="Help for training models - coming soon! 😎")

    with gr.Tab("Credits"):
        gr.Markdown(
            """
            Ilaria RVC made by [Ilaria](https://huggingface.co/TheStinger) suport her on [ko-fi](https://ko-fi.com/ilariaowo)
            
            The Inference code is made by [r3gm](https://huggingface.co/r3gm) (his module helped form this space 💖)

            made with ❤️ by [mikus](https://github.com/cappuch) - i make this ui........

            ## In loving memory of JLabDX 🕊️
            """
        )

demo.queue(api_open=False).launch(show_api=False) # idk ilaria if you want or dont want to
