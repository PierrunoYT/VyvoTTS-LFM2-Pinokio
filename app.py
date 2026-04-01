import gc
from snac import SNAC
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading SNAC model...")
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
snac_model = snac_model.to(device)

# Available models - LFM2 models
MODELS = {
    "Jenny": "Vyvo/VyvoTTS-LFM2-Jenny",
    "Neuvillette": "Vyvo/VyvoTTS-LFM2-Neuvillette",
    "Optimus Prime": "Vyvo/VyvoTTS-LFM2-Optimus-Prime",
    "Multi-Speaker": "Vyvo/VyvoTTS-LFM2-Multi-Speaker",
    "Itto": "Vyvo/VyvoTTS-LFM2-Itto",
    "Stephen_Fry": "Vyvo/VyvoTTS-LFM2-Stephen_Fry",
    "Alhaitham": "Vyvo/VyvoTTS-LFM2-Alhaitham",
    "Cyno": "Vyvo/VyvoTTS-LFM2-Cyno",
    "Dehya": "Vyvo/VyvoTTS-LFM2-Dehya",
    "Elise": "Vyvo/VyvoTTS-LFM2-Elise",
    "Kaeya": "Vyvo/VyvoTTS-LFM2-Kaeya",
    "Kaveh": "Vyvo/VyvoTTS-LFM2-Kaveh",
    "Ningguang": "Vyvo/VyvoTTS-LFM2-Ningguang",
    "Heizou": "Vyvo/VyvoTTS-LFM2-Heizou",
    "Thoma": "Vyvo/VyvoTTS-LFM2-Thoma",
    "Tighnari": "Vyvo/VyvoTTS-LFM2-Tighnari",
}

# Store for currently loaded model
current_model = None
current_tokenizer = None
current_model_choice = None

def load_model_if_needed(model_choice):
    """Load model and tokenizer, unloading previous model if different"""
    global current_model, current_tokenizer, current_model_choice
    
    if current_model_choice != model_choice:
        # Unload previous model if exists
        if current_model is not None:
            print(f"Unloading previous model: {current_model_choice}")
            del current_model
            del current_tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Load new model
        model_name = MODELS[model_choice]
        print(f"Loading {model_choice} model: {model_name}")
        current_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        current_model.to(device)
        current_tokenizer = AutoTokenizer.from_pretrained(model_name)
        current_model_choice = model_choice
        print(f"{model_choice} model loaded successfully!")
    
    return current_model, current_tokenizer

# LFM2 Special Tokens Configuration
TOKENIZER_LENGTH = 64400
START_OF_TEXT = 1
END_OF_TEXT = 7
START_OF_SPEECH = TOKENIZER_LENGTH + 1
END_OF_SPEECH = TOKENIZER_LENGTH + 2
START_OF_HUMAN = TOKENIZER_LENGTH + 3
END_OF_HUMAN = TOKENIZER_LENGTH + 4
START_OF_AI = TOKENIZER_LENGTH + 5
END_OF_AI = TOKENIZER_LENGTH + 6
PAD_TOKEN = TOKENIZER_LENGTH + 7
AUDIO_TOKENS_START = TOKENIZER_LENGTH + 10

# Process text prompt for LFM2
def process_prompt(prompt, tokenizer, device):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    start_token = torch.tensor([[START_OF_HUMAN]], dtype=torch.int64)
    end_tokens = torch.tensor([[END_OF_TEXT, END_OF_HUMAN]], dtype=torch.int64)
    
    modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
    
    # No padding needed for single input
    attention_mask = torch.ones_like(modified_input_ids)
    
    return modified_input_ids.to(device), attention_mask.to(device)

# Parse output tokens to audio for LFM2
def parse_output(generated_ids):
    token_to_find = START_OF_SPEECH
    token_to_remove = END_OF_SPEECH
    
    token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

    if len(token_indices[1]) > 0:
        last_occurrence_idx = token_indices[1][-1].item()
        cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
    else:
        cropped_tensor = generated_ids

    processed_rows = []
    for row in cropped_tensor:
        masked_row = row[row != token_to_remove]
        processed_rows.append(masked_row)

    code_lists = []
    for row in processed_rows:
        row_length = row.size(0)
        new_length = (row_length // 7) * 7
        trimmed_row = row[:new_length]
        trimmed_row = [t - AUDIO_TOKENS_START for t in trimmed_row]
        code_lists.append(trimmed_row)
        
    return code_lists[0]  # Return just the first one for single sample

# Redistribute codes for audio generation
def redistribute_codes(code_list, snac_model):
    device = next(snac_model.parameters()).device  # Get the device of SNAC model
    
    layer_1 = []
    layer_2 = []
    layer_3 = []
    for i in range((len(code_list)+1)//7):
        layer_1.append(code_list[7*i])
        layer_2.append(code_list[7*i+1]-4096)
        layer_3.append(code_list[7*i+2]-(2*4096))
        layer_3.append(code_list[7*i+3]-(3*4096))
        layer_2.append(code_list[7*i+4]-(4*4096))
        layer_3.append(code_list[7*i+5]-(5*4096))
        layer_3.append(code_list[7*i+6]-(6*4096))
        
    # Move tensors to the same device as the SNAC model
    codes = [
        torch.tensor(layer_1, device=device).unsqueeze(0),
        torch.tensor(layer_2, device=device).unsqueeze(0),
        torch.tensor(layer_3, device=device).unsqueeze(0)
    ]
    
    audio_hat = snac_model.decode(codes)
    return audio_hat.detach().squeeze().cpu().numpy()  # Always return CPU numpy array

# Main generation function
def generate_speech(text, model_choice, temperature, top_p, repetition_penalty, max_new_tokens, progress=gr.Progress()):
    if not text.strip():
        return None
    
    try:
        progress(0.1, "Loading model and processing text...")
        model, tokenizer = load_model_if_needed(model_choice)
        
        # Voice parameter is always None for LFM2 models
        input_ids, attention_mask = process_prompt(text, tokenizer, device)
        
        progress(0.3, "Generating speech tokens...")
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=1,
                eos_token_id=END_OF_SPEECH,
            )
        
        progress(0.6, "Processing speech tokens...")
        code_list = parse_output(generated_ids)
        
        progress(0.8, "Converting to audio...")
        audio_samples = redistribute_codes(code_list, snac_model)
        
        progress(1.0, "Completed!")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (24000, audio_samples)
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None

# Example texts
EXAMPLE_TEXTS = [
    "Hello! I am a speech system. I can read your text with a natural voice.",
    "Today is a beautiful day. The weather is perfect for a walk.",
    "The sun rises from the east and sets in the west. This is a rule of nature.",
    "Technology makes our lives easier every day."
]    

# Create modern Gradio interface using built-in theme
with gr.Blocks(title="VyvoTTS LFM2", theme=gr.themes.Soft()) as demo:
    # Header section
    gr.Markdown("""
    # VyvoTTS LFM2
    ### [Github](https://github.com/Vyvo-Labs/VyvoTTS) | [HF Models](https://huggingface.co/collections/Vyvo/lfm2-tts-689eedae5353ff5b048efd55)
    """)
    
    gr.Markdown("""
    VyvoTTS is a text-to-speech model by Vyvo team using LFM2 architecture, trained on multiple diverse open-source datasets. 
    Since some datasets may contain transcription errors or quality issues, output quality can vary. 
    Higher quality datasets typically produce better speech synthesis results.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            # Text input section
            text_input = gr.Textbox(
                label="Text Input",
                placeholder="Enter the text you want to convert to speech...",
                lines=6,
                max_lines=10
            )
            
            # Voice model selection
            model_choice = gr.Radio(
                choices=list(MODELS.keys()),
                value="Jenny",
                label="Voice Model",
                visible=True
            )
            
            # Advanced settings
            with gr.Accordion("Advanced Settings", open=False):
                temperature = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.3, step=0.05,
                    label="Temperature", 
                    info="Higher values create more expressive but less stable speech"
                )
                top_p = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.95, step=0.05,
                    label="Top P", 
                    info="Nucleus sampling threshold value"
                )
                repetition_penalty = gr.Slider(
                    minimum=1.0, maximum=2.0, value=1.2, step=0.05,
                    label="Repetition Penalty", 
                    info="Higher values discourage repetitive patterns"
                )
                max_new_tokens = gr.Slider(
                    minimum=100, maximum=2000, value=2000, step=100,
                    label="Maximum Length", 
                    info="Maximum length of generated audio (in tokens)"
                )
            
            # Action buttons
            with gr.Row():
                submit_btn = gr.Button("Generate Speech", variant="primary", size="lg")
                clear_btn = gr.Button("Clear", size="lg")
        
        with gr.Column(scale=1):
            # Output section
            audio_output = gr.Audio(
                label="Generated Audio",
                type="numpy",
                interactive=False
            )
    
    # Example texts at the bottom
    with gr.Row():
        example_1_btn = gr.Button(EXAMPLE_TEXTS[0], size="sm")
        example_2_btn = gr.Button(EXAMPLE_TEXTS[1], size="sm")
    
    with gr.Row():
        example_3_btn = gr.Button(EXAMPLE_TEXTS[2], size="sm")
        example_4_btn = gr.Button(EXAMPLE_TEXTS[3], size="sm")
    
    # Set up example button events
    example_1_btn.click(fn=lambda: EXAMPLE_TEXTS[0], outputs=text_input)
    example_2_btn.click(fn=lambda: EXAMPLE_TEXTS[1], outputs=text_input)
    example_3_btn.click(fn=lambda: EXAMPLE_TEXTS[2], outputs=text_input)
    example_4_btn.click(fn=lambda: EXAMPLE_TEXTS[3], outputs=text_input)
    
    # Set up event handlers
    submit_btn.click(
        fn=generate_speech,
        inputs=[text_input, model_choice, temperature, top_p, repetition_penalty, max_new_tokens],
        outputs=audio_output,
        show_progress=True
    )
    
    def clear_interface():
        return "", None
    
    clear_btn.click(
        fn=clear_interface,
        inputs=[],
        outputs=[text_input, audio_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.queue().launch(share=False, ssr_mode=False)
