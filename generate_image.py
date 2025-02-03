import os
import sys
import time
import logging
import torch
import numpy as np
import PIL.Image
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor

# Configuration
MODEL_PATH = "/app/models/janus-pro-1b"
OUTPUT_DIR = "/app/generated_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if os.getenv('DEBUG') else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optimize for CPU
torch.set_num_threads(4)
torch.set_flush_denormal(True)

def load_model():
    logger.info("Initializing model...")
    try:
        start_time = time.time()
        
        processor = VLChatProcessor.from_pretrained(MODEL_PATH)
        tokenizer = processor.tokenizer

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).cpu().eval()

        logger.info(f"Model loaded successfully in {time.time()-start_time:.1f}s")
        logger.debug(f"Model device: {next(model.parameters()).device}")
        logger.debug(f"Parameter count: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, processor, tokenizer
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        sys.exit(1)

def get_prompt():
    try:
        if sys.stdin.isatty():
            return input("\nEnter your prompt (or press Enter for default): ").strip()
        return os.getenv('PROMPT', '').strip()
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(0)

def verify_environment():
    """Check directory permissions and dependencies"""
    logger.info("Verifying environment...")
    
    # Test file writing
    test_file = os.path.join(OUTPUT_DIR, "write_test.txt")
    try:
        with open(test_file, "w") as f:
            f.write("Docker write test successful\n")
        os.remove(test_file)
        logger.debug("Write test passed")
    except Exception as e:
        logger.error(f"Write test failed: {str(e)}")
        sys.exit(1)

    # Check PyTorch (modified section)
    logger.debug(f"PyTorch version: {torch.__version__}")
    logger.debug(f"PyTorch device support: CPU={torch.cuda.is_available()}")  # Changed line

def generate_image():
    verify_environment()
    model, processor, tokenizer = load_model()
    
    prompt = get_prompt() or "A beautiful sunset over snow-capped mountains"
    logger.info(f"Starting generation for prompt: {prompt}")

    # Prepare conversation
    conversation = [
        {"role": "<|User|>", "content": prompt},
        {"role": "<|Assistant|>", "content": ""},
    ]

    try:
        sft_format = processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=processor.sft_format,
            system_prompt="",
        )
        full_prompt = sft_format + processor.image_start_tag
        logger.debug(f"Full prompt: {full_prompt}")
    except Exception as e:
        logger.error(f"Prompt processing failed: {str(e)}")
        sys.exit(1)

    # Generation parameters
    params = {
        "temperature": 1.2,
        "parallel_size": 1,
        "cfg_weight": 7.0,
        "image_token_num_per_image": 144,
        "img_size": 256,
        "patch_size": 16
    }

    try:
        start_time = time.time()
        image = generate(model, processor, full_prompt, **params)
        
        timestamp = int(time.time())
        output_path = os.path.join(OUTPUT_DIR, f"result_{timestamp}.jpg")
        image.save(output_path)
        
        logger.info(f"Image saved to {output_path}")
        logger.info(f"Total generation time: {time.time()-start_time:.1f}s")
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        sys.exit(1)

@torch.inference_mode()
def generate(mmgpt, processor, prompt, **kwargs):
    """Core generation function with enhanced logging"""
    logger.debug("Starting generation process")
    
    # Unpack parameters
    temperature = kwargs.get('temperature', 1.0)
    parallel_size = kwargs.get('parallel_size', 1)
    cfg_weight = kwargs.get('cfg_weight', 5.0)
    image_token_num = kwargs.get('image_token_num_per_image', 144)
    img_size = kwargs.get('img_size', 256)
    patch_size = kwargs.get('patch_size', 16)

    # Prepare inputs
    try:
        input_ids = processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)
        logger.debug(f"Encoded input IDs: {input_ids.shape}")
    except Exception as e:
        logger.error(f"Input encoding failed: {str(e)}")
        raise

    # Create token batch
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = processor.pad_id

    # Generate embeddings
    try:
        inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
        logger.debug(f"Input embeddings shape: {inputs_embeds.shape}")
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise

    generated_tokens = torch.zeros((parallel_size, image_token_num), dtype=torch.int)
    past_key_values = None

    # Generation loop
    for i in range(image_token_num):
        try:
            outputs = mmgpt.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            hidden_states = outputs.last_hidden_state

            logits = mmgpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token_cat = torch.cat([next_token.unsqueeze(1), next_token.unsqueeze(1)], dim=1).view(-1)
            img_embeds = mmgpt.prepare_gen_img_embeds(next_token_cat)
            inputs_embeds = img_embeds.unsqueeze(1)

            if (i + 1) % 10 == 0:
                logger.debug(f"Generated {i+1}/{image_token_num} tokens")

        except Exception as e:
            logger.error(f"Generation failed at step {i}: {str(e)}")
            raise

    # Decode and validate image
    try:
        logger.debug("Decoding image...")
        dec = mmgpt.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
        ).cpu().numpy().transpose(0, 2, 3, 1)

        if dec.size == 0:
            raise ValueError("Decoded image array is empty")
            
        logger.debug(f"Decoded image shape: {dec.shape}")
        logger.debug(f"Pixel value range: {dec.min()} - {dec.max()}")

        dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
        return PIL.Image.fromarray(dec[0])

    except Exception as e:
        logger.error(f"Image decoding failed: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("Starting Janus-Pro Image Generation")
    generate_image()
