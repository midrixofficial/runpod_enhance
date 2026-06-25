import os
import cv2
import numpy as np
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor
import base64
import requests
import runpod

# ================= CONFIG =================
ONNX_PATH = "realesr_general_x4v3.onnx"
# Cap thread count to prevent GIL bottlenecks and CPU scheduling overhead on high-core RunPod servers
NUM_THREADS = min(8, max(1, os.cpu_count() // 2))
SCALE = 4

# Initialize session globally (outside handler) to leverage RunPod warm containers
print("🚀 Initializing ONNX Runtime Session...")
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = NUM_THREADS
# For sequential models like Real-ESRGAN, ORT_SEQUENTIAL is more efficient and uses less memory
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.enable_cpu_mem_arena = True

# Setup CPU/OpenCV threads
cv2.setUseOptimized(True)
cv2.setNumThreads(NUM_THREADS)

# Try CUDA execution provider first, fallback to CPU
# We use DEFAULT convolution algorithm search to avoid CUDA search failures that produce NaN/black images.
providers = [
    ("CUDAExecutionProvider", {
        "cudnn_conv_algo_search": "DEFAULT",
    }),
    "CPUExecutionProvider"
]

try:
    session = ort.InferenceSession(ONNX_PATH, sess_options, providers=providers)
    print(f"✅ ONNX Session loaded. Active providers: {session.get_providers()}")
except Exception as e:
    print(f"⚠️ Error loading session with providers {providers}: {e}")
    print("🔄 Retrying with CPUExecutionProvider only...")
    session = ort.InferenceSession(ONNX_PATH, sess_options, providers=["CPUExecutionProvider"])
    print(f"✅ ONNX Session loaded on CPU.")

try:
    INPUT_NAME = session.get_inputs()[0].name
    OUTPUT_NAME = session.get_outputs()[0].name
    print("Input:", INPUT_NAME)
    print("Output:", OUTPUT_NAME)
    
    print("Model Inputs:")
    for inp_info in session.get_inputs():
        print(f"  {inp_info.name} : {inp_info.type} {inp_info.shape}")
    print("Model Outputs:")
    for out_info in session.get_outputs():
        print(f"  {out_info.name} : {out_info.type} {out_info.shape}")
except Exception as e:
    print("⚠️ Error getting model input/output names:", e)
    INPUT_NAME = "input"
    OUTPUT_NAME = "output"

# ================= IMAGE UTILS =================
def decode_base64_image(base64_str):
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid base64 image data")
    return img

def download_image_from_url(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    nparr = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Invalid image from URL: {url}")
    return img

def encode_image_to_base64(img, format_str=".jpg"):
    success, buffer = cv2.imencode(format_str, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise ValueError("Failed to encode image to buffer")
    return base64.b64encode(buffer).decode("utf-8")

# ================= CORE LOGIC =================
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) * (1.0 / 255.0)
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, 0)

def postprocess(output):
    output = np.squeeze(output)
    output = np.transpose(output, (1, 2, 0))
    output = (output * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

def process_tile(args):
    sess, img_padded, x1, x2, y1, y2, tile_pad, scale = args
    pad = tile_pad * 2
    tile = img_padded[y1:y2 + pad, x1:x2 + pad]
    
    inp = preprocess(tile)
    out = sess.run([OUTPUT_NAME], {INPUT_NAME: inp})[0]
    
    print("=" * 60)
    print("Provider:", sess.get_providers())
    print("dtype :", out.dtype)
    print("shape :", out.shape)
    print("min   :", np.min(out))
    print("max   :", np.max(out))
    print("mean  :", np.mean(out))
    print("nan   :", np.isnan(out).any())
    print("=" * 60)
    
    out = postprocess(out)
    
    p = tile_pad * scale
    out = out[p:-p, p:-p]
    return (y1, y2, x1, x2, out)

def enhance_image_logic(img, sharpen_amount=0.35, contrast_alpha=1.04, brightness_beta=3, tile_size=256, tile_pad=16):
    h, w, c = img.shape
    
    # Check if we should use tiling.
    # We skip tiling if:
    # 1. tile_size is set to <= 0 (explicit disable)
    # 2. Both dimensions are <= tile_size (image fits in one tile)
    # 3. Both dimensions are <= 1280px (image is medium-sized and GPU can process it in one go)
    # This prevents the GIL overhead of slicing, stitching, and parallel CPU queues.
    use_tiling = True
    if tile_size <= 0 or (h <= tile_size and w <= tile_size) or (h <= 1280 and w <= 1280 and tile_size >= 256):
        use_tiling = False

    if not use_tiling:
        print("⚡ Processing full image in one pass (no tiling)...")
        # Pad to multiple of 4 as Real-ESRGAN works best on multiples of 4
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        if pad_h > 0 or pad_w > 0:
            img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        else:
            img_padded = img
            
        inp = preprocess(img_padded)
        out = session.run([OUTPUT_NAME], {INPUT_NAME: inp})[0]
        
        print("=" * 60)
        print("Provider:", session.get_providers())
        print("dtype :", out.dtype)
        print("shape :", out.shape)
        print("min   :", np.min(out))
        print("max   :", np.max(out))
        print("mean  :", np.mean(out))
        print("nan   :", np.isnan(out).any())
        print("=" * 60)
        
        output = postprocess(out)
        
        if pad_h > 0 or pad_w > 0:
            output = output[:h * SCALE, :w * SCALE]
    else:
        print(f"🧩 Processing image using tiling (tile_size={tile_size})...")
        output = np.zeros((h * SCALE, w * SCALE, c), dtype=np.uint8)
        
        # Padding for natural edge blending
        img_padded = cv2.copyMakeBorder(img, tile_pad, tile_pad, tile_pad, tile_pad, cv2.BORDER_REFLECT_101)
        
        tasks = []
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                y1 = y
                y2 = min(y + tile_size, h)
                x1 = x
                x2 = min(x + tile_size, w)
                tasks.append((session, img_padded, x1, x2, y1, y2, tile_pad, SCALE))

        print(f"🧠 Processing {len(tasks)} tiles...")
        workers = min(len(tasks), NUM_THREADS)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(process_tile, tasks))

        for y1, y2, x1, x2, res in results:
            output[y1*SCALE:y2*SCALE, x1*SCALE:x2*SCALE] = res

    # Post-Processing
    if sharpen_amount > 0:
        kernel = np.array([[-sharpen_amount, -sharpen_amount, -sharpen_amount],
                           [-sharpen_amount, 1 + 8*sharpen_amount, -sharpen_amount],
                           [-sharpen_amount, -sharpen_amount, -sharpen_amount]])
        output = cv2.filter2D(output, -1, kernel)

    if contrast_alpha != 1.0 or brightness_beta != 0:
        output = cv2.convertScaleAbs(output, alpha=contrast_alpha, beta=brightness_beta)
        
    return output

# ================= RUNPOD HANDLER =================
def handler(job):
    """
    The handler function receives job requests from RunPod.
    """
    job_input = job.get("input", {})
    
    # 1. Fetch parameters
    image_url = job_input.get("image_url")
    image_b64 = job_input.get("image_b64")
    
    sharpen_val = job_input.get("sharpen_amount")
    sharpen_amount = float(sharpen_val) if sharpen_val is not None else 0.35
    
    contrast_val = job_input.get("contrast_alpha")
    contrast_alpha = float(contrast_val) if contrast_val is not None else 1.04
    
    brightness_val = job_input.get("brightness_beta")
    brightness_beta = int(brightness_val) if brightness_val is not None else 3
    
    tile_size_val = job_input.get("tile_size")
    tile_size = int(tile_size_val) if tile_size_val is not None else 256
    
    tile_pad_val = job_input.get("tile_pad")
    tile_pad = int(tile_pad_val) if tile_pad_val is not None else 16
    
    output_format = job_input.get("output_format", "jpeg") # jpeg or png
    
    # Validation
    if not image_url and not image_b64:
        return {"error": "Missing input: Please provide either 'image_url' or 'image_b64'"}
        
    try:
        # 2. Load input image
        if image_url:
            print(f"📥 Fetching image from URL: {image_url}")
            img = download_image_from_url(image_url)
        else:
            print("📥 Decoding base64 image...")
            img = decode_base64_image(image_b64)
            
        print(f"🖼 Image loaded. Shape: {img.shape}")
        
        # 3. Process image
        enhanced = enhance_image_logic(
            img,
            sharpen_amount=sharpen_amount,
            contrast_alpha=contrast_alpha,
            brightness_beta=brightness_beta,
            tile_size=tile_size,
            tile_pad=tile_pad
        )
        
        # 4. Encode and return output
        fmt = f".{output_format.lower()}"
        if fmt not in [".jpg", ".jpeg", ".png"]:
            fmt = ".jpg"
            
        output_b64 = encode_image_to_base64(enhanced, fmt)
        print("🎉 Enhancement complete. Returning base64 string.")
        
        return {
            "status": "success",
            "image_b64": output_b64,
            "width": enhanced.shape[1],
            "height": enhanced.shape[0]
        }
        
    except Exception as e:
        print(f"❌ Error during job execution: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
