# RunPod Serverless - Real-ESRGAN Image Upscaler

This directory contains the self-contained files required to package, test, and deploy the Real-ESRGAN upscaler model (`realesr_general_x4v3.onnx`) as a RunPod Serverless endpoint.

## Files Included

- `realesr_general_x4v3.onnx`: The exported ONNX model for high-performance image enhancement.
- `handler.py`: The entry point script that listens to RunPod serverless jobs and executes image processing.
- `requirements.txt`: Python package requirements.
- `Dockerfile`: Configuration for building the Docker container using a GPU/CUDA runtime environment.
- `test_input.json`: A sample request structure for testing.

---

## 1. Local Testing

To test the handler locally without building a Docker image, you need to set up the dependencies in your local environment.

### Step 1: Install requirements
Make sure you have python installed, and run:
```bash
pip install -r requirements.txt
```
*(Note: If you don't have a CUDA GPU locally, you can change `onnxruntime-gpu` in `requirements.txt` to `onnxruntime` to test on CPU, as the handler automatically falls back to CPU).*

### Step 2: Run local test
You can run the handler and feed it the `test_input.json` file using RunPod's CLI integration:
```bash
python handler.py --test_input @test_input.json
```
This will process the test image URL and output the result (including base64 encoded image data) to your terminal.

---

## 2. Docker Packaging

To deploy to RunPod, you need to package the app as a Docker image and push it to a container registry (like Docker Hub or GitHub Container Registry).

### Step 1: Build the image
Build the Docker image for the standard `linux/amd64` architecture:
```bash
docker build --platform linux/amd64 -t <your-docker-username>/realesrgan-runpod:v1 .
```

### Step 2: Test the Docker container locally
You can test the built image locally by running the container:
```bash
docker run -p 8000:8000 --gpus all <your-docker-username>/realesrgan-runpod:v1
```

### Step 3: Push the image
Push the built image to your registry:
```bash
docker push <your-docker-username>/realesrgan-runpod:v1
```

---

## 3. Deploying on RunPod Serverless

1. Log in to your **[RunPod Console](https://runpod.io)**.
2. Go to **Serverless** -> **Endpoints** -> Click **New Endpoint**.
3. Fill out the endpoint details:
   - **Endpoint Name**: `realesrgan-upscaler`
   - **Container Image**: `<your-docker-username>/realesrgan-runpod:v1`
   - **Container Registry Credentials**: (Optional, if using a private repository)
   - **Min Provisioned Workers**: `0` (for true scale-to-zero serverless, avoiding idle costs)
   - **Max Workers**: `3` (or higher depending on your traffic needs)
   - **GPU Type**: Select standard GPUs like RTX 4090 or L4 for optimal performance.
4. Click **Create**.

---

## 4. API Request & Response Formats

### Request Payload (JSON)

You can send a JSON payload with either `image_url` or `image_b64` (base64 string):

```json
{
  "input": {
    "image_url": "https://example.com/image.png",
    "sharpen_amount": 0.35,
    "contrast_alpha": 1.04,
    "brightness_beta": 3,
    "tile_size": 256,
    "tile_pad": 16,
    "output_format": "jpeg"
  }
}
```

### Response Payload (JSON)

```json
{
  "delayTime": 240,
  "executionTime": 850,
  "id": "job-id-12345",
  "output": {
    "status": "success",
    "image_b64": "/9j/4AAQSkZJRgABAQAAAQABAAD...",
    "width": 1024,
    "height": 1024
  },
  "status": "COMPLETED"
}
```
# runpod_enhance
