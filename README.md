# **Molmo2-HF-Demo**

> A Gradio-based demonstration for the AllenAI Molmo2-8B multimodal model, enabling image QA, multi-image pointing, video QA, and temporal tracking. Users upload images or videos, provide natural language prompts (e.g., "Point to the boats" or "Track the player"), and receive generated responses with visual annotations: red circles and labels for detected points on images/videos. Supports multi-image galleries and MP4 videos with frame sampling.

## Features

- **Multi-Image Processing**: Upload galleries for comparative QA or pointing tasks; extracts coordinates from model output and annotates with red ellipses and "target" labels.
- **Video Analysis**: Handles MP4 inputs for QA or tracking; samples frames, detects temporal points (timestamps), and overlays circles on annotated output videos.
- **Point/Track Extraction**: Parses `<points|tracks>` XML-like coords from responses; scales to original dimensions; supports IDs for multi-element detection.
- **Custom Theme**: OrangeRedTheme with gradients and enhanced typography for a modern interface.
- **Examples Integration**: Pre-loaded samples for quick testing (e.g., cat/dog pointing, football tracking).
- **Queueing Support**: Handles concurrent inferences with error display.
- **Efficient Inference**: Uses auto dtype/device_map; up to 2048 new tokens for verbose video responses.

---
<img width="1918" height="968" alt="Screenshot 2025-12-20 at 13-57-26 Molmo2 HF Demo - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/3830d8c4-1923-4b24-a920-fa80527bdd57" />


---


## Prerequisites

- Python 3.10 or higher.
- CUDA-compatible GPU (recommended for auto dtype; falls back to CPU).
- pip >= 23.0.0 (see pre-requirements.txt).
- Stable internet for initial model download (~8B params).

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/PRITHIVSAKTHIUR/Molmo2-HF-Demo.git
   cd Molmo2-HF-Demo
   ```

2. Install pre-requirements (for pip version):
   Create a `pre-requirements.txt` file with the following content, then run:
   ```
   pip install -r pre-requirements.txt
   ```

   **pre-requirements.txt content:**
   ```
   pip>=23.0.0
   ```

3. Install dependencies:
   Create a `requirements.txt` file with the following content, then run:
   ```
   pip install -r requirements.txt
   ```

   **requirements.txt content:**
   ```
   transformers==4.57.1
   huggingface_hub
   qwen-vl-utils
   pyvips-binary
   sentencepiece
   opencv-python
   torch==2.6.0
   docling-core
   molmo_utils
   python-docx
   torchvision
   supervision
   accelerate
   matplotlib
   pdf2image
   reportlab
   markdown
   requests
   pymupdf
   decord2
   hf_xet
   spaces
   pyvips
   pillow
   gradio
   einops
   httpx
   fpdf
   peft
   timm
   av
   ```

4. Start the application:
   ```
   python app.py
   ```
   The demo launches at `http://localhost:7860` (or the provided URL if using Spaces).

## Usage

Switch between "Images" and "Video" tabs.

### Images Tab
1. **Upload Gallery**: Select multiple images (e.g., via file paths).
2. **Enter Prompt**: E.g., "Point to the cats" or "Compare these two images."
3. **Run Analysis**: Click "Run Image Analysis."
4. **Output**: Text response + annotated gallery (red circles on pointed elements).

### Video Tab
1. **Upload Video**: Select an MP4 file.
2. **Enter Prompt**: E.g., "Track the football" or "Explain the video."
3. **Run Analysis**: Click "Run Video Analysis" (note: longer due to frame sampling).
4. **Output**: Text response + annotated MP4 (circles on tracked points).

### Examples

| Tab    | Example Inputs                          | Prompt Example                  |
|--------|-----------------------------------------|---------------------------------|
| Images | ["example-images/cat1.jpg", "example-images/cat2.jpg", "example-images/dog1.jpg"] | "Point to the cats."           |
| Images | ["example-images/compare1.jpg", "example-images/compare2.jpeg"] | "Compare these two images."    |
| Video  | "example-videos/sample_video.mp4"      | "Track the football."          |
| Video  | "example-videos/drink.mp4"             | "Explain the video."           |

## Troubleshooting

- **Model Loading Errors**: Verify transformers 4.57.1 and torch 2.6.0; check device_map="auto" for multi-GPU. Use `dtype=torch.float32` if bfloat16 fails.
- **Point Extraction Fails**: Ensure prompt requests "point" or "track"; regex handles `<points coords="...">` format. Console logs raw text.
- **Video Annotation Issues**: OpenCV requires MP4V codec; scale factors applied if metadata differs. Test with short clips.
- **Gallery Paths**: Use 'filepath' type; tuples handled for older Gradio.
- **OOM on GPU**: Reduce max_new_tokens or batch size; clear cache with `torch.cuda.empty_cache()`.
- **Utils Missing**: Install `molmo_utils` for video processing; `process_vision_info` samples frames.
- **UI Rendering**: Set `ssr_mode=True` if gradients fail; CSS for container max-width.

## Contributing

Contributions encouraged! Fork the repo, add examples or enhance parsing (e.g., for multi-video), and submit PRs with tests. Focus areas:
- Support for more modalities (e.g., audio).
- Custom annotation styles.
- Batch video processing.

Repository: [https://github.com/PRITHIVSAKTHIUR/Molmo2-HF-Demo.git](https://github.com/PRITHIVSAKTHIUR/Molmo2-HF-Demo.git)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

Built by Prithiv Sakthi. Report issues via the repository.
