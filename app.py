import gradio as gr
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image, ImageDraw
import numpy as np
import spaces
import cv2
import re
import os
from molmo_utils import process_vision_info

from typing import Iterable
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.orange_red = colors.Color(
    name="orange_red",
    c50="#FFF0E5",
    c100="#FFE0CC",
    c200="#FFC299",
    c300="#FFA366",
    c400="#FF8533",
    c500="#FF4500",
    c600="#E63E00",
    c700="#CC3700",
    c800="#B33000",
    c900="#992900",
    c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()

MODEL_ID = "allenai/Molmo2-8B"

print(f"Loading {MODEL_ID}...")
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto"
)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto"
)
print("Model loaded successfully.")

COORD_REGEX = re.compile(rf"<(?:points|tracks).*? coords=\"([0-9\t:;, .]+)\"/?>")
FRAME_REGEX = re.compile(rf"(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)")
POINTS_REGEX = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")

def _points_from_num_str(text, image_w, image_h):
    for points in POINTS_REGEX.finditer(text):
        ix, x, y = points.group(1), points.group(2), points.group(3)
        # our points format assume coordinates are scaled by 1000
        x, y = float(x)/1000*image_w, float(y)/1000*image_h
        if 0 <= x <= image_w and 0 <= y <= image_h:
            yield ix, x, y

def extract_multi_image_points(text, image_w, image_h, extract_ids=False):
    """Extract pointing coordinates for images."""
    all_points = []
    # Handle list of dimensions for multi-image
    if isinstance(image_w, (list, tuple)) and isinstance(image_h, (list, tuple)):
        assert len(image_w) == len(image_h)
        diff_res = True
    else:
        diff_res = False
        
    for coord in COORD_REGEX.finditer(text):
        for point_grp in FRAME_REGEX.finditer(coord.group(1)):
            # For images, frame_id corresponds to the image index (1-based in text usually, but we need to check)
            frame_id = int(point_grp.group(1)) if diff_res else float(point_grp.group(1))
            
            if diff_res:
                # bounds check
                idx = int(frame_id) - 1
                if 0 <= idx < len(image_w):
                    w, h = (image_w[idx], image_h[idx])
                else:
                    continue
            else:
                w, h = (image_w, image_h)
                
            for idx, x, y in _points_from_num_str(point_grp.group(2), w, h):
                if extract_ids:
                    all_points.append((frame_id, idx, x, y))
                else:
                    all_points.append((frame_id, x, y))
    return all_points

def extract_video_points(text, image_w, image_h, extract_ids=False):
    """Extract video pointing coordinates (t, x, y)."""
    all_points = []
    for coord in COORD_REGEX.finditer(text):
        for point_grp in FRAME_REGEX.finditer(coord.group(1)):
            frame_id = float(point_grp.group(1)) # This is usually timestamp in seconds or frame index
            w, h = (image_w, image_h)
            for idx, x, y in _points_from_num_str(point_grp.group(2), w, h):
                if extract_ids:
                    all_points.append((frame_id, idx, x, y))
                else:
                    all_points.append((frame_id, x, y))
    return all_points

def draw_points_on_images(images, points):
    """Draws points on a list of PIL Images."""
    annotated_images = [img.copy() for img in images]
    
    # Points format: [(image_index_1_based, x, y), ...]
    for p in points:
        img_idx = int(p[0]) - 1 # Convert 1-based index to 0-based
        x, y = p[1], p[2]
        
        if 0 <= img_idx < len(annotated_images):
            draw = ImageDraw.Draw(annotated_images[img_idx])
            r = 10 # radius
            # Draw a red circle with outline
            draw.ellipse((x-r, y-r, x+r, y+r), outline="red", width=3)
            draw.text((x+r, y), "target", fill="red")
            
    return annotated_images

def draw_points_on_video(video_path, points, original_width, original_height):
    """
    Draws points on video. 
    points format: [(timestamp_seconds, x, y), ...]
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Scale factor if Molmo processed a resized version vs original video file
    # Note: Molmo points are usually scaled to the dimensions passed in metadata.
    # If the video metadata passed to Molmo matches the file, x/y are correct for the file.
    scale_x = vid_w / original_width
    scale_y = vid_h / original_height
    
    # Organize points by frame index for faster lookup
    # Molmo outputs timestamps. frame_idx = timestamp * fps
    points_by_frame = {}
    for t, x, y in points:
        f_idx = int(round(t * fps))
        if f_idx not in points_by_frame:
            points_by_frame[f_idx] = []
        points_by_frame[f_idx].append((x * scale_x, y * scale_y))

    # Output setup
    output_path = "annotated_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (vid_w, vid_h))
    
    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Draw points if they exist for this frame (or nearby frames to persist visualization slightly)
        # Simple approach: Exact frame match
        if current_frame in points_by_frame:
            for px, py in points_by_frame[current_frame]:
                cv2.circle(frame, (int(px), int(py)), 10, (0, 0, 255), -1)
                cv2.circle(frame, (int(px), int(py)), 12, (255, 255, 255), 2)
        
        out.write(frame)
        current_frame += 1
        
    cap.release()
    out.release()
    return output_path

@spaces.GPU
def process_images(user_text, input_images):
    if not input_images:
        return "Please upload at least one image.", None
    
    # input_images from Gradio Gallery is a list of (path, caption) tuples 
    # OR a list of paths depending on type. We requested 'filepath' type in Gradio.
    pil_images = []
    for img_path in input_images:
        # If type='filepath' in Gallery, img_path is just the string path
        # If using old gradio versions it might be a tuple.
        if isinstance(img_path, tuple):
            img_path = img_path[0]
        pil_images.append(Image.open(img_path).convert("RGB"))

    # Construct messages
    content = [dict(type="text", text=user_text)]
    for img in pil_images:
        content.append(dict(type="image", image=img))
        
    messages = [{"role": "user", "content": content}]

    # Process inputs
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)

    generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Check for points
    widths = [img.width for img in pil_images]
    heights = [img.height for img in pil_images]
    
    points = extract_multi_image_points(generated_text, widths, heights)
    
    output_gallery = pil_images
    if points:
        output_gallery = draw_points_on_images(pil_images, points)
        
    return generated_text, output_gallery

@spaces.GPU
def process_video(user_text, video_path):
    if not video_path:
        return "Please upload a video.", None

    # Construct messages
    # Note: Molmo expects a URL or a path it can read. 
    messages = [
        {
            "role": "user",
            "content": [
                dict(type="text", text=user_text),
                dict(type="video", video=video_path),
            ],
        }
    ]

    # Process Vision Info (Molmo Utils)
    # This samples the video and prepares tensors
    _, videos, video_kwargs = process_vision_info(messages)
    videos, video_metadatas = zip(*videos)
    videos, video_metadatas = list(videos), list(video_metadatas)

    # Chat Template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Inputs
    inputs = processor(
        videos=videos,
        video_metadata=video_metadatas,
        text=text,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=2048)

    generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Point/Track processing
    vid_meta = video_metadatas[0] # Assuming single video
    points = extract_video_points(generated_text, image_w=vid_meta["width"], image_h=vid_meta["height"])
    
    annotated_video_path = None
    if points:
        print(f"Found {len(points)} points/track-coords. Annotating video...")
        annotated_video_path = draw_points_on_video(
            video_path, 
            points, 
            original_width=vid_meta["width"], 
            original_height=vid_meta["height"]
        )
    
    # Return original video if no points found, otherwise annotated
    out_vid = annotated_video_path if annotated_video_path else video_path
    
    return generated_text, out_vid

css="""
#col-container {
    margin: 0 auto;
    max-width: 960px;
}
#main-title h1 {font-size: 2.1em !important;}
"""

with gr.Blocks() as demo:
    gr.Markdown("# **Molmo2 HF Demo🖥️**", elem_id="main-title")
    gr.Markdown("Perform multi-image QA, pointing, general video QA, and tracking using the [Molmo2](https://huggingface.co/allenai/Molmo2-8B) multimodal model.")

    with gr.Tabs():
        with gr.Tab("Images (QA & Pointing)"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Gallery(label="Input Images", type="filepath", height=400)
                    img_prompt = gr.Textbox(label="Prompt", placeholder="e.g. 'Describe this' or 'Point to the boats'")
                    img_btn = gr.Button("Run Image Analysis", variant="primary")
                
                with gr.Column():
                    img_text_out = gr.Textbox(label="Generated Text", interactive=True, lines=5)
                    img_out = gr.Gallery(label="Annotated Images (Pointing if applicable)", height=378)
                    
            gr.Examples(
                examples=[
                    [["example-images/compare1.jpg", "example-images/compare2.jpeg"], "Compare these two images."],
                    [["example-images/cat1.jpg", "example-images/cat2.jpg", "example-images/dog1.jpg"], "Point to the cats."],
                    [["example-images/candy.JPG"], "Point to all the candies."],
                    [["example-images/premium_photo-1691752881339-d78da354ee7e.jpg"], "Point to the girls."],
                    ],
                inputs=[img_input, img_prompt],
                label="Image Examples"
            )
            img_btn.click(
                fn=process_images,
                inputs=[img_prompt, img_input],
                outputs=[img_text_out, img_out]
            )
        
        with gr.Tab("Video (QA, Pointing & Tracking)"):
            gr.Markdown("**Note:** Video processing takes longer as frames are sampled.")
            with gr.Row():
                with gr.Column():
                    vid_input = gr.Video(label="Input Video", format="mp4", height=400)
                    vid_prompt = gr.Textbox(label="Prompt", placeholder="e.g. 'What is happening?' or 'Track the player'")
                    vid_btn = gr.Button("Run Video Analysis", variant="primary")
                
                with gr.Column():
                    vid_text_out = gr.Textbox(label="Generated Text", interactive=True, lines=5)
                    vid_out = gr.Video(label="Output Video (Annotated if applicable)", height=378)
                    
            gr.Examples(
                examples=[
                    ["example-videos/sample_video.mp4", "Track the football."],
                    ["example-videos/drink.mp4", "Explain the video."],
                    ],
                    inputs=[vid_input, vid_prompt],
                    label="Video Examples"
                )
            vid_btn.click(
                fn=process_video,
                inputs=[vid_prompt, vid_input],
                outputs=[vid_text_out, vid_out]
            )

if __name__ == "__main__":
    demo.launch(theme=orange_red_theme, css=css, mcp_server=True, ssr_mode=False, show_error=True)