import os
import moviepy.editor as mp

VIDEO_ROOT = "videos"   # Folder where your raw videos are
OUTPUT_ROOT = "clips"   # Where you want WAV files saved

def convert_video_to_audio(video_path, output_path):
    try:
        video = mp.VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(output_path, codec='pcm_s16le')  # WAV format
        print(f"✅ Converted: {output_path}")
    except Exception as e:
        print(f"❌ Failed to convert {video_path}: {e}")

def batch_convert():
    categories = ["deception", "truthful"]
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for category in categories:
        input_dir = os.path.join(VIDEO_ROOT, category)
        output_dir = os.path.join(OUTPUT_ROOT, category)
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(input_dir):
            if filename.endswith((".mp4", ".mov", ".avi")):
                video_path = os.path.join(input_dir, filename)
                output_filename = os.path.splitext(filename)[0] + ".wav"
                output_path = os.path.join(output_dir, output_filename)
                convert_video_to_audio(video_path, output_path)

if __name__ == "__main__":
    batch_convert()