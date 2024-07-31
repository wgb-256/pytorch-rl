# utils.py
import imageio

def create_gif(frames, gif_path):
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"GIF saved: {gif_path}")