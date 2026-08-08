from pathlib import Path

content = r'''import torch
import torchvision.transforms.functional as T
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
import cv2
import numpy as np
import warnings

warnings.filterwarnings(
    "ignore",
    message="xFormers is available.*",
    category=UserWarning,
    module=r"dinov2\.layers\..*",
)

dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval()


def preprocess(frames):
    frames = frames.float() / 255.0
    return torch.stack([
        T.resize(x, [224, 224], InterpolationMode.BICUBIC, antialias=True)
        for x in frames
    ])


def decode_clip(video_path, T=24, window_sec=2.0, center_time=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if total <= 0:
        cap.release()
        raise ValueError(f"Could not read frame count: {video_path}")

    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    duration = total / fps

    if center_time is None:
        center_time = duration / 2.0

    half = window_sec / 2.0
    start_time = max(0.0, center_time - half)
    end_time = min(duration, center_time + half)

    start_frame = int(start_time * fps)
    end_frame = max(start_frame + 1, int(end_time * fps))
    end_frame = min(end_frame, total)

    frame_indices = np.linspace(start_frame, end_frame - 1, num=T).astype(int)

    frames = []
    last_rgb = None

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()

        if not ret:
            if last_rgb is not None:
                rgb = last_rgb
            else:
                continue
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_rgb = rgb

        chw = np.transpose(rgb, (2, 0, 1))
        frames.append(chw)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Could not decode any frames: {video_path}")

    while len(frames) < T:
        frames.append(frames[-1])

    arr = np.stack(frames[:T], axis=0)
    return torch.from_numpy(arr)


def extract_pixel_embeddings(video_paths, T=24, window_sec=2.0):
    outs = []
    for p in video_paths:
        try:
            clip = decode_clip(p, T=T, window_sec=window_sec)
            preprocess_clip = preprocess(clip)
            outs.append(preprocess_clip.flatten(1))
        except Exception as e:
            print(f"Error decoding clip: {e}")

    if len(outs) == 0:
        raise RuntimeError("No videos could be decoded.")

    return torch.stack(outs, dim=0)


def extract_dinov2_embeddings(video_paths, device=None, T=24, window_sec=2.0):
    device = device or torch.device("cpu")
    model = dinov2.to(device)

    outs = []
    for p in video_paths:
        try:
            clip = decode_clip(p, T=T, window_sec=window_sec)
            preprocess_clip = preprocess(clip)
            outs.append(preprocess_clip)
        except Exception as e:
            print(f"Error decoding clip: {e}")

    if len(outs) == 0:
        raise RuntimeError("No videos could be decoded.")

    batch = torch.cat(outs, dim=0).to(device)

    with torch.no_grad():
        feats = model.forward_features(batch)
        cls = feats["x_norm_clstoken"].unsqueeze(1)
        patches = feats["x_norm_patchtokens"]
        tokens = torch.cat([cls, patches], dim=1)
        Z_flat = tokens.flatten(1)
        Z = Z_flat.view(len(outs), outs[0].shape[0], -1)
        return Z


def compute_temporal_geometry(Z):
    delta = Z[:, 1:, :] - Z[:, :-1, :]
    d = delta.norm(dim=-1)
    cos = F.cosine_similarity(delta[:, :-1, :], delta[:, 1:, :], dim=-1)
    theta = torch.rad2deg(torch.acos(cos.clamp(-1, 1)))
    return d, theta


def moment4(x):
    mu = x.mean(dim=-1)
    mn = x.amin(dim=-1)
    mx = x.amax(dim=-1)
    var = x.var(dim=-1, unbiased=False)
    return mu, mn, mx, var


def features_from_Z(Z):
    d, t = compute_temporal_geometry(Z)
    d7 = d[:, :7]
    t6 = t[:, :6]
    mu_d, mn_d, mx_d, var_d = moment4(d)
    mu_t, mn_t, mx_t, var_t = moment4(t)
    stats = torch.stack([mu_d, mn_d, mx_d, var_d, mu_t, mn_t, mx_t, var_t], dim=1)
    return torch.cat([d7, t6, stats], dim=1)
'''

Path("dinov2_features.py").write_text(content, encoding="utf-8")
print("rewrote dinov2_features.py with OpenCV decoder")