#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wyszkuajnie wspolnego czasu dla RGB i Thermalnych,
Znalezienie MaxWidth i MaxHeight i rozciagniecie mniejszego do wspolnych rozmiarow,
Polaczenie do wspolnej macierzy uwzgledniajac najblizesz klatki miedzy soba( sprawdzanie po timestampach najblizszych sasiadow)
w wyniki czego trafiaja pod jeden indeks w tablicy NP o shape [N, H,W, C] 
N - liczba klatek
H,W - wysokosc, szerokosc,
C - RGBT ( 4 kanaly, RGB lub BGR ostatni kanal T:Thermal)
Zapisanie do pliku
macierzy oraz timestampow wspolnych
wczytanie takiego pliku 

data = np.load(npy_path, mmap_mode='r')  
# N - liczba klatek
# H, W - wysokosc, szerokosc obrazka
# 4 - R, G, B, T 
frames      = data["frames"]      # shape: (N, H, W, 4)  uint8 
timestamps  = data["timestamps"]  # shape: (N,)         float64 (sekundy od 1970)


"""

import argparse
import glob
import os
import re
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm

def load_ts_npy(path: str) -> np.ndarray:
    """
    Wczytuje tablice timestampow z .npy.
    Jesli wartosci wygladaja na milisekundy (>1e11), konwertuje na sekundy.
    """
    ts = np.load(path).astype(np.float64)
    if ts.size == 0:
        raise ValueError(f"Pusty plik z timestampami: {path}")
    # time.time() w sekundach - 1.7e9 (rok 2025); w ms - 1.7e12
    if ts.mean() > 1e11:
        ts /= 1000.0
    return ts


def resize_to_target(frame: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """ Dwuliniowo skaluje pojedyncza klatke do (H, W). """
    return cv2.resize(frame, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)


def find_nearest_idx(array: np.ndarray, value: float) -> int:
    """ Indeks elementu w rosnacej tablicy najblizszego podanej wartosci. """
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or abs(value - array[idx - 1]) < abs(value - array[idx])):
        return idx - 1
    return idx


def extract_batch_idx(filename: str) -> int:
    """ Sortuje 'frameblock_00023.npy' -> 23. """
    m = re.search(r"frameblock_(\d+)\.npy", filename)
    return int(m.group(1)) if m else -1


#  Glowna logika 
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Laczenie klatek RGB i termalnych w tablice 4-kanalowa (RGB+T)."
    )
    ap.add_argument("-i", "--input_dir",  required=False, help="Katalog z danymi wejsciowymi")
    ap.add_argument("-o", "--output_dir", required=False, help="Gdzie zapisac wynik .npz")
    ap.add_argument("--max_rgb_batches", type=int, default=30,
                    help="Limit paczek RGB do wczytania (None = wszystkie)")
    args = ap.parse_args()


    in_dir = r"C:\Users\admin\Desktop\Dataset\2025-06-22_18-45-24.447018"
    out_dir = r"C:\Users\admin\Desktop\Dataset\2025-06-22_18-45-24.447018\rgbt_combined"
    in_dir = os.path.abspath(in_dir)
    out_dir = os.path.abspath(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    #  Sciezki plikow
    rgb_folder = os.path.join(in_dir, "u235c")
    rgb_ts_path = os.path.join(in_dir,"u235c", "timestamps.npy")
    thermal_frames_path = os.path.join(in_dir, "boson", "video.npy")
    thermal_ts_path = os.path.join(in_dir, "boson", "timestamps.npy")
    output_path = os.path.join(out_dir, "rgb_thermal_combined_resized.npz")

    #  1.  RGB 
    rgb_files = sorted(
        glob.glob(os.path.join(rgb_folder, "frameblock_*.npy")),
        key=extract_batch_idx,
    )
    if args.max_rgb_batches is not None:
        rgb_files = rgb_files[: args.max_rgb_batches]

    print(f"[RGB]  Ladowanie timestampow: {rgb_ts_path}")
    all_rgb_ts = load_ts_npy(rgb_ts_path)

    rgb_frames_lst, rgb_ts_lst, frame_ptr = [], [], 0
    print(f"[RGB]  Ladowanie {len(rgb_files)} paczek ")
    for fname in tqdm(rgb_files, unit="batch"):
        arr = np.load(fname)               # (n, H, W, 3)
        n   = arr.shape[0]
        rgb_frames_lst.append(arr)
        rgb_ts_lst.append(all_rgb_ts[frame_ptr: frame_ptr + n])
        frame_ptr += n

    rgb_frames = np.concatenate(rgb_frames_lst, axis=0)
    rgb_ts     = np.concatenate(rgb_ts_lst,   axis=0)
    print(f"[INFO] RGB: klatek = {len(rgb_ts):,}   "
          f"ts {rgb_ts.min():.3f} - {rgb_ts.max():.3f}")

    #  2.  Termalne 
    print(f"[T]    Ladowanie klatek termalnych: {thermal_frames_path}")
    thermal_frames = np.load(thermal_frames_path)          # (M, H, W)

    print(f"[T]    Ladowanie timestampow:       {thermal_ts_path}")
    thermal_ts = load_ts_npy(thermal_ts_path)
    print(f"[INFO] TH:  klatek = {len(thermal_ts):,}   "
          f"ts {thermal_ts.min():.3f} - {thermal_ts.max():.3f}")

    #  3.  Wspolny zakres czasu 
    t_start = max(rgb_ts[0], thermal_ts[0])
    t_end   = min(rgb_ts[-1], thermal_ts[-1])
    print(f"[INFO] Wspolne okno: {t_start:.3f} - {t_end:.3f}  "
          f"(Roznica = {(t_end - t_start):.3f} s)")

    rgb_mask     = (rgb_ts     >= t_start) & (rgb_ts     <= t_end)
    thermal_mask = (thermal_ts >= t_start) & (thermal_ts <= t_end)
    print(f"[INFO] Po przycieciu: RGB {rgb_mask.sum():,}/{rgb_ts.size:,}   "
          f"TH {thermal_mask.sum():,}/{thermal_ts.size:,}")

    rgb_frames,  rgb_ts      = rgb_frames[rgb_mask],   rgb_ts[rgb_mask]
    thermal_frames, thermal_ts = thermal_frames[thermal_mask], thermal_ts[thermal_mask]

    #  4.  Dopasowanie nearest
    print("[ALIGN] Dopasowywanie termalnych do RGB ")
    idx_nearest = np.array([find_nearest_idx(thermal_ts, t) for t in rgb_ts],
                           dtype=np.int64)
    aligned_t = thermal_frames[idx_nearest]
    if aligned_t.ndim == 3:                       # (N, H, W) -> (N, H, W, 1)
        aligned_t = aligned_t[..., np.newaxis]

    #  5.  Skalowanie 
    tgt_h = max(rgb_frames.shape[1], aligned_t.shape[1])
    tgt_w = max(rgb_frames.shape[2], aligned_t.shape[2])
    print(f"[INFO] Rozmiary oryg.: RGB={rgb_frames.shape[2]}x{rgb_frames.shape[1]}  "
          f"TH={aligned_t.shape[2]}x{aligned_t.shape[1]}  -> target={tgt_w}x{tgt_h}")

    rgb_resized = np.empty((rgb_frames.shape[0], tgt_h, tgt_w, 3), dtype=rgb_frames.dtype)
    t_resized   = np.empty((aligned_t.shape[0], tgt_h, tgt_w, 1), dtype=aligned_t.dtype)

    for i in tqdm(range(rgb_frames.shape[0]), desc="Resizing", unit="frame"):
        rgb_resized[i]        = resize_to_target(rgb_frames[i],        (tgt_h, tgt_w))
        t_resized[i, ..., 0]  = resize_to_target(aligned_t[i, ..., 0], (tgt_h, tgt_w))

    #  6.  Zapis 
    combined = np.concatenate([rgb_resized, t_resized], axis=-1)      # (N, H, W, 4)
    np.savez(output_path, frames=combined, timestamps=rgb_ts)
    print(f"[SAVE] Zapisano {combined.shape}  -> {output_path}")


if __name__ == "__main__":
    main()
