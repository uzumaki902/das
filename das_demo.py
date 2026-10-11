"""
DAS processing demo: waterfall, bandpass filter, spectrogram, f-k plot.

Usage:
    python das_demo.py                                  -> synthetic data (test)
    python das_demo.py file.h5 --inspect                -> print file structure
    python das_demo.py file.h5 --fs 200 --spacing 2     -> real data (OOI OptaSense, FS200Hz files)
    python das_demo.py file.h5 --fs 200 --spacing 2 --ch 20000 22000 --seconds 60
    Options: --dataset NAME  --channels-first  --lo 1 --hi 15

Install: pip install numpy scipy matplotlib h5py
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, spectrogram


# ---------- 1. LOAD DATA ----------
def inspect_file(path):
    import h5py
    with h5py.File(path, "r") as f:
        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"DATASET {name}  shape={obj.shape}  dtype={obj.dtype}")
            else:
                print(f"group   {name}")
        f.visititems(show)
        for k, v in f.attrs.items():
            print(f"attr {k} = {v}")


def load_real(path, ch0, ch1, seconds, fs, dataset=None, channels_first=False):
    """Read ONLY a slice of the file (files are huge).
    Returns array shaped (time, channels)."""
    import h5py
    with h5py.File(path, "r") as f:
        if dataset is None:
            best = {"name": None, "size": 0}
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.ndim == 2 and obj.size > best["size"]:
                    best["name"], best["size"] = name, obj.size
            f.visititems(visit)
            dataset = best["name"]
        ds = f[dataset]
        print(f"Using dataset: {dataset}, shape={ds.shape}")
        nt = int(seconds * fs) if seconds else None
        if channels_first:
            ch1 = min(ch1, ds.shape[0]) if ch1 else ds.shape[0]
            data = ds[ch0:ch1, :nt].T
        else:
            ch1 = min(ch1, ds.shape[1]) if ch1 else ds.shape[1]
            data = ds[:nt, ch0:ch1]
    return np.asarray(data, dtype=np.float32)


def make_synthetic(fs=200, duration=60, n_ch=400, spacing=10.0):
    """Synthetic DAS record: noise + moving ship + earthquake-like arrival."""
    rng = np.random.default_rng(0)
    t = np.arange(int(fs * duration)) / fs
    data = rng.normal(0, 1.0, (len(t), n_ch)).astype(np.float32)
    x = np.arange(n_ch) * spacing  # metres along cable

    # Ship: low-frequency hum that moves along the cable (~5 m/s)
    for i, xi in enumerate(x):
        t0 = xi / 5.0
        env = np.exp(-((t - t0) ** 2) / (2 * 4.0 ** 2))
        data[:, i] += 6 * env * np.sin(2 * np.pi * 3.0 * t)

    # Earthquake-like arrival: fast wave (~3000 m/s) with a decaying burst
    for i, xi in enumerate(x):
        t0 = 20 + xi / 3000.0
        m = t > t0
        data[m, i] += 8 * np.exp(-(t[m] - t0) / 2.0) * np.sin(2 * np.pi * 8 * (t[m] - t0))
    return data, fs, spacing


# ---------- 2. PROCESSING ----------
def bandpass(data, fs, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, data, axis=0)


def preprocess(data):
    data = data - data.mean(axis=0)            # remove DC offset per channel
    data = data / (data.std(axis=0) + 1e-9)    # channel normalization
    return data


# ---------- 3. PLOTS ----------
def plot_waterfall(data, fs, spacing, title, fname):
    extent = [0, data.shape[1] * spacing / 1000, data.shape[0] / fs, 0]
    vmax = np.percentile(np.abs(data), 99)
    plt.figure(figsize=(8, 6))
    plt.imshow(data, aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax, extent=extent)
    plt.xlabel("Distance along cable (km)")
    plt.ylabel("Time (s)")
    plt.title(title)
    plt.colorbar(label="Normalized strain rate")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def plot_spectrogram(trace, fs, ch, fname):
    f, t, S = spectrogram(trace, fs=fs, nperseg=256, noverlap=192)
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(t, f, 10 * np.log10(S + 1e-12), shading="gouraud")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Spectrogram, channel {ch}")
    plt.colorbar(label="Power (dB)")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def plot_fk(data, fs, spacing, fname):
    F = np.fft.fftshift(np.fft.fft2(data))
    P = 20 * np.log10(np.abs(F) + 1e-9)
    freqs = np.fft.fftshift(np.fft.fftfreq(data.shape[0], 1 / fs))
    ks = np.fft.fftshift(np.fft.fftfreq(data.shape[1], spacing))
    plt.figure(figsize=(6, 5))
    plt.imshow(P, aspect="auto", origin="lower",
               extent=[ks[0], ks[-1], freqs[0], freqs[-1]], cmap="viridis")
    plt.ylim(0, fs / 2)
    plt.xlabel("Wavenumber (1/m)")
    plt.ylabel("Frequency (Hz)")
    plt.title("f-k spectrum")
    plt.colorbar(label="Amplitude (dB)")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


# ---------- 4. MAIN ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", help="HDF5 data file (omit for synthetic test)")
    ap.add_argument("--fs", type=float, default=200.0, help="sampling rate in Hz")
    ap.add_argument("--spacing", type=float, default=2.0, help="channel spacing in m")
    ap.add_argument("--ch", type=int, nargs=2, default=[0, 2000], metavar=("START", "END"))
    ap.add_argument("--seconds", type=float, default=60, help="seconds of data to read")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--channels-first", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--lo", type=float, default=1.0)
    ap.add_argument("--hi", type=float, default=15.0)
    a = ap.parse_args()

    if a.file and a.inspect:
        inspect_file(a.file)
        sys.exit(0)
    if a.file:
        data = load_real(a.file, a.ch[0], a.ch[1], a.seconds, a.fs, a.dataset, a.channels_first)
        fs, spacing, source = a.fs, a.spacing, "real data"
    else:
        data, fs, spacing = make_synthetic()
        source = "synthetic data"
    print(f"Loaded {source}: {data.shape[0]} samples x {data.shape[1]} channels, fs={fs} Hz, spacing={spacing} m")

    data = preprocess(data)
    plot_waterfall(data, fs, spacing, "Waterfall: raw (normalized)", "1_waterfall_raw.png")

    filt = bandpass(data, fs, a.lo, min(a.hi, fs / 2 - 1))
    plot_waterfall(filt, fs, spacing, f"Waterfall: bandpass {a.lo}-{a.hi} Hz", "2_waterfall_filtered.png")

    ch = data.shape[1] // 2
    plot_spectrogram(data[:, ch], fs, ch, "3_spectrogram.png")
    plot_fk(filt, fs, spacing, "4_fk_spectrum.png")
    print("Saved: 1_waterfall_raw.png, 2_waterfall_filtered.png, 3_spectrogram.png, 4_fk_spectrum.png")