"""
throat2air_sklearn.py

喉マイク音声を空気マイクに近づける回帰モデル（scikit-learn版）。
- 特徴量: 40次ログMel + Δ + ΔΔ（=120次）に文脈スタック（±2フレーム）→ 120*5=600次元
- 目的値: 空気マイクの40次ログMel
- モデル: StandardScaler → MLPRegressor(2層)（RidgeCVに切替可）
- 再合成: Griffin-Lim（librosa）または事前学習済みHiFi-GAN（PyTorch）

前提:
- 16kHz/monoで処理（読み込み時にリサンプリング）
- ディレクトリ:
    data/throat/*.wav  (喉マイク 100ファイル; 1フレーズごと)
    data/air/*.wav     (空気マイク 100ファイル; 同名ファイルでペア)
- ファイル名は同名対応（例: 001.wav が喉と空気でペア）

使い方:
    pip install numpy scipy librosa scikit-learn joblib tqdm
    # Use default model path
    python throat2air_sklearn.py --train
    python throat2air_sklearn.py --enhance input.wav output.wav

    # Use custom model path
    python throat2air_sklearn.py --train --model-path models/my_custom_model.joblib
    python throat2air_sklearn.py --enhance recordings/External_Microphone/20251021232813.wav recordings/External_Microphone/20251021232813_enhanced.wav --model-path models/throat2air_External_Microphone_20251023_mlp.joblib

    # Example with all custom parameters
    python throat2air_sklearn.py --train --throat-dir recordings/GHW_USB_AUDIO --air-dir recordings/Marantz_Umpire_Mic --model-path models/custom_throat2air.joblib

    # Bulk enhance all WAV files in a directory
    python throat2air_sklearn.py --bulk data/throat enhanced_output

    # With custom model
    python throat2air_sklearn.py --bulk recordings/throat_mics enhanced_results --model-path models/custom_model.joblib

    # Complete workflow example
    python throat2air_sklearn.py --train --throat-dir recordings/throat --air-dir recordings/air --model-path models/my_model.joblib
    python throat2air_sklearn.py --bulk recordings/External_Microphone recordings/External_Microphone_Enhanced --model-path models/throat2air_External_Microphone_20251023_mlp.joblib  
    python throat2air_sklearn.py --bulk recordings/GHW_USB_AUDIO recordings/GHW_USB_AUDIO_Enhanced --model-path models/throat2air_GHW_USB_AUDIO_20251023_mlp.joblib

    # HiFi-GAN vocoder example
    ## train the model
    python throat2air_sklearn_hifi-gan.py \
        --train \
        --model-path models/throat2air_External_Microphone_LJcompatible_20251101.joblib \
        --sr 22050 --n-mels 80 --fft 1024 --hop 256 --win 1024 --fmin 0 --fmax 8000 --power 1.0 \
        --throat-dir recordings/External_Microphone --air-dir recordings/Marantz_Umpire_Mic
    python throat2air_sklearn_hifi-gan.py \
        --train \
        --model-path models/throat2air_GHW_USB_AUDIO_LJcompatible_20251101.joblib \
        --sr 22050 --n-mels 80 --fft 1024 --hop 256 --win 1024 --fmin 0 --fmax 8000 --power 1.0 \
        --throat-dir recordings/GHW_USB_AUDIO --air-dir recordings/Marantz_Umpire_Mic
    ## enhance with HiFi-GAN vocoder
    python throat2air_sklearn_hifi-gan.py \
        --bulk recordings/External_Microphone recordings/External_Microphone_Enhanced_LJ_V1 \
        --model-path models/throat2air_External_Microphone_LJcompatible_20251101.joblib \
        --vocoder hifigan \
        --hifigan-checkpoint pretrained/LJ_V1/generator_v1 \
        --hifigan-config pretrained/LJ_V1/config.json \
        --sr 22050 --n-mels 80 --fft 1024 --hop 256 --win 1024 --fmin 0 --fmax 8000 --power 1.0    
    python throat2air_sklearn_hifi-gan.py \
        --bulk recordings/GHW_USB_AUDIO recordings/GHW_USB_AUDIO_Enhanced_LJ_V1 \
        --model-path models/throat2air_GHW_USB_AUDIO_LJcompatible_20251101.joblib \
        --vocoder hifigan \
        --hifigan-checkpoint pretrained/LJ_V1/generator_v1 \
        --hifigan-config pretrained/LJ_V1/config.json \
        --sr 22050 --n-mels 80 --fft 1024 --hop 256 --win 1024 --fmin 0 --fmax 8000 --power 1.0    
    ### VCTK_V1
    python throat2air_sklearn_hifi-gan.py \
        --bulk recordings/External_Microphone recordings/External_Microphone_Enhanced_VCTK_V1 \
        --model-path models/throat2air_External_Microphone_LJcompatible_20251101.joblib \
        --vocoder hifigan \
        --hifigan-checkpoint pretrained/VCTK_V1/generator_v1 \
        --hifigan-config pretrained/VCTK_V1/config.json \
        --sr 22050 --n-mels 80 --fft 1024 --hop 256 --win 1024 --fmin 0 --fmax 8000 --power 1.0    
    python throat2air_sklearn_hifi-gan.py \
        --bulk recordings/GHW_USB_AUDIO recordings/GHW_USB_AUDIO_Enhanced_VCTK_V1 \
        --model-path models/throat2air_GHW_USB_AUDIO_LJcompatible_20251101.joblib \
        --vocoder hifigan \
        --hifigan-checkpoint pretrained/VCTK_V1/generator_v1 \
        --hifigan-config pretrained/VCTK_V1/config.json \
        --sr 22050 --n-mels 80 --fft 1024 --hop 256 --win 1024 --fmin 0 --fmax 8000 --power 1.0    
    ### UNIVERSAL_V1
    python throat2air_sklearn_hifi-gan.py \
        --bulk recordings/External_Microphone recordings/External_Microphone_Enhanced_UNIVERSAL_V1 \
        --model-path models/throat2air_External_Microphone_LJcompatible_20251101.joblib \
        --vocoder hifigan \
        --hifigan-checkpoint pretrained/UNIVERSAL_V1/g_02500000 \
        --hifigan-config pretrained/UNIVERSAL_V1/config.json \
        --sr 22050 --n-mels 80 --fft 1024 --hop 256 --win 1024 --fmin 0 --fmax 8000 --power 1.0    
    python throat2air_sklearn_hifi-gan.py \
        --bulk recordings/GHW_USB_AUDIO recordings/GHW_USB_AUDIO_Enhanced_UNIVERSAL_V1 \
        --model-path models/throat2air_GHW_USB_AUDIO_LJcompatible_20251101.joblib \
        --vocoder hifigan \
        --hifigan-checkpoint pretrained/UNIVERSAL_V1/g_02500000 \
        --hifigan-config pretrained/UNIVERSAL_V1/config.json \
        --sr 22050 --n-mels 80 --fft 1024 --hop 256 --win 1024 --fmin 0 --fmax 8000 --power 1.0    
        

"""

import os
import glob
import argparse
from typing import Optional
import numpy as np
import librosa
import librosa.feature
import soundfile as sf
from tqdm import tqdm
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

SR = 16000
N_MELS = 40
FFT = 1024
HOP = 256
WIN = 1024
FMIN = 50
FMAX = 7600  # 16kHzの上限に合わせつつ子音域もカバー
POWER = 2.0 # パワースペクトル由来
N_ITER_GRIFFINLIM = 32

# 文脈設定（±CONTEXTのフレームをスタック）
CONTEXT = 2

MODEL_PATH = "models/throat2air_mlp.joblib"
VOCODER_GRIFFINLIM = "griffinlim"
VOCODER_HIFIGAN = "hifigan"


def override_signal_params(
    sr: Optional[int] = None,
    n_mels: Optional[int] = None,
    fft: Optional[int] = None,
    hop: Optional[int] = None,
    win: Optional[int] = None,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    power: Optional[float] = None,
    griffinlim_iters: Optional[int] = None,
):
    """Override global spectral parameters to match a pretrained HiFi-GAN setup."""
    global SR, N_MELS, FFT, HOP, WIN, FMIN, FMAX, POWER, N_ITER_GRIFFINLIM

    if sr:
        SR = sr
    if n_mels:
        N_MELS = n_mels
    if fft:
        FFT = fft
    if hop:
        HOP = hop
    if win:
        WIN = win
    if fmin is not None:
        FMIN = fmin
    if fmax:
        FMAX = fmax
    if power:
        POWER = power
    if griffinlim_iters:
        N_ITER_GRIFFINLIM = griffinlim_iters

def load_audio_align_pair(throat_path, air_path):
    """喉と空気の時間ズレを簡易整列（RMS包絡の相互相関でラグ推定）して返す。"""
    y_t, _ = librosa.load(throat_path, sr=SR, mono=True)
    y_a, _ = librosa.load(air_path, sr=SR, mono=True)

    # 粗い前処理（プリエンファシスは省略可）
    y_t = librosa.util.normalize(y_t)
    y_a = librosa.util.normalize(y_a)

    # 簡易ラグ推定：RMS包絡の相互相関（ダウンサンプルで軽量化）
    hop_env = 512
    env_t = librosa.feature.rms(y=y_t, frame_length=2048, hop_length=hop_env).flatten()
    env_a = librosa.feature.rms(y=y_a, frame_length=2048, hop_length=hop_env).flatten()
    env_t = (env_t - env_t.mean()) / (env_t.std() + 1e-8)
    env_a = (env_a - env_a.mean()) / (env_a.std() + 1e-8)

    corr = np.correlate(env_t, env_a, mode='full')
    lag_frames = np.argmax(corr) - (len(env_a) - 1)
    lag_samples = int(lag_frames * hop_env)

    if lag_samples > 0:
        y_t = y_t[lag_samples:]
        y_a = y_a[:len(y_t)]
    elif lag_samples < 0:
        y_a = y_a[-lag_samples:]
        y_t = y_t[:len(y_a)]
    L = min(len(y_t), len(y_a))
    y_t = y_t[:L]
    y_a = y_a[:L]
    return y_t, y_a

def feat_logmel(y):
    """ログMel + Δ + ΔΔ を返す（[T, 120]）。"""
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=FFT, hop_length=HOP, win_length=WIN,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=POWER
    )
    logmel = np.log(np.maximum(S, 1e-10))  # 自然対数
    d1 = librosa.feature.delta(logmel, order=1)
    d2 = librosa.feature.delta(logmel, order=2)
    feat = np.concatenate([logmel, d1, d2], axis=0)  # [3*N_MELS, T]
    return feat.T  # [T, 120]

def stack_context(X, left=CONTEXT, right=CONTEXT):
    """時間文脈をスタック（端はパディング）: [T, D*(1+left+right)]"""
    T, D = X.shape
    pads = []
    for s in range(-left, right+1):
        if s < 0:
            pad = np.vstack([np.repeat(X[[0]], -s, axis=0), X[:T+s]])
        elif s > 0:
            pad = np.vstack([X[s:], np.repeat(X[[-1]], s, axis=0)])
        else:
            pad = X
        pads.append(pad)
    return np.hstack(pads)


class GriffinLimVocoder:
    """Baseline vocoder that mirrors the previous Griffin-Lim behaviour."""

    def __call__(self, log_mel: np.ndarray) -> np.ndarray:
        mel_power = np.exp(log_mel).T
        audio = librosa.feature.inverse.mel_to_audio(
            M=mel_power,
            sr=SR,
            n_fft=FFT,
            hop_length=HOP,
            win_length=WIN,
            fmin=FMIN,
            fmax=FMAX,
            power=POWER,
            n_iter=N_ITER_GRIFFINLIM,
        )
        return audio.astype(np.float32)


def create_vocoder(
    name: str,
    hifigan_checkpoint: Optional[str] = None,
    hifigan_config: Optional[str] = None,
    device: Optional[str] = None,
):
    if name == VOCODER_GRIFFINLIM:
        return GriffinLimVocoder()
    if name == VOCODER_HIFIGAN:
        if not hifigan_checkpoint:
            raise ValueError("--hifigan-checkpoint is required when --vocoder hifigan is specified")
        if not os.path.isfile(hifigan_checkpoint):
            raise FileNotFoundError(f"HiFi-GAN checkpoint not found: {hifigan_checkpoint}")
        if hifigan_config and not os.path.isfile(hifigan_config):
            raise FileNotFoundError(f"HiFi-GAN config not found: {hifigan_config}")
        from hifigan_vocoder import HiFiGANVocoder

        return HiFiGANVocoder(
            checkpoint_path=hifigan_checkpoint,
            config_path=hifigan_config,
            device=device,
        )
    raise ValueError(f"Unsupported vocoder '{name}'")

def build_dataset(throat_dir="data/throat", air_dir="data/air"):
    """喉→空気のフレーム対応データ行列を作成（X: [N, 600], Y: [N, 40]）。"""
    X_list, Y_list = [], []
    throat_files = sorted(glob.glob(os.path.join(throat_dir, "*.wav")))
    assert len(throat_files) > 0, "喉マイクwavが見つかりません"

    for tpath in tqdm(throat_files, desc="Building dataset"):
        name = os.path.basename(tpath)
        apath = os.path.join(air_dir, name)
        if not os.path.exists(apath):
            # 拡張子違い対策など必要ならここで対応
            continue

        yt, ya = load_audio_align_pair(tpath, apath)

        Ft = feat_logmel(yt)     # [T, 120]
        Fa = feat_logmel(ya)     # [T, 120] だが目的は最初の40バンドのみ
        T = min(Ft.shape[0], Fa.shape[0])
        Ft = Ft[:T]
        Fa = Fa[:T]

        Xt = stack_context(Ft, CONTEXT, CONTEXT)  # [T, 120*(2C+1)]
        Ya = np.ascontiguousarray(Fa[:, :N_MELS]) # 目的: 空気側ログMel（40次）

        X_list.append(Xt)
        Y_list.append(Ya)

    X = np.vstack(X_list)  # [N, Din]
    Y = np.vstack(Y_list)  # [N, 40]
    return X, Y

def make_model(kind="mlp"):
    """モデル生成（'mlp' or 'ridge'）。"""
    if kind == "mlp":
        mlp = MLPRegressor(
            hidden_layer_sizes=(100, 100),
            activation='tanh',  # or 'relu'
            solver='adam',
            learning_rate_init=1e-3,
            max_iter=50,        # 必要なら増やす
            verbose=False,
            random_state=42,
            batch_size=256
        )
        model = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("reg", mlp)
        ])
    else:
        ridge = RidgeCV(alphas=np.logspace(-4, 3, 20))
        model = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("reg", ridge)
        ])
    return model

def train_and_save(model_out=MODEL_PATH, model_kind="mlp", throat_dir="data/throat", air_dir="data/air"):
    X, Y = build_dataset(throat_dir=throat_dir, air_dir=air_dir)
    print(f"Dataset: X={X.shape}, Y={Y.shape}")
    model = make_model(model_kind)
    model.fit(X, Y)
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    dump(model, model_out)
    print(f"Saved model to {model_out}")

def enhance_wav(
    in_wav,
    out_wav,
    model_path=MODEL_PATH,
    vocoder=None,
    vocoder_name=VOCODER_GRIFFINLIM,
    hifigan_checkpoint=None,
    hifigan_config=None,
    device=None,
):
    """喉wavを読み込み→ログMel(喉)→モデルで空気ログMel推定→再合成→保存"""
    assert os.path.exists(model_path), "学習済みモデルが見つかりません。先に --train を実行してください。"
    model = load(model_path)

    if vocoder is None:
        vocoder = create_vocoder(
            vocoder_name,
            hifigan_checkpoint=hifigan_checkpoint,
            hifigan_config=hifigan_config,
            device=device,
        )

    yt, _ = librosa.load(in_wav, sr=SR, mono=True)
    Ft = feat_logmel(yt)                     # [T, 120]
    Xt = stack_context(Ft, CONTEXT, CONTEXT) # [T, 120*(2C+1)]

    Yhat_mel_log = model.predict(Xt)         # [T, 40]（自然対数Melパワー）
    y_rec = vocoder(Yhat_mel_log)
    out_dir = os.path.dirname(out_wav)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sf.write(out_wav, y_rec, SR)
    print(f"Enhanced wav written: {out_wav}")

def bulk_enhance(
    in_dir,
    out_dir,
    model_path=MODEL_PATH,
    vocoder=None,
    vocoder_name=VOCODER_GRIFFINLIM,
    hifigan_checkpoint=None,
    hifigan_config=None,
    device=None,
):
    """ディレクトリ内の全wavファイルを一括変換"""
    assert os.path.exists(model_path), "学習済みモデルが見つかりません。先に --train を実行してください。"
    assert os.path.isdir(in_dir), f"入力ディレクトリが見つかりません: {in_dir}"
    
    # 出力ディレクトリを作成
    os.makedirs(out_dir, exist_ok=True)
    
    # 入力ディレクトリ内の全wavファイルを取得
    wav_files = sorted(glob.glob(os.path.join(in_dir, "*.wav")))
    
    if len(wav_files) == 0:
        print(f"警告: {in_dir} 内にwavファイルが見つかりません")
        return
    
    print(f"一括変換開始: {len(wav_files)}ファイルを処理します")
    
    # モデルを一度だけ読み込み（効率化）
    model = load(model_path)
    if vocoder is None:
        vocoder = create_vocoder(
            vocoder_name,
            hifigan_checkpoint=hifigan_checkpoint,
            hifigan_config=hifigan_config,
            device=device,
        )
    
    for in_wav in tqdm(wav_files, desc="Bulk enhancing"):
        filename = os.path.basename(in_wav)
        name, ext = os.path.splitext(filename)
        out_wav = os.path.join(out_dir, f"{name}_enhanced{ext}")
        
        try:
            # enhance_wavの中身を直接実行（モデル読み込みを避けるため）
            yt, _ = librosa.load(in_wav, sr=SR, mono=True)
            Ft = feat_logmel(yt)
            Xt = stack_context(Ft, CONTEXT, CONTEXT)
            
            Yhat_mel_log = model.predict(Xt)
            y_rec = vocoder(Yhat_mel_log)
            sf.write(out_wav, y_rec, SR)
            
        except Exception as e:
            print(f"エラー: {filename} の処理に失敗しました - {e}")
            continue
    
    print(f"一括変換完了: 結果は {out_dir} に保存されました")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="学習を実行してモデルを保存")
    ap.add_argument("--model", type=str, default="mlp", choices=["mlp", "ridge"], help="モデル種別")
    ap.add_argument("--enhance", nargs=2, metavar=("IN_WAV", "OUT_WAV"), help="1ファイルを変換")
    ap.add_argument("--bulk", nargs=2, metavar=("IN_DIR", "OUT_DIR"), help="ディレクトリ内の全wavファイルを一括変換")
    ap.add_argument("--throat-dir", type=str, default="data/throat", help="喉マイク音声ディレクトリ")
    ap.add_argument("--air-dir", type=str, default="data/air", help="空気マイク音声ディレクトリ")
    ap.add_argument("--model-path", type=str, default=MODEL_PATH, help="モデルファイルのパス")
    ap.add_argument(
        "--vocoder",
        type=str,
        default=VOCODER_GRIFFINLIM,
        choices=[VOCODER_GRIFFINLIM, VOCODER_HIFIGAN],
        help="mel_to_audioに使用するボコーダ（griffinlim または hifigan）。",
    )
    ap.add_argument(
        "--hifigan-checkpoint",
        type=str,
        default=None,
        help="事前学習済みHiFi-GANジェネレータ(.pt/.pth)。--vocoder hifigan 時は必須。",
    )
    ap.add_argument(
        "--hifigan-config",
        type=str,
        default=None,
        help="HiFi-GANのconfig JSON（省略可）。サンプルレート/Mel数に合わせてください。",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="HiFi-GAN推論に使うPyTorchデバイス（例: cuda:0）。省略時は自動判定。",
    )
    ap.add_argument("--sr", type=int, default=SR, help="処理サンプルレート。HiFi-GANの学習条件に合わせてください。")
    ap.add_argument("--n-mels", type=int, default=N_MELS, help="ログMelバンド数。HiFi-GANの条件と一致させます。")
    ap.add_argument("--fft", type=int, default=FFT, help="STFTのn_fft。")
    ap.add_argument("--hop", type=int, default=HOP, help="STFTのhop length。")
    ap.add_argument("--win", type=int, default=WIN, help="STFTのwindow length。")
    ap.add_argument("--fmin", type=float, default=FMIN, help="Melスペクトログラムの最低周波数。")
    ap.add_argument("--fmax", type=float, default=FMAX, help="Melスペクトログラムの最高周波数。")
    ap.add_argument("--power", type=float, default=POWER, help="Mel計算時のpower。")
    ap.add_argument(
        "--griffin-iters",
        type=int,
        default=N_ITER_GRIFFINLIM,
        help="Griffin-Limボコーダの反復回数（griffinlim選択時のみ有効）。",
    )
    args = ap.parse_args()

    override_signal_params(
        sr=args.sr,
        n_mels=args.n_mels,
        fft=args.fft,
        hop=args.hop,
        win=args.win,
        fmin=args.fmin,
        fmax=args.fmax,
        power=args.power,
        griffinlim_iters=args.griffin_iters,
    )

    if args.train:
        train_and_save(model_out=args.model_path, model_kind=args.model, throat_dir=args.throat_dir, air_dir=args.air_dir)

    vocoder = None
    if args.enhance or args.bulk:
        vocoder = create_vocoder(
            args.vocoder,
            hifigan_checkpoint=args.hifigan_checkpoint,
            hifigan_config=args.hifigan_config,
            device=args.device,
        )

    if args.enhance:
        enhance_wav(
            args.enhance[0],
            args.enhance[1],
            model_path=args.model_path,
            vocoder=vocoder,
            vocoder_name=args.vocoder,
            hifigan_checkpoint=args.hifigan_checkpoint,
            hifigan_config=args.hifigan_config,
            device=args.device,
        )
    if args.bulk:
        bulk_enhance(
            args.bulk[0],
            args.bulk[1],
            model_path=args.model_path,
            vocoder=vocoder,
            vocoder_name=args.vocoder,
            hifigan_checkpoint=args.hifigan_checkpoint,
            hifigan_config=args.hifigan_config,
            device=args.device,
        )

if __name__ == "__main__":
    main()
