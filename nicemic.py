import threading, queue, time
import numpy as np
import sounddevice as sd
from scipy.signal import iirnotch, sosfilt, sosfiltfilt, zpk2sos, butter, tf2sos
from nicegui import ui

SR=48000; BLOCK=256; CH=1

# ==== 可変パラメータ（UIから弄る） ====
params = {
    'hpf_fc': 100.0,
    'notch_hum': 50.0,      # 0で無効, 50 or 60
    'notch_q': 25.0,
    'mud_cut_freq': 350.0,  # “こもり”中心周波数
    'mud_cut_db': -6.0,
    'presence_db': +6.0,
    'presence_fc': 3000.0,
    'air_db': +6.0,
    'air_fc': 7000.0,
    'exciter_amt': 0.1,     # 0.0〜0.5
    'comp_thresh': -16.0,
    'comp_ratio': 2.0,
    'out_gain_db': 0.0,
}

def db2lin(db): return 10**(db/20)

# ==== フィルタ設計 ====
def sos_hpf(fc):
    z,p,k = butter(2, fc/(SR/2), btype='highpass', output='zpk')
    return zpk2sos(z,p,k)
def sos_peaking(fc, gain_db, q=0.8):
    A = db2lin(gain_db)
    w0 = 2*np.pi*fc/SR
    alpha = np.sin(w0)/(2*q)
    b0 = 1 + alpha*A
    b1 = -2*np.cos(w0)
    b2 = 1 - alpha*A
    a0 = 1 + alpha/A
    a1 = -2*np.cos(w0)
    a2 = 1 - alpha/A
    sos = np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1/a0, a2/a0]])
    return sos

def sos_high_shelf(fc, gain_db):
    # 簡易ハイシェルフ（2次）
    A = db2lin(gain_db)
    w0 = 2*np.pi*fc/SR
    alpha = np.sin(w0)/2*np.sqrt((A + 1/A)*(1/1 - 1) + 2) if False else np.sin(w0)/2
    # 実運用ではRBJ cookbookの棚フィルタ式推奨。ここは簡略で十分に効く。
    z,p,k = butter(2, fc/(SR/2), btype='highpass', output='zpk')
    sos = zpk2sos(z,p,k)
    # “近似棚”として高域だけ持ち上げたい：高域側をゲインでスケール
    sos[:, :3] *= A
    return sos

# ==== エキサイタ（簡易） ====
def excitator(x, amt, air_fc):
    if amt<=0: return x
    # 高域ノイズを生成して、元のエネルギー包絡でAMゲート
    noise = np.random.randn(len(x)).astype(np.float32) * 1e-3
    # ハイパスで高域だけ残す
    sos = sos_hpf(air_fc)
    hiss = sosfilt(sos, noise)
    # 元音声の短時間RMSでゲート
    win = 64
    pad = np.pad(x, (0, (-len(x)) % win))
    rms = np.sqrt(np.mean(pad.reshape(-1,win)**2, axis=1)+1e-8)
    env = np.repeat(rms, win)[:len(x)]
    add = hiss * (env/ (np.max(env)+1e-6)) * amt*4.0
    return x + add

# ==== ダイナミクス（ワンポール簡易コンプ） ====
def compressor(x, thresh_db=-18.0, ratio=2.0, atk=0.005, rel=0.08):
    thr = db2lin(thresh_db)
    y = np.empty_like(x)
    gain = 1.0
    for i, s in enumerate(x):
        lvl = abs(s)
        over = max(lvl/thr, 1.0)
        gr = over**(1.0-1.0/ratio)  # >1 で下げる
        target = 1.0/gr
        coef = 1-np.exp(-1.0/(SR*(atk if target<gain else rel)))
        gain += (target - gain)*coef
        y[i] = s * gain
    return y

# ==== I/O & Processing ====
q_in, q_out = queue.Queue(32), queue.Queue(32)

def process_block(x):
    # 1) HPF
    x = sosfilt(sos_hpf(params['hpf_fc']), x)
    # 2) ハムノッチ
    if params['notch_hum']>0:
        b, a = iirnotch(params['notch_hum'], params['notch_q'], fs=SR)
        sos = tf2sos(b, a)
        x = sosfilt(sos, x)
    # 3) こもり帯削り（ピーキングのマイナス）
    x = sosfilt(sos_peaking(params['mud_cut_freq'], params['mud_cut_db']), x)
    # 4) プレゼンス
    x = sosfilt(sos_peaking(params['presence_fc'], params['presence_db']), x)
    # 5) エア & エキサイタ
    x = sosfilt(sos_high_shelf(params['air_fc'], params['air_db']), x)
    x = excitator(x, params['exciter_amt'], params['air_fc'])
    # 6) コンプレッサ
    x = compressor(x, params['comp_thresh'], params['comp_ratio'])
    # 7) ソフトサチュ & 出力ゲイン
    x = np.tanh(x*1.5) * (1/np.tanh(1.5))
    x *= db2lin(params['out_gain_db'])
    return np.clip(x, -0.99, 0.99)

def proc_thread():
    while True:
        blk = q_in.get()
        if blk is None: break
        y = process_block(blk)
        try: q_out.put_nowait(y)
        except queue.Full: pass

def audio_cb(indata, outdata, frames, timeinfo, status):
    if status: pass
    try: q_in.put_nowait(indata[:,0].copy())
    except queue.Full: pass
    try:
        out = q_out.get_nowait()
    except queue.Empty:
        out = np.zeros(frames, dtype=np.float32)
    outdata[:] = out.reshape(-1,1)

# ==== Web UI ====
def add_slider(label, key, mn, mx, step):
    def format_value(val):
        return f'{val:.2f}' if step < 1 else f'{val:.0f}'
    def on_slider_change(e):
        value = float(e.value)
        params[key] = value
        value_lbl.text = format_value(value)
    with ui.row().classes('items-center gap-4 w-full'):
        ui.label(label).classes('w-44 text-sm font-medium')
        slider = ui.slider(min=mn, max=mx, step=step, value=params[key], on_change=on_slider_change).classes('w-80 max-w-full')
        value_lbl = ui.label(format_value(params[key])).classes('w-16 text-right text-sm')

def launch_ui():
    with ui.row():
        add_slider('HPF fc (Hz)','hpf_fc', 40, 200, 5)
        add_slider('Mud freq (Hz)','mud_cut_freq', 200, 600, 10)
        add_slider('Mud gain (dB)','mud_cut_db', -12, 0, 1)
    with ui.row():
        add_slider('Presence fc (Hz)','presence_fc', 1500, 5000, 100)
        add_slider('Presence (dB)','presence_db', 0, 12, 1)
    with ui.row():
        add_slider('Air fc (Hz)','air_fc', 5000, 12000, 200)
        add_slider('Air (dB)','air_db', 0, 12, 1)
        add_slider('Exciter amt','exciter_amt', 0.0, 0.5, 0.01)
    with ui.row():
        add_slider('Comp thresh (dBFS)','comp_thresh', -30, -6, 1)
        add_slider('Comp ratio','comp_ratio', 1.0, 4.0, 0.1)
        add_slider('Out gain (dB)','out_gain_db', -12, +12, 1)
    ui.label('Open: http://127.0.0.1:8080').style('margin-top:16px;')

@ui.page('/')
def main_page():
    with ui.column().classes('gap-4 p-4 max-w-xl'):
        ui.label('Voice Enhancer').classes('text-lg font-semibold')
        launch_ui()

if __name__ in {'__main__', '__mp_main__'}:
    t = threading.Thread(target=proc_thread, daemon=True); t.start()
    stream = sd.Stream(channels=CH, samplerate=SR, blocksize=BLOCK, dtype='float32', callback=audio_cb)
    stream.start()
    ui.run(native=False, port=8080)
