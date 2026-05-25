import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import psutil
import sys
from datetime import datetime

num_cores = psutil.cpu_count(logical=True)
tf.config.threading.set_intra_op_parallelism_threads(num_cores)
tf.config.threading.set_inter_op_parallelism_threads(num_cores)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU: {[g.name for g in gpus]}")
else:
    print("No GPU — CPU mode")

import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

class Tee:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log      = open(filepath, 'w', encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()

data_dir  = "C:\\Users\\sandh\\Downloads\\CICIoMT\\attacks"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path  = f"{data_dir}/run_kl_crps16_{timestamp}.txt"
sys.stdout = Tee(log_path)
print(f"Logging to: {log_path}")
print("TF:", tf.__version__)

def kl_crps_combined_loss(label_smoothing=0.1, crps_weight=0.3):
    def loss(y_true, y_pred):
        n_classes     = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true_smooth = (1.0 - label_smoothing) * y_true + label_smoothing / n_classes
        y_pred        = tf.clip_by_value(y_pred,        1e-7, 1.0 - 1e-7)
        y_true_smooth = tf.clip_by_value(y_true_smooth, 1e-7, 1.0)

        kl      = tf.reduce_sum(
            y_true_smooth * tf.math.log(y_true_smooth / y_pred), axis=-1)
        kl_loss = tf.reduce_mean(kl)

        cdf_pred  = tf.cumsum(y_pred,        axis=-1)
        cdf_true  = tf.cumsum(y_true_smooth, axis=-1)
        crps      = tf.reduce_sum(tf.square(cdf_pred - cdf_true), axis=-1)
        crps_loss = tf.reduce_mean(crps)

        return (1.0 - crps_weight) * kl_loss + crps_weight * crps_loss
    return loss

data_dir  = "C:\\Users\\sandh\\Downloads\\CICIoMT\\attacks"
E_MIN, E_MAX = 1, 4
S_MIN, S_MAX = 128, 512

MAPPINGS = [
    ('ARP_Spoofing',           'ARP_Spoofing'),
    ('MQTT-DDoS-Connect_Flood','MQTT-DDoS-Connect_Flood'),
    ('MQTT-DDoS-Publish_Flood','MQTT-DDoS-Publish_Flood'),
    ('MQTT-DoS-Connect_Flood', 'MQTT-DoS-Connect_Flood'),
    ('MQTT-DoS-Publish_Flood', 'MQTT-DoS-Publish_Flood'),
    ('MQTT-Malformed_Data',    'MQTT-Malformed_Data'),
    ('Recon-OS_Scan',          'Recon'),
    ('Recon-Ping_Sweep',       'Recon'),
    ('Recon-Port_Scan',        'Recon'),
    ('Recon-VulScan',          'Recon'),
    ('TCP_IP-DDoS-ICMP',       'DDoS-ICMP'),
    ('TCP_IP-DDoS-SYN',        'DDoS-SYN'),
    ('TCP_IP-DDoS-TCP',        'DDoS-TCP'),
    ('TCP_IP-DDoS-UDP',        'DDoS-UDP'),
    ('TCP_IP-DoS-ICMP',        'DoS-ICMP'),
    ('TCP_IP-DoS-SYN',         'DoS-SYN'),
    ('TCP_IP-DoS-TCP',         'DoS-TCP'),
    ('TCP_IP-DoS-UDP',         'DoS-UDP'),
    ('Benign',                 'Benign'),
]

ALL_CLASSES_16 = sorted([
    'ARP_Spoofing', 'Benign',
    'DDoS-ICMP', 'DDoS-SYN', 'DDoS-TCP', 'DDoS-UDP',
    'DoS-ICMP',  'DoS-SYN',  'DoS-TCP',  'DoS-UDP',
    'MQTT-DDoS-Connect_Flood', 'MQTT-DDoS-Publish_Flood',
    'MQTT-DoS-Connect_Flood',  'MQTT-DoS-Publish_Flood',
    'MQTT-Malformed_Data',     'Recon',
])
N_CLASSES = 16

def check_ram():
    print(f"RAM: {psutil.virtual_memory().percent}%")

# ── CHANGE 1: Two new UDP-specific features added ──────────────────────────
# udp_srate_concentration: DoS-UDP has one source so UDP tracks closely with
#   Srate. DDoS-UDP has many sources so UDP is spread, Srate per source lower.
# udp_duration_density: packets-per-second for UDP, helps separate bursty DoS
#   (high density, short duration) from distributed DDoS (lower density).
def add_features(df):
    eps = 1e-6
    df['src_dst_rate_ratio']        = df['Srate'] / (df['Drate'] + eps)
    df['rate_duration_ratio']       = df['Rate']  / (df['Duration'] + eps)
    df['syn_ack_ratio']             = df['syn_flag_number'] / (df['ack_flag_number'] + eps)
    df['fin_rst_sum']               = df['fin_flag_number'] + df['rst_flag_number']
    df['flag_entropy']              = (df['syn_flag_number'] + df['ack_flag_number'] +
                                       df['fin_flag_number'] + df['rst_flag_number'] +
                                       df['psh_flag_number'])
    df['avg_size_ratio']            = df['AVG'] / (df['Tot size'] + eps)
    df['std_avg_ratio']             = df['Std'] / (df['AVG'] + eps)
    df['max_min_ratio']             = df['Max'] / (df['Min'] + eps)
    df['log_duration']              = np.log1p(np.abs(df['Duration']))
    df['log_rate']                  = np.log1p(np.abs(df['Rate']))
    df['log_tot_size']              = np.log1p(np.abs(df['Tot size']))
    df['log_iat']                   = np.log1p(np.abs(df['IAT']))
    df['tcp_udp_ratio']             = df['TCP']  / (df['UDP']  + eps)
    df['icmp_tcp_ratio']            = df['ICMP'] / (df['TCP']  + eps)
    df['iat_rate_product']          = df['IAT'] * df['Rate']
    df['iat_srate_ratio']           = df['IAT'] / (df['Srate'] + eps)
    df['log_iat_sq']                = df['log_iat'] ** 2
    df['srate_rate_ratio']          = df['Srate'] / (df['Rate'] + eps)
    df['dos_ddos_indicator']        = df['Srate'] / (df['Drate'] + eps) * df['IAT']
    df['pkt_size_consistency']      = 1.0 / (df['Std'] / (df['AVG'] + eps) + eps)
    df['duration_rate_product']     = df['Duration'] * df['Rate']
    df['icmp_per_flow']             = df['ICMP'] / (df['Duration'] + eps)
    df['icmp_rate_ratio']           = df['ICMP'] / (df['Rate'] + eps)
    df['pkt_rate_size_interaction'] = df['Rate'] * df['AVG']
    df['small_pkt_ratio']           = df['Min'] / (df['AVG'] + eps)
    df['rate_per_duration']         = df['Rate'] / (df['log_duration'] + eps)
    df['iat_consistency']           = df['Std'] / (df['IAT'] + eps)
    df['srate_dominance']           = df['Srate'] / (df['Rate'] + eps)
    df['flow_concentration']        = (df['Srate'] * df['Duration']) / (df['Tot size'] + eps)
    df['iat_srate_product']         = df['IAT'] * df['Srate']

    # CHANGE 1A — UDP source-concentration: single-source DoS produces high
    # UDP count relative to Srate; distributed DDoS spreads packets across
    # many sources so this ratio is lower.
    df['udp_srate_concentration']   = df['UDP'] / (df['Srate'] + eps)

    # CHANGE 1B — UDP temporal density: bursty single-source DoS floods at a
    # high packet rate over a short window; DDoS tends to be more sustained.
    df['udp_duration_density']      = df['UDP'] / (df['Duration'] + eps)

    clip_cols = [
        'src_dst_rate_ratio', 'rate_duration_ratio', 'syn_ack_ratio',
        'avg_size_ratio', 'std_avg_ratio', 'max_min_ratio',
        'tcp_udp_ratio', 'icmp_tcp_ratio', 'iat_srate_ratio',
        'srate_rate_ratio', 'dos_ddos_indicator', 'pkt_size_consistency',
        'icmp_per_flow', 'icmp_rate_ratio',
        'pkt_rate_size_interaction', 'small_pkt_ratio',
        'rate_per_duration', 'iat_consistency',
        'srate_dominance', 'flow_concentration', 'iat_srate_product',
        # CHANGE 1C — clip the two new features alongside the others
        'udp_srate_concentration', 'udp_duration_density',
    ]
    for col in clip_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6)
    return df

# ── CHANGE 2: DoS-UDP boost raised 18→25, MQTT-DoS-Publish set explicitly ──
# DoS-UDP raised because DDoS-UDP capping (Change 3) redistributes gradient
# budget — the model now sees a balanced pair but needs a strong signal to
# not collapse DoS-UDP recall. 25x = conservative upper bound.
#
# MQTT-DoS-Publish was previously only getting sklearn balanced weights (no
# explicit boost), which caused it to act as a catch-all for other MQTT
# classes. Setting it to 3.0x is deliberately lower than DoS-Publish (8x)
# so the model learns to be selective rather than safe-defaulting to it.
def compute_class_weights(y_series):
    classes = np.unique(y_series)
    weights = compute_class_weight('balanced', classes=classes, y=y_series)
    cw      = dict(zip(classes, weights))
    boosts  = {
        'DoS-UDP':                 25.0,   # CHANGED: was 18.0
        'DoS-ICMP':                12.0,
        'DoS-TCP':                 10.0,
        'DoS-SYN':                  4.0,
        'Recon':                    3.0,
        'MQTT-Malformed_Data':      8.0,
        'MQTT-DDoS-Publish_Flood': 20.0,
        'MQTT-DoS-Publish_Flood':   3.0,   # CHANGED: was absent (catch-all fix)
        'ARP_Spoofing':             4.0,
    }
    for cls, f in boosts.items():
        if cls in cw:
            cw[cls] *= f
    return cw

def convert_weights_to_indexed(cwd, le):
    result = {i: 1.0 for i in range(N_CLASSES)}
    for c, w in cwd.items():
        if c in le.classes_:
            result[le.transform([c])[0]] = w
    return result

def load_and_label(path):
    df = pd.read_csv(path)
    df['label_16'] = 'Unknown'
    for key, lbl in MAPPINGS:
        df.loc[df['file'].str.contains(key, na=False), 'label_16'] = lbl
    df = df[df['label_16'] != 'Unknown']
    return add_features(df)

def verify_classes(df, stage):
    counts  = df['label_16'].value_counts()
    missing = [c for c in ALL_CLASSES_16 if counts.get(c, 0) == 0]
    if missing:
        print(f"  WARNING [{stage}] missing: {missing}")
    else:
        print(f"  OK [{stage}] — all 16 classes present")
    print(f"  Class counts [{stage}]:")
    for cls in ALL_CLASSES_16:
        print(f"    {cls}: {counts.get(cls, 0)}")

# ── CHANGE 3: DDoS-UDP pre-capping before proportional subsampling ──────────
# Previously DDoS-UDP had ~2.88x as many samples as DoS-UDP per node.
# Weight boosting alone can't overcome a 3:1 volume imbalance when both
# classes share similar features — the gradient from the dominant class
# always wins in practice.
#
# Fix: cap DDoS-UDP to 2x the DoS-UDP count in the current split BEFORE
# any other subsampling. This gives the model a near-balanced pair to learn
# from, then the 25x DoS-UDP weight handles the remaining within-batch
# emphasis. The 2x cap is intentionally conservative — DDoS-UDP recall was
# already 0.96+ so we have headroom to trade a few points there.
def smart_subsample(X, y, n=400000):
    # CHANGE 3A — cap DDoS-UDP to 2x DoS-UDP before proportional subsampling
    dos_udp_count  = int((y == 'DoS-UDP').sum())
    ddos_udp_count = int((y == 'DDoS-UDP').sum())
    ddos_udp_cap   = dos_udp_count * 2

    if ddos_udp_count > ddos_udp_cap:
        ddos_udp_idx = np.where(y == 'DDoS-UDP')[0]
        np.random.shuffle(ddos_udp_idx)
        keep_mask              = np.ones(len(X), dtype=bool)
        keep_mask[ddos_udp_idx[ddos_udp_cap:]] = False
        X = X.iloc[keep_mask].reset_index(drop=True)
        y = y.iloc[keep_mask].reset_index(drop=True)
        print(f"    DDoS-UDP capped: {ddos_udp_count:,} → {ddos_udp_cap:,}  "
              f"(DoS-UDP: {dos_udp_count:,})")
    else:
        print(f"    DDoS-UDP within cap ({ddos_udp_count:,} ≤ {ddos_udp_cap:,}) — no capping")

    # Rest of function unchanged from original
    protected_minimums = {
        'MQTT-DDoS-Publish_Flood': 9000,
        'MQTT-Malformed_Data':     1700,
        'MQTT-DoS-Connect_Flood':  4000,
    }
    if len(X) <= n:
        print(f"    No subsampling needed: {len(X):,} samples fit within cap")
        return X, y

    idx_keep = []
    idx_pool = []
    for cls in y.unique():
        cls_idx  = np.where(y == cls)[0]
        min_keep = protected_minimums.get(cls, 0)
        if min_keep > 0:
            np.random.shuffle(cls_idx)
            keep = cls_idx[:min(min_keep, len(cls_idx))]
            idx_keep.extend(keep.tolist())
            idx_pool.extend(cls_idx[len(keep):].tolist())
        else:
            idx_pool.extend(cls_idx.tolist())

    budget = n - len(idx_keep)
    if budget <= 0:
        idx_keep = np.array(idx_keep)
        np.random.shuffle(idx_keep)
        return (X.iloc[idx_keep[:n]].reset_index(drop=True),
                y.iloc[idx_keep[:n]].reset_index(drop=True))

    pool_y            = y.iloc[idx_pool]
    pool_class_counts = pool_y.value_counts()
    pool_total        = len(idx_pool)

    pool_idx_by_class = {}
    for i in idx_pool:
        c = y.iloc[i]
        if c not in pool_idx_by_class:
            pool_idx_by_class[c] = []
        pool_idx_by_class[c].append(i)

    sampled_pool = []
    for cls, cnt in pool_class_counts.items():
        take     = max(1, int(budget * cnt / pool_total))
        cls_pool = np.array(pool_idx_by_class.get(cls, []))
        np.random.shuffle(cls_pool)
        sampled_pool.extend(cls_pool[:min(take, len(cls_pool))].tolist())

    final_idx = np.array(idx_keep + sampled_pool)
    np.random.shuffle(final_idx)
    final_idx = final_idx[:n]

    X_out = X.iloc[final_idx].reset_index(drop=True)
    y_out = y.iloc[final_idx].reset_index(drop=True)

    counts_out = y_out.value_counts()
    print(f"    Subsample result ({len(X_out):,} total):")
    for cls in ALL_CLASSES_16:
        if cls in counts_out:
            print(f"      {cls}: {counts_out[cls]:,}")
    return X_out, y_out

def preprocess(X_train, y_train, X_test, y_test, le):
    y_tr = to_categorical(le.transform(y_train), N_CLASSES)
    y_te = to_categorical(le.transform(y_test),  N_CLASSES)
    sc   = MinMaxScaler()
    return sc.fit_transform(X_train), sc.transform(X_test), y_tr, y_te, sc

def build_model(n_features):
    model = Sequential([
        Input(shape=(n_features,)),
        Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(), Dropout(0.4),
        Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(), Dropout(0.4),
        Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(), Dropout(0.3),
        Dense(64,  activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(), Dropout(0.3),
        Dense(N_CLASSES, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(0.0003),
        loss=kl_crps_combined_loss(label_smoothing=0.1, crps_weight=0.3),
        metrics=['accuracy']
    )
    return model

def federated_averaging(global_weights, node_weights_list, sizes, momentum=0.3):
    total   = sum(sizes)
    new_avg = [
        sum(node_weights_list[j][layer] * (sizes[j] / total)
            for j in range(len(sizes)))
        for layer in range(len(global_weights))
    ]
    return [
        momentum * global_weights[layer] + (1 - momentum) * new_avg[layer]
        for layer in range(len(global_weights))
    ]

def compute_node_params(node_accs, e_min, e_max, s_min, s_max):
    mu_acc = np.mean(list(node_accs.values()))
    params = {}
    for nid, ac in node_accs.items():
        if ac <= mu_acc:
            sigma = (mu_acc - ac) / (mu_acc - min(node_accs.values()) + 1e-9)
            sigma = min(1.0, sigma)
        else:
            sigma = 0.0
        ce = int(round(e_min + (e_max - e_min) * sigma))
        cs = int(round(s_max - (s_max - s_min) * sigma))
        params[nid] = (max(e_min, min(e_max, ce)), max(s_min, min(s_max, cs)))
    return params

def train_local(model, X, y, cw, epochs, batch_size):
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=2, min_lr=1e-6, verbose=0),
        EarlyStopping(monitor='val_loss', patience=3,
                      restore_best_weights=True, verbose=0),
    ]
    h = model.fit(X, y, epochs=epochs, batch_size=batch_size,
                  class_weight=cw, callbacks=callbacks,
                  validation_split=0.1, verbose=0)
    return (model.get_weights(),
            np.mean(h.history['loss']),
            np.mean(h.history['accuracy']),
            np.mean(h.history['val_loss']),
            np.mean(h.history['val_accuracy']))

def predict_with_icmp_threshold(model, X, le, dos_icmp_threshold=0.25):
    probs         = model.predict(X, verbose=0)
    dos_icmp_idx  = le.transform(['DoS-ICMP'])[0]
    ddos_icmp_idx = le.transform(['DDoS-ICMP'])[0]
    preds         = np.argmax(probs, axis=1)
    mask          = preds == ddos_icmp_idx
    preds[mask]   = np.where(
        probs[mask, dos_icmp_idx] >= dos_icmp_threshold,
        dos_icmp_idx, ddos_icmp_idx)
    return preds

# ── Load Data ──────────────────────────────────────────────────────────────
print("Loading train...")
train_df = load_and_label(f"{data_dir}/train_combined.csv")
verify_classes(train_df, "TRAIN")
X = train_df.drop(columns=['file', 'label_16'])
y = train_df['label_16']
del train_df; gc.collect()

X_temp, X_node1, y_temp, y_node1 = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y)
X_node2, X_node3, y_node2, y_node3 = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
del X, y, X_temp, y_temp; gc.collect()

print("\nNode 1 subsampling:")
X_node1, y_node1 = smart_subsample(X_node1, y_node1, n=400000)
print("\nNode 2 subsampling:")
X_node2, y_node2 = smart_subsample(X_node2, y_node2, n=400000)
print("\nNode 3 subsampling:")
X_node3, y_node3 = smart_subsample(X_node3, y_node3, n=400000)

print("\nLoading test...")
test_df = load_and_label(f"{data_dir}/test_combined.csv")
verify_classes(test_df, "TEST")
X_ta = test_df.drop(columns=['file', 'label_16'])
y_ta = test_df['label_16']
del test_df; gc.collect()

X_tt, X_test1, y_tt, y_test1 = train_test_split(
    X_ta, y_ta, test_size=0.33, random_state=42, stratify=y_ta)
X_test2, X_test3, y_test2, y_test3 = train_test_split(
    X_tt, y_tt, test_size=0.50, random_state=42, stratify=y_tt)
del X_ta, y_ta, X_tt, y_tt; gc.collect()

le = LabelEncoder()
le.fit(ALL_CLASSES_16)

wn1 = compute_class_weights(y_node1)
wn2 = compute_class_weights(y_node2)
wn3 = compute_class_weights(y_node3)

X_tr1, X_te1, y_tr1, y_te1, sc1 = preprocess(X_node1, y_node1, X_test1, y_test1, le)
X_tr2, X_te2, y_tr2, y_te2, sc2 = preprocess(X_node2, y_node2, X_test2, y_test2, le)
X_tr3, X_te3, y_tr3, y_te3, sc3 = preprocess(X_node3, y_node3, X_test3, y_test3, le)
del X_node1, X_node2, X_node3, X_test1, X_test2, X_test3; gc.collect()

N_FEATURES      = X_tr1.shape[1]
cw1             = convert_weights_to_indexed(wn1, le)
cw2             = convert_weights_to_indexed(wn2, le)
cw3             = convert_weights_to_indexed(wn3, le)
node_data_sizes = [len(X_tr1), len(X_tr2), len(X_tr3)]
print(f"\nReady — features:{N_FEATURES}  classes:{N_CLASSES}"); check_ram()

NODE_IDS  = [0, 1, 2]
node_Xtr  = [X_tr1, X_tr2, X_tr3]
node_ytr  = [y_tr1, y_tr2, y_tr3]
node_Xte  = [X_te1, X_te2, X_te3]
node_yte  = [y_te1, y_te2, y_te3]
node_cw   = [cw1, cw2, cw3]

FL_ROUNDS      = 20
PATIENCE       = 6
global_model   = build_model(N_FEATURES)
global_weights = global_model.get_weights()
node_models    = [build_model(N_FEATURES) for _ in NODE_IDS]

a_max          = 0.0
sc             = 0
best_weights   = global_weights
round_metrics  = []
recall_history = {cls: [] for cls in ALL_CLASSES_16}
node_accs_prev = {nid: 0.5 for nid in NODE_IDS}

print("=" * 60)
print("  Adaptive FL — MLP + KL+CRPS — 16 Classes (Recon merged)")
print(f"  E:[{E_MIN},{E_MAX}]  BatchSize:[{S_MIN},{S_MAX}]  Patience:{PATIENCE}")
print("  Loss: 70% KL Divergence + 30% CRPS")
print("  Changes: DDoS-UDP capped | DoS-UDP w=25 | MQTT-DoS-Pub w=3 | +2 UDP features")
print("=" * 60)

for fl_round in range(1, FL_ROUNDS + 1):
    print(f"\n-- Round {fl_round}/{FL_ROUNDS} --")

    client_params = compute_node_params(
        node_accs_prev, E_MIN, E_MAX, S_MIN, S_MAX)

    all_node_weights = []
    node_val_accs    = {}

    for nid in NODE_IDS:
        epochs, batch_size = client_params[nid]
        node_models[nid].set_weights(global_weights)
        w, loss, acc, vl, va = train_local(
            node_models[nid], node_Xtr[nid], node_ytr[nid],
            node_cw[nid], epochs, batch_size)
        all_node_weights.append(w)
        node_val_accs[nid] = va
        print(f"  Node {nid+1} — epochs:{epochs} batch:{batch_size} "
              f"loss:{loss:.4f} acc:{acc:.4f} val_acc:{va:.4f}")

    global_weights = federated_averaging(
        global_weights, all_node_weights, node_data_sizes, momentum=0.3)
    global_model.set_weights(global_weights)

    all_node_accs = {}
    for nid in NODE_IDS:
        yp = global_model.predict(node_Xte[nid], verbose=0)
        all_node_accs[nid] = float(
            np.mean(np.argmax(yp, 1) == np.argmax(node_yte[nid], 1)))
    node_accs_prev = all_node_accs

    mu_acc     = np.mean(list(all_node_accs.values()))
    y_pred_raw = global_model.predict(node_Xte[0], verbose=0)
    macro_f1   = f1_score(np.argmax(node_yte[0], 1),
                          np.argmax(y_pred_raw, 1), average='macro')

    y_pred_idx = predict_with_icmp_threshold(
        global_model, node_Xte[0], le, dos_icmp_threshold=0.25)
    y_true_l   = le.inverse_transform(np.argmax(node_yte[0], 1))
    y_pred_l   = le.inverse_transform(y_pred_idx)

    all_recalls = {}
    for cls in ALL_CLASSES_16:
        mask             = y_true_l == cls
        all_recalls[cls] = (y_pred_l[mask] == cls).mean() \
                           if mask.sum() > 0 else 0.0

    dos_recalls     = {k: all_recalls[k] for k in
                       ['DoS-ICMP', 'DoS-SYN', 'DoS-TCP', 'DoS-UDP']}
    dos_mean_recall = np.mean(list(dos_recalls.values()))
    composite_score = 0.5 * macro_f1 + 0.5 * dos_mean_recall

    print(f"  Global — mu_acc:{mu_acc:.4f}  F1:{macro_f1:.4f}  "
          f"DoS:{dos_mean_recall:.4f}  Score:{composite_score:.4f}")
    print(f"  node_accs: {[f'{all_node_accs[n]:.4f}' for n in NODE_IDS]}")
    print(f"  DoS  : { {k: f'{v:.3f}' for k,v in dos_recalls.items()} }")
    print(f"  DDoS : TCP={all_recalls['DDoS-TCP']:.3f}  "
          f"UDP={all_recalls['DDoS-UDP']:.3f}  "
          f"SYN={all_recalls['DDoS-SYN']:.3f}  "
          f"ICMP={all_recalls['DDoS-ICMP']:.3f}")
    print(f"  MQTT : Connect={all_recalls['MQTT-DDoS-Connect_Flood']:.3f}  "
          f"Publish={all_recalls['MQTT-DDoS-Publish_Flood']:.3f}  "
          f"Malformed={all_recalls['MQTT-Malformed_Data']:.3f}")
    print(f"  Recon:{all_recalls['Recon']:.3f}  "
          f"Benign:{all_recalls['Benign']:.3f}  "
          f"ARP:{all_recalls['ARP_Spoofing']:.3f}")

    # MQTT-DoS-Publish precision monitoring (catch-all warning)
    mqtt_dos_pub_mask = y_true_l == 'MQTT-DoS-Publish_Flood'
    mqtt_dos_pub_pred_mask = y_pred_l == 'MQTT-DoS-Publish_Flood'
    if mqtt_dos_pub_pred_mask.sum() > 0:
        mqtt_dos_pub_precision = (
            (y_pred_l[mqtt_dos_pub_pred_mask] == y_true_l[mqtt_dos_pub_pred_mask]).sum()
            / mqtt_dos_pub_pred_mask.sum()
        )
        print(f"  MQTT-DoS-Pub precision: {mqtt_dos_pub_precision:.3f}  "
              f"(target >0.70, catch-all if <0.55)")

    for cls in ALL_CLASSES_16:
        rec = all_recalls[cls]
        recall_history[cls].append(rec)
        if len(recall_history[cls]) >= 3:
            prev_avg = np.mean(recall_history[cls][:-1])
            if prev_avg > 0.1 and (prev_avg - rec) > 0.15:
                print(f"  !! WARNING: {cls} dropped {prev_avg:.3f} -> {rec:.3f}")

    round_metrics.append({
        'round':           fl_round,
        'mu_acc':          mu_acc,
        'f1':              macro_f1,
        'dos_mean_recall': dos_mean_recall,
        'composite_score': composite_score,
        **{f'node{n+1}_acc': all_node_accs[n] for n in NODE_IDS},
        **{f'{k}_recall': v for k, v in all_recalls.items()}
    })

    if composite_score > a_max:
        a_max        = composite_score
        sc           = 0
        best_weights = [w.copy() for w in global_weights]
        global_model.save(f"{data_dir}/kl_crps16_best_{timestamp}.keras")
        print(f"  Best saved — score:{a_max:.4f}  F1:{macro_f1:.4f}  "
              f"DoS:{dos_mean_recall:.4f}")
    else:
        sc += 1
        print(f"  No improvement — patience {sc}/{PATIENCE}")

    if sc > PATIENCE:
        print(f"  Early stopping at round {fl_round}")
        global_model.set_weights(best_weights)
        break

    if fl_round % 5 == 0:
        global_model.save(f"{data_dir}/kl_crps16_round{fl_round}_{timestamp}.keras")
        print("  Checkpoint saved")

    gc.collect()

global_model.set_weights(best_weights)
global_model.save(f"{data_dir}/kl_crps16_final_{timestamp}.keras")

metrics_df = pd.DataFrame(round_metrics)
metrics_df.to_csv(f"{data_dir}/metrics_kl_crps16_{timestamp}.csv", index=False)
print(f"\nMetrics saved to: metrics_kl_crps16_{timestamp}.csv")

print("\n=== Final Evaluation ===")
for nid, (Xte, yte) in enumerate(
        [(X_te1, y_te1), (X_te2, y_te2), (X_te3, y_te3)], 1):
    yp_idx      = predict_with_icmp_threshold(
        global_model, Xte, le, dos_icmp_threshold=0.25)
    y_true_eval = le.inverse_transform(np.argmax(yte, 1))
    y_pred_eval = le.inverse_transform(yp_idx)
    report      = classification_report(
        y_true_eval, y_pred_eval,
        target_names=ALL_CLASSES_16, zero_division=0)
    print(f"\n=== Node {nid} ===")
    print(report)
    report_path = f"{data_dir}/report_klcrps_node{nid}_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write(f"Node {nid} — KL+CRPS Combined Loss\n")
        f.write(f"Run timestamp: {timestamp}\n\n")
        f.write(report)
    print(f"  Report saved to: report_klcrps_node{nid}_{timestamp}.txt")

print(f"\nDone. Best composite score: {a_max:.4f}")
sys.stdout.log.close()
sys.stdout = sys.stdout.terminal
print(f"Log saved to: {log_path}")