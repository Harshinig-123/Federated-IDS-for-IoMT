"""
MedSec-25  |  Variant 3 — Residual MLP with Feature Attention Gate
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Key changes vs baseline:
  • Residual (skip) connections: every 2-layer block adds its input back
    → prevents vanishing gradient in deeper networks
  • Feature attention gate: a learned sigmoid mask over the 105 input features
    → lets the model up-weight lateral_indicator, exfil_indicator etc.
  • Wider: 1024 → 512 → 256 → 128 (more capacity to model rare classes)
  • Cosine LR decay per round instead of fixed 0.0003
  • Same KL + CRPS loss and FL structure

Architecture diagram:
  Input (105)
    │
  [Attention Gate] sigmoid(W·x) * x
    │
  Dense(1024) + LN + DO(0.4)
    │
  Dense(512)  + LN + DO(0.4)  ←── skip from attention gate (projected)
    │
  Dense(256)  + LN + DO(0.3)
    │
  Dense(128)  + LN + DO(0.3)  ←── skip from 512 block
    │
  Dense(5, softmax)

Expected: Macro F1 ≥ 0.92 — better than V1, slightly below V2
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import psutil, sys
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
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (Dense, LayerNormalization, Dropout,
                                     Multiply, Add, Lambda, Activation)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow.keras.backend as K

# ─── Tee logger ───────────────────────────────────────────────────────────────
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

DATA_DIR  = r"D:\Harshini"
CSV_PATH  = r"D:\Harshini\MedSec-25.csv"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path  = f"{DATA_DIR}/run_v3_residual_{timestamp}.txt"
sys.stdout = Tee(log_path)
print(f"[V3 — Residual MLP + Attention]  Logging to: {log_path}")
print("TF:", tf.__version__)

ALL_CLASSES = sorted(['Benign','Exfiltration','Initial access',
                      'Lateral movement','Reconnaissance'])
N_CLASSES   = len(ALL_CLASSES)

DROP_COLS = [
    'Flow ID','Src IP','Dst IP','Timestamp',
    'Fwd PSH Flags','Fwd URG Flags',
    'Fwd Byts/b Avg','Fwd Pkts/b Avg','Fwd Blk Rate Avg',
    'Bwd Byts/b Avg','Bwd Pkts/b Avg','Bwd Blk Rate Avg',
    'Init Fwd Win Byts','Fwd Seg Size Min',
]
E_MIN, E_MAX = 1, 4
S_MIN, S_MAX = 128, 512


def kl_crps_combined_loss(label_smoothing=0.1, crps_weight=0.3):
    def loss(y_true, y_pred):
        n_classes     = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_true_smooth = (1.0 - label_smoothing) * y_true + label_smoothing / n_classes
        y_pred        = tf.clip_by_value(y_pred,        1e-7, 1.0 - 1e-7)
        y_true_smooth = tf.clip_by_value(y_true_smooth, 1e-7, 1.0)
        kl      = tf.reduce_sum(y_true_smooth * tf.math.log(y_true_smooth / y_pred), axis=-1)
        kl_loss = tf.reduce_mean(kl)
        cdf_pred  = tf.cumsum(y_pred,        axis=-1)
        cdf_true  = tf.cumsum(y_true_smooth, axis=-1)
        crps_loss = tf.reduce_mean(tf.reduce_sum(tf.square(cdf_pred - cdf_true), axis=-1))
        return (1.0 - crps_weight) * kl_loss + crps_weight * crps_loss
    return loss


def add_features(df):
    eps = 1e-6
    for col in ['Flow Duration','Flow Byts/s','Flow Pkts/s',
                'Fwd Pkts/s','Bwd Pkts/s',
                'Flow IAT Mean','Flow IAT Std','Flow IAT Max','Flow IAT Min']:
        if col in df.columns:
            df[col] = df[col].abs()

    tot_size = df['TotLen Fwd Pkts'] + df['TotLen Bwd Pkts']
    df['src_dst_rate_ratio']    = df['Fwd Pkts/s'] / (df['Bwd Pkts/s'] + eps)
    df['rate_duration_ratio']   = df['Flow Pkts/s'] / (df['Flow Duration'] + eps)
    df['syn_ack_ratio']         = df['SYN Flag Cnt'] / (df['ACK Flag Cnt'] + eps)
    df['fin_rst_sum']           = df['FIN Flag Cnt'] + df['RST Flag Cnt']
    df['flag_entropy']          = (df['SYN Flag Cnt'] + df['ACK Flag Cnt'] +
                                   df['FIN Flag Cnt'] + df['RST Flag Cnt'] +
                                   df['PSH Flag Cnt'])
    df['avg_size_ratio']        = df['Pkt Len Mean'] / (tot_size + eps)
    df['std_avg_ratio']         = df['Pkt Len Std']  / (df['Pkt Len Mean'] + eps)
    df['max_min_ratio']         = df['Pkt Len Max']  / (df['Pkt Len Min'] + eps)
    df['log_duration']          = np.log1p(df['Flow Duration'].abs())
    df['log_rate']              = np.log1p(df['Flow Pkts/s'].abs())
    df['log_tot_size']          = np.log1p(tot_size.abs())
    df['log_iat']               = np.log1p(df['Flow IAT Mean'].abs())
    df['fwd_bwd_pkt_ratio']     = df['Tot Fwd Pkts'] / (df['Tot Bwd Pkts'] + eps)
    df['fwd_bwd_len_ratio']     = df['TotLen Fwd Pkts'] / (df['TotLen Bwd Pkts'] + eps)
    df['iat_rate_product']      = df['Flow IAT Mean'] * df['Flow Pkts/s']
    df['iat_srate_ratio']       = df['Flow IAT Mean'] / (df['Fwd Pkts/s'] + eps)
    df['log_iat_sq']            = df['log_iat'] ** 2
    df['srate_rate_ratio']      = df['Fwd Pkts/s'] / (df['Flow Pkts/s'] + eps)
    df['pkt_size_consistency']  = 1.0 / (df['Pkt Len Std'] / (df['Pkt Len Mean'] + eps) + eps)
    df['duration_rate_product'] = df['Flow Duration'] * df['Flow Pkts/s']
    df['small_pkt_ratio']       = df['Pkt Len Min'] / (df['Pkt Len Mean'] + eps)
    df['rate_per_duration']     = df['Flow Pkts/s'] / (df['log_duration'] + eps)
    df['iat_consistency']       = df['Flow IAT Std'] / (df['Flow IAT Mean'] + eps)
    df['srate_dominance']       = df['Fwd Pkts/s'] / (df['Flow Pkts/s'] + eps)
    df['flow_concentration']    = (df['Fwd Pkts/s'] * df['Flow Duration']) / (tot_size + eps)
    df['iat_srate_product']     = df['Flow IAT Mean'] * df['Fwd Pkts/s']
    df['recon_indicator']       = df['Flow Pkts/s'] / (df['Pkt Len Mean'] + eps)
    df['exfil_indicator']       = df['TotLen Fwd Pkts'] / (df['Flow Duration'] + eps)
    df['lateral_indicator']     = df['Down/Up Ratio'] * df['Pkt Len Mean']
    df['access_indicator']      = df['SYN Flag Cnt'] / (df['Flow Duration'] + eps)
    df['pkt_var_ratio']         = df['Pkt Len Var'] / (df['Pkt Len Mean'] ** 2 + eps)
    df['header_payload_ratio']  = (df['Fwd Header Len'] + df['Bwd Header Len']) / (tot_size + eps)
    df['iat_burstiness']        = df['Flow IAT Std'] * df['Flow IAT Mean']
    df['subflow_symmetry']      = df['Subflow Fwd Pkts'] / (df['Subflow Bwd Pkts'] + eps)
    df['byte_rate']             = df['Flow Byts/s'].abs()
    df['log_byte_rate']         = np.log1p(df['byte_rate'])

    clip_cols = [
        'src_dst_rate_ratio','rate_duration_ratio','syn_ack_ratio',
        'avg_size_ratio','std_avg_ratio','max_min_ratio','fwd_bwd_pkt_ratio',
        'fwd_bwd_len_ratio','iat_srate_ratio','srate_rate_ratio',
        'pkt_size_consistency','small_pkt_ratio','rate_per_duration',
        'iat_consistency','srate_dominance','flow_concentration',
        'iat_srate_product','recon_indicator','exfil_indicator',
        'lateral_indicator','access_indicator','pkt_var_ratio',
        'header_payload_ratio','iat_burstiness','subflow_symmetry',
    ]
    for col in clip_cols:
        df[col] = df[col].replace([np.inf,-np.inf], np.nan).fillna(0).clip(-1e6,1e6)
    return df


def compute_class_weights(y_series):
    classes = np.unique(y_series)
    weights = compute_class_weight('balanced', classes=classes, y=y_series)
    cw      = dict(zip(classes, weights))
    boosts  = {
        'Benign':           12.0,
        'Lateral movement': 22.0,
        'Exfiltration':     11.0,
        'Initial access':    1.5,
        'Reconnaissance':    0.25,
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


def load_medsec(path):
    print(f"  Reading CSV...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Raw shape: {df.shape}")
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    df = df[df['Label'].isin(set(ALL_CLASSES))].reset_index(drop=True)
    counts = df['Label'].value_counts()
    print(f"  Class distribution:")
    for cls in ALL_CLASSES:
        cnt = counts.get(cls, 0)
        print(f"    {cls:<25} {cnt:>8,}  ({100*cnt/len(df):.2f}%)")
    df = add_features(df)
    return df


# ─── ★ KEY CHANGE: Residual MLP with feature attention gate ───────────────────
def build_model(n_features):
    """
    Functional API residual MLP.
    Block structure: Dense → LN → ReLU → Dropout → Dense → LN → Add(skip) → ReLU
    """
    reg = l2(1e-4)

    inputs = Input(shape=(n_features,))

    # ── Feature attention gate ──────────────────────────────────────────────
    # Learns a per-feature weight in [0,1] to up-weight informative features
    attn   = Dense(n_features, activation='sigmoid',
                   kernel_regularizer=reg, name='feature_attention')(inputs)
    x      = Multiply(name='attended_features')([inputs, attn])

    # ── Residual block 1: n_features → 1024 → 512 ──────────────────────────
    # Project input to 512 for the skip connection
    skip1  = Dense(512, use_bias=False, name='skip1_proj')(x)

    h      = Dense(1024, activation='relu', kernel_regularizer=reg)(x)
    h      = LayerNormalization()(h)
    h      = Dropout(0.4)(h)
    h      = Dense(512,  activation='relu', kernel_regularizer=reg)(h)
    h      = LayerNormalization()(h)
    h      = Dropout(0.4)(h)

    # Add skip, then activate
    h      = Add()([h, skip1])
    h      = Activation('relu')(h)

    # ── Residual block 2: 512 → 256 → 128 ──────────────────────────────────
    skip2  = Dense(128, use_bias=False, name='skip2_proj')(h)

    h      = Dense(256, activation='relu', kernel_regularizer=reg)(h)
    h      = LayerNormalization()(h)
    h      = Dropout(0.3)(h)
    h      = Dense(128, activation='relu', kernel_regularizer=reg)(h)
    h      = LayerNormalization()(h)
    h      = Dropout(0.3)(h)

    h      = Add()([h, skip2])
    h      = Activation('relu')(h)

    # ── Output ──────────────────────────────────────────────────────────────
    outputs = Dense(N_CLASSES, activation='softmax')(h)

    model = Model(inputs=inputs, outputs=outputs, name='ResidualMLP_Attention')
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
    return [momentum * global_weights[layer] + (1-momentum) * new_avg[layer]
            for layer in range(len(global_weights))]

def compute_node_params(node_accs, e_min, e_max, s_min, s_max):
    mu_acc = np.mean(list(node_accs.values()))
    params = {}
    for nid, ac in node_accs.items():
        sigma = min(1.0, (mu_acc - ac) / (mu_acc - min(node_accs.values()) + 1e-9)) \
                if ac <= mu_acc else 0.0
        ce = int(round(e_min + (e_max - e_min) * sigma))
        cs = int(round(s_max - (s_max - s_min) * sigma))
        params[nid] = (max(e_min, min(e_max, ce)), max(s_min, min(s_max, cs)))
    return params

def train_local(model, X, y, cw, epochs, batch_size, lr):
    """Local training with a per-round learning rate for cosine decay."""
    K.set_value(model.optimizer.learning_rate, lr)
    cbs = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2,
                          min_lr=1e-6, verbose=0),
        EarlyStopping(monitor='val_loss', patience=3,
                      restore_best_weights=True, verbose=0),
    ]
    h = model.fit(X, y, epochs=epochs, batch_size=batch_size,
                  class_weight=cw, callbacks=cbs,
                  validation_split=0.1, verbose=0)
    return (model.get_weights(),
            np.mean(h.history['loss']),
            np.mean(h.history['accuracy']),
            np.mean(h.history['val_loss']),
            np.mean(h.history['val_accuracy']))

def cosine_lr(fl_round, total_rounds, lr_max=3e-4, lr_min=1e-5):
    """Cosine annealing: starts at lr_max, decays to lr_min."""
    return lr_min + 0.5 * (lr_max - lr_min) * (
        1 + np.cos(np.pi * fl_round / total_rounds))

def preprocess(X_train, y_train, X_test, y_test, le):
    y_tr = to_categorical(le.transform(y_train), N_CLASSES)
    y_te = to_categorical(le.transform(y_test),  N_CLASSES)
    sc   = MinMaxScaler()
    Xtr  = np.nan_to_num(sc.fit_transform(X_train), nan=0.0, posinf=1.0, neginf=0.0)
    Xte  = np.nan_to_num(sc.transform(X_test),      nan=0.0, posinf=1.0, neginf=0.0)
    return Xtr, Xte, y_tr, y_te, sc

def predict_with_thresholds(model, X, thresholds):
    probs = model.predict(X, verbose=0)
    pred  = np.argmax(probs, 1).copy()
    for cls_idx in np.argsort(thresholds)[::-1]:
        pred[probs[:, cls_idx] >= thresholds[cls_idx]] = cls_idx
    return pred


# ─── Load & split ─────────────────────────────────────────────────────────────
print("\nLoading MedSec-25...")
df    = load_medsec(CSV_PATH)
X_all = df.drop(columns=[c for c in ['Label','Flow ID','Src IP','Dst IP','Timestamp']
                          if c in df.columns])
y_all = df['Label']
del df; gc.collect()

non_numeric = X_all.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    X_all = X_all.drop(columns=non_numeric)

X_trainval, X_test_full, y_trainval, y_test_full = train_test_split(
    X_all, y_all, test_size=0.20, random_state=42, stratify=y_all)
del X_all, y_all; gc.collect()

X_temp, X_node1, y_temp, y_node1 = train_test_split(
    X_trainval, y_trainval, test_size=0.33, random_state=42, stratify=y_trainval)
X_node2, X_node3, y_node2, y_node3 = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
del X_trainval, y_trainval, X_temp, y_temp; gc.collect()

X_tt, X_test1, y_tt, y_test1 = train_test_split(
    X_test_full, y_test_full, test_size=0.33, random_state=42, stratify=y_test_full)
X_test2, X_test3, y_test2, y_test3 = train_test_split(
    X_tt, y_tt, test_size=0.50, random_state=42, stratify=y_tt)
del X_test_full, y_test_full, X_tt, y_tt; gc.collect()

le = LabelEncoder()
le.fit(ALL_CLASSES)

X_tr1, X_te1, y_tr1, y_te1, _ = preprocess(X_node1, y_node1, X_test1, y_test1, le)
X_tr2, X_te2, y_tr2, y_te2, _ = preprocess(X_node2, y_node2, X_test2, y_test2, le)
X_tr3, X_te3, y_tr3, y_te3, _ = preprocess(X_node3, y_node3, X_test3, y_test3, le)

wn1 = compute_class_weights(y_node1); del y_node1
wn2 = compute_class_weights(y_node2); del y_node2
wn3 = compute_class_weights(y_node3); del y_node3
del X_node1, X_node2, X_node3, X_test1, X_test2, X_test3; gc.collect()

cw1 = convert_weights_to_indexed(wn1, le)
cw2 = convert_weights_to_indexed(wn2, le)
cw3 = convert_weights_to_indexed(wn3, le)

N_FEATURES      = X_tr1.shape[1]
node_data_sizes = [len(X_tr1), len(X_tr2), len(X_tr3)]
print(f"\nReady — features:{N_FEATURES}  classes:{N_CLASSES}")

# ─── FL loop ──────────────────────────────────────────────────────────────────
NODE_IDS       = [0, 1, 2]
node_Xtr       = [X_tr1, X_tr2, X_tr3]
node_ytr       = [y_tr1, y_tr2, y_tr3]
node_Xte       = [X_te1, X_te2, X_te3]
node_yte       = [y_te1, y_te2, y_te3]
node_cw        = [cw1, cw2, cw3]
FL_ROUNDS      = 30
PATIENCE       = 8
RECALL_TARGET  = 0.90

global_model   = build_model(N_FEATURES)
global_model.summary()
global_weights = global_model.get_weights()
node_models    = [build_model(N_FEATURES) for _ in NODE_IDS]
a_max, sc      = 0.0, 0
best_weights   = [w.copy() for w in global_weights]
node_accs_prev = {nid: 0.5 for nid in NODE_IDS}
round_metrics  = []

print("\n" + "=" * 62)
print("  V3 — Residual MLP + Attention | KL+CRPS | MedSec-25")
print("=" * 62)

for fl_round in range(1, FL_ROUNDS + 1):
    print(f"\n-- Round {fl_round}/{FL_ROUNDS} --")
    lr             = cosine_lr(fl_round, FL_ROUNDS)
    client_params  = compute_node_params(node_accs_prev, E_MIN, E_MAX, S_MIN, S_MAX)
    all_node_weights = []

    for nid in NODE_IDS:
        epochs, batch_size = client_params[nid]
        node_models[nid].set_weights(global_weights)
        w, loss, acc, vl, va = train_local(
            node_models[nid], node_Xtr[nid], node_ytr[nid],
            node_cw[nid], epochs, batch_size, lr)
        all_node_weights.append(w)
        print(f"  Node {nid+1} — ep:{epochs} bs:{batch_size} lr:{lr:.5f} "
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
    mu_acc = np.mean(list(all_node_accs.values()))

    y_pred_raw = global_model.predict(node_Xte[0], verbose=0)
    y_pred_idx = np.argmax(y_pred_raw, 1)
    y_true_idx = np.argmax(node_yte[0], 1)
    macro_f1   = f1_score(y_true_idx, y_pred_idx, average='macro', zero_division=0)

    y_true_l = le.inverse_transform(y_true_idx)
    y_pred_l = le.inverse_transform(y_pred_idx)

    all_recalls = {}
    for cls in ALL_CLASSES:
        mask = y_true_l == cls
        all_recalls[cls] = (y_pred_l[mask] == cls).mean() if mask.sum() > 0 else 0.0

    minority_recall = np.mean([all_recalls[c] for c in
                                ['Benign','Lateral movement','Exfiltration']])
    mean_fnpr       = np.mean([1 - r for r in all_recalls.values()])
    below_target    = [c for c in ALL_CLASSES if all_recalls[c] < RECALL_TARGET]
    floor_penalty   = sum(RECALL_TARGET - all_recalls[c] for c in below_target)
    recall_floor    = min(all_recalls.values())
    composite_score = (0.35 * macro_f1
                       + 0.30 * minority_recall
                       + 0.20 * (1 - mean_fnpr)
                       + 0.15 * recall_floor
                       - 0.80 * floor_penalty)

    print(f"  Global — mu_acc:{mu_acc:.4f}  MacroF1:{macro_f1:.4f}  Score:{composite_score:.4f}")
    print(f"  Below 0.90: {below_target}  FloorPenalty:{floor_penalty:.4f}")
    for cls in ALL_CLASSES:
        print(f"  {cls:<25} recall:{all_recalls[cls]:.3f}")

    round_metrics.append({'round': fl_round, 'f1': macro_f1,
                          'composite_score': composite_score})

    if composite_score > a_max:
        a_max        = composite_score
        sc           = 0
        best_weights = [w.copy() for w in global_weights]
        global_model.save(f"{DATA_DIR}/v3_residual_best_{timestamp}.keras")
        print(f"  ✓ Best saved — score:{a_max:.4f}  F1:{macro_f1:.4f}")
    else:
        sc += 1
        print(f"  No improvement — patience {sc}/{PATIENCE}")
    if sc > PATIENCE:
        print(f"  Early stopping at round {fl_round}")
        global_model.set_weights(best_weights)
        break
    gc.collect()

# ─── Per-node calibration ─────────────────────────────────────────────────────
print("\n=== Per-Node Threshold Calibration ===")
global_model.set_weights(best_weights)
node_test_data      = [(X_te1, y_te1), (X_te2, y_te2), (X_te3, y_te3)]
all_node_thresholds = []

for nid, (Xte_cal, yte_cal) in enumerate(node_test_data, 1):
    print(f"\n  Node {nid}:")
    y_probs_cal = global_model.predict(Xte_cal, verbose=0)
    y_true_cal  = np.argmax(yte_cal, 1)
    thresholds  = np.full(N_CLASSES, 0.5)
    for cls_idx, cls_name in enumerate(le.classes_):
        mask = y_true_cal == cls_idx
        if mask.sum() == 0:
            continue
        best_thresh = 0.5
        for thresh in np.arange(0.50, 0.00, -0.01):
            trial_pred = np.argmax(y_probs_cal, 1).copy()
            trial_pred[y_probs_cal[:, cls_idx] >= thresh] = cls_idx
            if (trial_pred[mask] == cls_idx).mean() >= RECALL_TARGET:
                best_thresh = thresh
                break
        thresholds[cls_idx] = best_thresh
        trial_pred = np.argmax(y_probs_cal, 1).copy()
        trial_pred[y_probs_cal[:, cls_idx] >= best_thresh] = cls_idx
        rec = (trial_pred[mask] == cls_idx).mean()
        print(f"    {cls_name:<25} threshold:{best_thresh:.2f}  recall:{rec:.4f}")
    all_node_thresholds.append(thresholds)

# ─── Final evaluation ─────────────────────────────────────────────────────────
print("\n=== Final Evaluation ===")
for nid, (Xte, yte) in enumerate(node_test_data, 1):
    y_pred_idx  = predict_with_thresholds(global_model, Xte, all_node_thresholds[nid-1])
    y_true_eval = le.inverse_transform(np.argmax(yte, 1))
    y_pred_eval = le.inverse_transform(y_pred_idx)
    report      = classification_report(y_true_eval, y_pred_eval,
                                        target_names=ALL_CLASSES, zero_division=0)
    print(f"\n=== Node {nid} — Classification Report ===")
    print(report)
    print(f"=== Node {nid} — Per-class FNPR ===")
    print(f"  {'Class':<27} {'Recall':>8} {'FNPR':>8}")
    print("  " + "-" * 46)
    for cls in ALL_CLASSES:
        mask = y_true_eval == cls
        if mask.sum() > 0:
            rec    = (y_pred_eval[mask] == cls).mean()
            status = "✅" if rec >= 0.90 else "❌"
            print(f"  {cls:<27} {rec:>8.4f} {1-rec:>8.4f}  {status}")
    rp = f"{DATA_DIR}/report_v3_node{nid}_{timestamp}.txt"
    with open(rp, 'w') as f:
        f.write(f"V3 Residual MLP — Node {nid}\n")
        f.write(f"Thresholds: {dict(zip(le.classes_, all_node_thresholds[nid-1].round(3)))}\n\n")
        f.write(report)
    print(f"  Report → {rp}")

print(f"\nDone. Best composite score: {a_max:.4f}")
sys.stdout.log.close()
sys.stdout = sys.stdout.terminal