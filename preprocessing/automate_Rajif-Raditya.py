"""
automate_Rajif-Athallah.py
──────────────────────────────────────────────────────────────────────────────
MLOps Preprocessing Pipeline — Diabetes Prediction Dataset
Dicoding Submission: Membangun Sistem Machine Learning (Tier Advance)

Menjalankan semua tahap preprocessing secara modular dan otomatis:
  1. load_data()          — Memuat dataset dari file CSV
  2. remove_duplicates()  — Menghapus baris duplikat
  3. remove_noise()       — Menghapus kategori noise ('Other' pada gender)
  4. encode_features()    — Encoding fitur kategorikal (Binary + Ordinal)
  5. scale_features()     — Standard scaling fitur numerik
  6. validate()           — Validasi hasil akhir
  7. export_data()        — Menyimpan dataset hasil preprocessing

Usage:
    python automate_Rajif-Athallah.py \
        --input  diabetes_prediction_dataset.csv \
        --output diabetes_prediction_preprocessing.csv
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  [%(levelname)s]  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('preprocessing.log', mode='w', encoding='utf-8'),
    ],
)
log = logging.getLogger(__name__)

# ─── Konstanta ────────────────────────────────────────────────────────────────
NUMERIC_COLS   = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
GENDER_MAP     = {'Female': 0, 'Male': 1}
SMOKING_ORDER  = {
    'No Info'    : 0,
    'never'      : 1,
    'ever'       : 2,
    'former'     : 3,
    'not current': 4,
    'current'    : 5,
}
TARGET_COL     = 'diabetes'


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load Data
# ──────────────────────────────────────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """
    Memuat dataset CSV ke dalam DataFrame pandas.

    Parameters
    ----------
    filepath : str
        Path ke file CSV dataset mentah.

    Returns
    -------
    pd.DataFrame
        DataFrame berisi data mentah.
    """
    log.info(f'[STEP 1] Memuat dataset dari: {filepath}')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'File tidak ditemukan: {filepath}')

    df = pd.read_csv(filepath)
    log.info(f'         Shape awal : {df.shape[0]:,} baris × {df.shape[1]} kolom')
    log.info(f'         Kolom      : {df.columns.tolist()}')
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — Remove Duplicates
# ──────────────────────────────────────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris duplikat dari dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.

    Returns
    -------
    pd.DataFrame
        DataFrame tanpa duplikat, index direset.
    """
    log.info('[STEP 2] Menghapus duplikat ...')
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    log.info(f'         Dihapus : {removed:,} baris  |  Tersisa : {len(df):,} baris')
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — Remove Noise
# ──────────────────────────────────────────────────────────────────────────────
def remove_noise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris dengan nilai kategori noise.

    Saat ini menangani:
      - gender == 'Other'  (jumlah sangat kecil, tidak representatif)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame input.

    Returns
    -------
    pd.DataFrame
        DataFrame setelah noise dihapus.
    """
    log.info("[STEP 3] Menghapus noise kategorikal (gender='Other') ...")
    before = len(df)
    df = df[df['gender'] != 'Other'].reset_index(drop=True)
    removed = before - len(df)
    log.info(f'         Dihapus : {removed:,} baris  |  Tersisa : {len(df):,} baris')
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — Encode Features
# ──────────────────────────────────────────────────────────────────────────────
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan encoding pada fitur kategorikal.

    - gender         : Binary Encoding  (Female=0, Male=1)
    - smoking_history: Ordinal Encoding (No Info < never < ever < former
                                         < not current < current)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame setelah noise removal.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan fitur kategorikal yang sudah di-encode.

    Raises
    ------
    ValueError
        Jika terdapat nilai yang tidak dikenal pada kolom kategorikal.
    """
    log.info('[STEP 4] Encoding fitur kategorikal ...')
    df = df.copy()

    # Binary Encoding: gender
    unknown_gender = set(df['gender'].unique()) - set(GENDER_MAP.keys())
    if unknown_gender:
        raise ValueError(f'Nilai gender tidak dikenal: {unknown_gender}')
    df['gender'] = df['gender'].map(GENDER_MAP)
    log.info(f'         gender          → Binary Encoding  {GENDER_MAP}')

    # Ordinal Encoding: smoking_history
    unknown_smoke = set(df['smoking_history'].unique()) - set(SMOKING_ORDER.keys())
    if unknown_smoke:
        raise ValueError(f'Nilai smoking_history tidak dikenal: {unknown_smoke}')
    df['smoking_history'] = df['smoking_history'].map(SMOKING_ORDER)
    log.info(f'         smoking_history → Ordinal Encoding  {SMOKING_ORDER}')

    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — Scale Features
# ──────────────────────────────────────────────────────────────────────────────
def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menerapkan StandardScaler pada fitur numerik kontinyu.

    Fitur yang di-scale: age, bmi, HbA1c_level, blood_glucose_level

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame setelah encoding.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan fitur numerik yang sudah di-scale.
    """
    log.info('[STEP 5] Feature scaling (StandardScaler) ...')
    df = df.copy()
    scaler = StandardScaler()
    df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])

    for col in NUMERIC_COLS:
        log.info(
            f'         {col:<25}  '
            f'mean={df[col].mean():.4f}  '
            f'std={df[col].std():.4f}'
        )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 6 — Validate
# ──────────────────────────────────────────────────────────────────────────────
def validate(df: pd.DataFrame) -> None:
    """
    Memvalidasi integritas dataset setelah preprocessing.

    Pemeriksaan:
      - Tidak ada missing values
      - Tidak ada baris duplikat
      - Semua kolom bertipe numerik
      - Kolom target hanya berisi 0 dan 1

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame hasil preprocessing.

    Raises
    ------
    AssertionError
        Jika salah satu pemeriksaan gagal.
    """
    log.info('[STEP 6] Validasi dataset ...')

    missing = df.isna().sum().sum()
    assert missing == 0, f'GAGAL: Terdapat {missing} missing values!'
    log.info('         ✓ Tidak ada missing values')

    dupes = df.duplicated().sum()
    assert dupes == 0, f'GAGAL: Terdapat {dupes} baris duplikat!'
    log.info('         ✓ Tidak ada duplikat')

    non_numeric = df.select_dtypes(include='object').columns.tolist()
    assert not non_numeric, f'GAGAL: Kolom non-numerik masih ada: {non_numeric}'
    log.info('         ✓ Semua kolom bertipe numerik')

    target_vals = set(df[TARGET_COL].unique())
    assert target_vals.issubset({0, 1}), \
        f'GAGAL: Kolom target memiliki nilai tidak terduga: {target_vals}'
    log.info(f'         ✓ Target valid: {sorted(target_vals)}')

    log.info(f'         Shape final : {df.shape[0]:,} baris × {df.shape[1]} kolom')


# ──────────────────────────────────────────────────────────────────────────────
# STEP 7 — Export Data
# ──────────────────────────────────────────────────────────────────────────────
def export_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Menyimpan dataset hasil preprocessing ke file CSV.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame final yang siap disimpan.
    output_path : str
        Path tujuan penyimpanan file CSV.
    """
    log.info(f'[STEP 7] Menyimpan dataset ke: {output_path}')
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    size_kb = os.path.getsize(output_path) / 1024
    log.info(f'         ✅ Berhasil disimpan  ({size_kb:.1f} KB)')


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE UTAMA
# ──────────────────────────────────────────────────────────────────────────────
def run_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Menjalankan seluruh pipeline preprocessing secara berurutan.

    Parameters
    ----------
    input_path : str
        Path ke dataset mentah (CSV).
    output_path : str
        Path tujuan dataset hasil preprocessing (CSV).

    Returns
    -------
    pd.DataFrame
        Dataset final yang sudah diproses.
    """
    log.info('=' * 60)
    log.info('  DIABETES PREDICTION — PREPROCESSING PIPELINE')
    log.info('=' * 60)

    df = load_data(input_path)
    df = remove_duplicates(df)
    df = remove_noise(df)
    df = encode_features(df)
    df = scale_features(df)
    validate(df)
    export_data(df, output_path)

    log.info('=' * 60)
    log.info('  PIPELINE SELESAI ✅')
    log.info('=' * 60)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Preprocessing pipeline untuk Diabetes Prediction Dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--input', '-i',
        default='diabetes_prediction_dataset.csv',
        help='Path ke file CSV dataset mentah.',
    )
    parser.add_argument(
        '--output', '-o',
        default='diabetes_prediction_preprocessing.csv',
        help='Path tujuan file CSV hasil preprocessing.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    try:
        run_pipeline(input_path=args.input, output_path=args.output)
    except Exception as exc:
        log.error(f'Pipeline gagal: {exc}', exc_info=True)
        sys.exit(1)
