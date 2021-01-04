from pathlib import Path

# More-or-less untouched data
DATA_DIR = Path('./data/')

ORIG_DIR = DATA_DIR / 'original'

PURE_DIR = ORIG_DIR / 'pure'
SPLIT_DIR = ORIG_DIR / 'mp_split'

PURE_TRAIN_PATH = PURE_DIR / 'Classify_train_processed_cleaned.csv'
PURE_VAL_PATH = PURE_DIR / 'Classify_validation_processed_cleaned.csv'

X_TRAIN_PATH = ORIG_DIR / 'train_data.csv'
Y_TRAIN_PATH = ORIG_DIR / 'train_targets.csv'
X_VAL_PATH = ORIG_DIR / 'val_data.csv'
Y_VAL_PATH = ORIG_DIR / 'val_targets.csv'

X_TRAIN_MP = SPLIT_DIR / 'train'
X_VAL_MP = SPLIT_DIR / 'val'


# Over/undersampled data
SAMP_DIR = DATA_DIR / 'sampling'
OVER_DIR = SAMP_DIR / 'over'
UNDER_DIR = SAMP_DIR / 'under'


# Masks (from feature selection)
MASK_DIR = DATA_DIR / 'masks'


# Transformed data
TSFM_DIR = DATA_DIR / 'transformed'
TSFM_TRAIN_DIR = TSFM_DIR / 'train'
TSFM_VAL_DIR = TSFM_DIR / 'val'


# Models
MODS_DIR = Path('./models/')
