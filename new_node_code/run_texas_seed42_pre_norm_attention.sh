#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-analysis_results/texas_seed42_pre_norm_attention_best_models}"
SEED="${SEED:-42}"
SPLIT_INDEX="${SPLIT_INDEX:-0}"
EPOCHS="${EPOCHS:-1000}"
PATIENCE="${PATIENCE:-200}"

PRE_NORM_DIR="${OUTPUT_ROOT}/pre_norm_attention"
mkdir -p "$PRE_NORM_DIR"

ensure_texas_support() {
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path

path = Path("NodeClassification_CPU.py")
text = path.read_text()
original = text

text = text.replace(
    "choices=['Cora', 'CiteSeer', 'PubMed', 'Cornell'],",
    "choices=['Cora', 'CiteSeer', 'PubMed', 'Cornell', 'Texas', 'Wisconsin'],",
)

if "--split-index" not in text:
    text = text.replace(
        "    parser.add_argument('--dataset', type=str, default='PubMed', \n"
        "                       choices=['Cora', 'CiteSeer', 'PubMed', 'Cornell', 'Texas', 'Wisconsin'],\n"
        "                       help='Dataset to use')\n",
        "    parser.add_argument('--dataset', type=str, default='PubMed', \n"
        "                       choices=['Cora', 'CiteSeer', 'PubMed', 'Cornell', 'Texas', 'Wisconsin'],\n"
        "                       help='Dataset to use')\n"
        "    parser.add_argument('--split-index', type=int, default=0,\n"
        "                       help='Split column to use for WebKB multi-split masks.')\n",
    )

if "def select_split_masks(data, split_index):" not in text:
    split_helper = '''

def select_split_masks(data, split_index):
    """Select one official split from PyG multi-split masks."""
    required_masks = ('train_mask', 'val_mask', 'test_mask')
    if not all(hasattr(data, mask_name) for mask_name in required_masks):
        raise ValueError('Dataset does not provide train/val/test masks.')

    train_mask = data.train_mask
    if train_mask.dim() <= 1:
        if split_index != 0:
            raise ValueError(
                f'--split-index {split_index} was requested, but this dataset only has one split.'
            )
        return data

    num_splits = train_mask.size(1)
    if split_index < 0 or split_index >= num_splits:
        raise ValueError(
            f'--split-index {split_index} is out of range for this dataset; '
            f'valid split indices are 0..{num_splits - 1}.'
        )

    for mask_name in required_masks:
        mask = getattr(data, mask_name)
        if mask.dim() != 2 or mask.size(1) != num_splits:
            raise ValueError(
                f'{mask_name} has shape {tuple(mask.shape)}, expected a 2D mask '
                f'with {num_splits} split columns.'
            )
        setattr(data, mask_name, mask[:, split_index])
    return data
'''
    text = text.replace(
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n" + split_helper,
    )

text = text.replace(
    "    elif args.dataset == 'Cornell':\n"
    "        dataset = datasets.WebKB(root=data_path, name='Cornell')\n",
    "    elif args.dataset in ['Cornell', 'Texas', 'Wisconsin']:\n"
    "        dataset = datasets.WebKB(root=data_path, name=args.dataset)\n",
)

text = text.replace(
    "    # Handle multiple masks (e.g., in WebKB datasets like Cornell)\n"
    "    if hasattr(data, 'train_mask') and data.train_mask.dim() > 1:\n"
    "        data.train_mask = data.train_mask[:, 0]\n"
    "        data.val_mask = data.val_mask[:, 0]\n"
    "        data.test_mask = data.test_mask[:, 0]\n",
    "    data = select_split_masks(data, args.split_index)\n",
)

if 'print(f"Split index: {args.split_index}")' not in text:
    text = text.replace(
        '    print(f"Dataset: {args.dataset}")\n',
        '    print(f"Dataset: {args.dataset}")\n'
        '    print(f"Split index: {args.split_index}")\n',
    )

if 'print(f"Training nodes: {data.train_mask.sum().item()}")' not in text:
    text = text.replace(
        '    print(f"Clustered kernel features: {args.cluster_wl_features}")\n',
        '    print(f"Clustered kernel features: {args.cluster_wl_features}")\n'
        '    print(f"Training nodes: {data.train_mask.sum().item()}")\n'
        '    print(f"Validation nodes: {data.val_mask.sum().item()}")\n'
        '    print(f"Test nodes: {data.test_mask.sum().item()}")\n',
    )

if text != original:
    path.write_text(text)
    print("Patched NodeClassification_CPU.py for Texas WebKB split support.")
else:
    print("NodeClassification_CPU.py already has Texas WebKB split support.")
PY
}

run_symmetric() {
  "$PYTHON_BIN" NodeClassification_CPU.py \
    --dataset Texas \
    --split-index "$SPLIT_INDEX" \
    --num-layers 3 \
    --hop 3 \
    --wl 3 \
    --dim_hidden 32 \
    --lr 0.01 \
    --dropout 0.2 \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --numheads 1 \
    --GL_k 5 \
    --batch_size 32 \
    --kernels WL \
    --isgnn True \
    --cluster-wl-features \
    --n-clusters 64 \
    --seed "$SEED" \
    --outdir "${OUTPUT_ROOT}/symmetric_seed_${SEED}_run" \
    --params-str "True_1_False_WL_32_3_5_3l_3h_clustered_C64_0.2_0.01_32" \
    --save-pre-norm-attention \
    --pre-norm-attention-output-dir "$PRE_NORM_DIR" \
    --skip-curves
}

run_asymmetric() {
  "$PYTHON_BIN" NodeClassification_CPU.py \
    --dataset Texas \
    --split-index "$SPLIT_INDEX" \
    --num-layers 1 \
    --hop 3 \
    --wl 5 \
    --dim_hidden 128 \
    --lr 0.01 \
    --dropout 0.0 \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --numheads 1 \
    --GL_k 5 \
    --batch_size 32 \
    --kernels WL \
    --isgnn True \
    --cluster-wl-features \
    --n-clusters 16 \
    --seed "$SEED" \
    --outdir "${OUTPUT_ROOT}/asymmetric_seed_${SEED}_run" \
    --params-str "True_1_False_WL_128_5_5_1l_3h_clustered_C16_0.0_0.01_32" \
    --asymmetric-gate \
    --save-pre-norm-attention \
    --pre-norm-attention-output-dir "$PRE_NORM_DIR" \
    --skip-curves
}

ensure_texas_support

echo "Running Texas symmetric seed=${SEED} pre-norm attention export..."
run_symmetric

echo "Running Texas asymmetric seed=${SEED} pre-norm attention export..."
run_asymmetric

echo "Done. Pre-norm attention files are in: ${PRE_NORM_DIR}"
