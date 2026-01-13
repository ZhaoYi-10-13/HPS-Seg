#!/bin/bash
# 检查 A-150 评估结果

EVAL_DIR="output/enhanced_catseg/eval_ade150"
LOG_FILE="output/enhanced_catseg/eval_ade150_run.log"

echo "=== 检查评估状态 ==="
if [ -f "$EVAL_DIR/inference/sem_seg_evaluation.pth" ]; then
    echo "评估已完成！"
    echo ""
    echo "=== A-150 评估结果 ==="
    /venv/hps-seg/bin/python -c "
import torch
d = torch.load('$EVAL_DIR/inference/sem_seg_evaluation.pth', map_location='cpu')
print('mIoU: {:.4f}'.format(d.get('mIoU', 0)))
print('fwIoU: {:.4f}'.format(d.get('fwIoU', 0)))
print('mACC: {:.4f}'.format(d.get('mACC', 0)))
print('pACC: {:.4f}'.format(d.get('pACC', 0)))
"
    echo ""
    echo "=== copypaste 格式 ==="
    grep "copypaste" "$EVAL_DIR/log.txt" 2>/dev/null | tail -1
else
    echo "评估仍在进行中..."
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "=== 最新进度 ==="
        tail -5 "$LOG_FILE" | grep -E "Inference|done|iter" | tail -3
    fi
fi
