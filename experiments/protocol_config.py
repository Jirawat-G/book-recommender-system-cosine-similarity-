# experiments/protocol_config.py

# -----------------------------
# Classification protocol
# -----------------------------
CLASSIFICATION_PRIMARY_METRIC = "test_f1_weighted"
CLASSIFICATION_SECONDARY_METRIC = "test_accuracy"

# -----------------------------
# Recommendation protocol
# -----------------------------
# ใช้ metric หลักสำหรับเลือก strategy สุดท้าย
STRATEGY_PRIMARY_METRIC = "f1@5"
STRATEGY_SECONDARY_METRIC = "hit@5"
STRATEGY_TERTIARY_METRIC = "mrr"

# metric แบบคนละมุมมอง
PRECISION_ORIENTED_METRIC = "f1@5"
HIT_ORIENTED_METRIC = "hit@5"

# -----------------------------
# Deployment decision
# -----------------------------
# ถ้า top1 ชนะชัดเจน ให้ใช้ top1
# ถ้า top2 ใกล้เคียงกับ top1 มาก และ hit@5 ดีกว่า ให้แนะนำ top2
DEPLOYMENT_MAX_F1_DROP_FOR_TOP2 = 0.03
PREFER_TOP2_IF_HIT_IMPROVES = True

# strategy prefixes
COSINE_ONLY_NAME = "cosine_only"
HYBRID_PREFIX = "hybrid_top"
CLASSIFY_ONLY_NAME = "classify_only_top1"