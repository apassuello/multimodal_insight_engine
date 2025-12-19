# 🍎 Hardware-Optimized Debugging Plan for M4 Pro / MPS

## Hardware Context

**Your Setup**:
- MacBook Pro M4 Pro
- 48GB RAM
- MPS (Metal Performance Shaders) backend
- Max practical batch size: **64** (larger causes severe performance degradation)

**Research context**:
- CLIP uses batch size 32,768 (512x larger than what you can use)
- Most papers assume NVIDIA GPUs with large VRAM

**Reality check**: You need MPS-optimized solutions!

---

## 🎯 **Revised Root Cause Analysis**

Given your hardware constraints, here's what's realistic:

### ✅ **Can Be Fixed** (High Priority)

1. **Match IDs broken** → Diagonal-only matching
   - Impact: **CRITICAL** - Will cause collapse regardless of batch size
   - Fix: Ensure match_ids group semantically similar samples
   - **FIX THIS FIRST**

2. **Temperature too low** (0.07 → should be 0.2-0.3 initially)
   - Impact: **HIGH** - Too sharp, causes early collapse
   - Fix: Increase to 0.2 or make learnable
   - **Easy win, high impact**

3. **Conflicting hot patches** in contrastive_loss.py
   - Impact: **MEDIUM** - Causes inconsistent behavior
   - Fix: Use clean implementation without patches
   - **Medium effort, prevents weird bugs**

### ⚠️ **Hardware Limitations** (Cannot Fix)

4. **Batch size limited to 64**
   - Impact: **MODERATE** - Fewer negatives per batch
   - Cannot fix: Larger batches kill performance on MPS
   - **Mitigation**: Use memory queue or gradient accumulation strategically

5. **MPS-specific issues**
   - Impact: **LOW to MODERATE** - Some operations slower/different on MPS
   - Cannot fix: Hardware limitation
   - **Mitigation**: Use MPS-optimized models (ViT, not BERT)

### 📊 **Realistic Expectations with Batch Size 64**

**Research findings**:
- SigLIP achieves good results with batch sizes as low as 256
- With proper techniques, batch size 64 can work but requires:
  - **Perfect match_id grouping** (critical!)
  - **Memory queue** to increase effective negatives
  - **Higher temperature** to compensate for fewer negatives
  - **More epochs** (slower convergence)

**Your Flickr30k Setup**:
- 30k images × 5 captions = 150k samples
- Batch size 64 with gradient accumulation steps 4 = effective batch 256
- BUT: MPS might make gradient accumulation slow too

---

## 🔧 **MPS-Optimized Debugging Plan**

### **Phase 1: Critical Fixes** (Do These First)

#### Fix 1.1: Match IDs (CRITICAL!)
```bash
# Run diagnostic
python debug_scripts/diagnose_match_ids.py

# Expected: ~12-13 images per batch (64 samples ÷ 5 captions/image)
# If all match_ids unique → THIS IS YOUR BUG
```

#### Fix 1.2: Temperature (EASY WIN!)
```python
# In contrastive_loss.py, change:
self.temperature = 0.07  # Too low for batch size 64

# To (for MPS with small batches):
self.temperature = 0.3  # Higher temp compensates for fewer negatives
# Or make it learnable:
self.log_temperature = nn.Parameter(torch.log(torch.tensor(1.0 / 0.3)))
```

**Why**: With batch size 64, you have ~60 negatives per sample (vs 32,767 in CLIP). Higher temperature makes the distribution softer, preventing premature collapse.

#### Fix 1.3: Remove Hot Patches
Use the `SimpleContrastiveLoss` from `minimal_training_test.py` - it's clean and MPS-friendly.

---

### **Phase 2: MPS-Specific Optimizations**

#### Optimization 2.1: Memory Queue (Increases Negatives Without Batch Size)

```python
class MPSOptimizedContrastiveLoss(nn.Module):
    """
    Contrastive loss optimized for small batches on MPS.
    Uses memory queue to increase effective negatives.
    """

    def __init__(self, temperature=0.3, queue_size=2048):
        super().__init__()
        self.temperature = temperature
        self.queue_size = queue_size

        # Memory queue (stores past embeddings)
        self.register_buffer("vision_queue", torch.randn(queue_size, 768))
        self.register_buffer("text_queue", torch.randn(queue_size, 768))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Normalize queue
        self.vision_queue = F.normalize(self.vision_queue, p=2, dim=1)
        self.text_queue = F.normalize(self.text_queue, p=2, dim=1)

    def forward(self, vision_features, text_features, match_ids):
        batch_size = vision_features.shape[0]

        # Normalize
        vision_features = F.normalize(vision_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # In-batch similarity
        similarity_batch = torch.matmul(vision_features, text_features.T) / self.temperature

        # Queue similarity (additional negatives)
        similarity_queue = torch.matmul(vision_features, self.text_queue.T) / self.temperature

        # Combine: [batch_size, batch_size + queue_size]
        similarity = torch.cat([similarity_batch, similarity_queue], dim=1)

        # Build targets (only in-batch positives)
        targets = []
        for i in range(batch_size):
            matches = [j for j in range(batch_size) if match_ids[i] == match_ids[j]]
            targets.append(matches[0] if matches else i)

        targets = torch.tensor(targets, dtype=torch.long, device=similarity.device)

        # Cross-entropy loss
        loss = F.cross_entropy(similarity, targets)

        # Update queue (detach to prevent backprop through queue)
        with torch.no_grad():
            self._update_queue(vision_features, text_features)

        return loss

    @torch.no_grad()
    def _update_queue(self, vision_features, text_features):
        batch_size = vision_features.shape[0]
        ptr = int(self.queue_ptr)

        # Circular queue update
        if ptr + batch_size > self.queue_size:
            # Wrap around
            remaining = self.queue_size - ptr
            self.vision_queue[ptr:] = vision_features[:remaining]
            self.text_queue[ptr:] = text_features[:remaining]

            overflow = batch_size - remaining
            self.vision_queue[:overflow] = vision_features[remaining:]
            self.text_queue[:overflow] = text_features[remaining:]

            ptr = overflow
        else:
            self.vision_queue[ptr:ptr+batch_size] = vision_features
            self.text_queue[ptr:ptr+batch_size] = text_features
            ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr
```

**Benefit**: With queue size 2048, you get **2048 extra negatives** without increasing batch size!
- In-batch: 64 samples
- Queue: 2048 samples
- **Total effective negatives: ~2100** (vs 64 without queue)

#### Optimization 2.2: MPS-Friendly Model Selection

```python
# AVOID these on MPS (cause issues):
text_model = "bert-base-uncased"  # Slow on MPS
text_model = "google/mobilebert"  # Has MPS compatibility issues

# USE these on MPS:
text_model = "albert-base-v2"  # Faster and more compatible
vision_model = "google/vit-base-patch16-224"  # Native MPS support
```

#### Optimization 2.3: Mixed Precision (MPS FP16)

```python
# Enable MPS mixed precision
if torch.backends.mps.is_available():
    # MPS supports fp16
    scaler = torch.cuda.amp.GradScaler()  # Works on MPS too

    with torch.cuda.amp.autocast():  # Also works on MPS
        loss = model(inputs)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Benefit**: ~30-40% memory reduction → can fit slightly larger batches or models

#### Optimization 2.4: Gradient Checkpointing

```python
# For vision transformer (saves memory)
from torch.utils.checkpoint import checkpoint

class CheckpointedViT(nn.Module):
    def forward(self, x):
        # Checkpoint transformer layers
        x = checkpoint(self.transformer_layer1, x)
        x = checkpoint(self.transformer_layer2, x)
        return x
```

**Benefit**: Trades computation for memory → might enable batch size 96

---

### **Phase 3: Realistic Training Strategy**

Given your hardware, here's a realistic training approach:

#### Strategy 3.1: Staged Batch Size Scaling

```python
# Stage 1: Warm-up with small batches
stage1_config = {
    'batch_size': 32,  # Very small, fast iterations
    'temperature': 0.4,  # Higher temp for fewer negatives
    'epochs': 5,
    'purpose': 'Initialize embeddings, prevent early collapse'
}

# Stage 2: Medium batches
stage2_config = {
    'batch_size': 64,  # Your max
    'temperature': 0.3,  # Slightly lower
    'queue_size': 2048,  # Add memory queue
    'epochs': 20,
    'purpose': 'Main training with queue negatives'
}

# Stage 3: Fine-tuning
stage3_config = {
    'batch_size': 48,  # Smaller for stability
    'temperature': 0.2,  # Lower for sharper gradients
    'queue_size': 4096,  # Larger queue
    'epochs': 10,
    'purpose': 'Refine representations'
}
```

#### Strategy 3.2: Simplified Model for Testing

Start with a **much simpler model** to validate training works:

```python
# Instead of full ViT + BERT:
vision_encoder = nn.Sequential(
    nn.Conv2d(3, 64, 7, stride=2),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((7, 7)),
    nn.Flatten(),
    nn.Linear(64 * 7 * 7, 768)
)

text_encoder = nn.Embedding(10000, 768)  # Simple embedding

# This should train FAST on MPS
# If this works but full model doesn't → problem is in pretrained models
# If this also fails → problem is in loss/dataloader
```

---

## 📋 **MPS-Specific Checklist**

### **Today:**
- [ ] Run `diagnose_match_ids.py` - verify match_ids are correct
- [ ] Change temperature from 0.07 to 0.3
- [ ] Run `minimal_training_test.py` - verify loss works
- [ ] Check MPS device compatibility: `torch.backends.mps.is_available()`

### **This Week:**
- [ ] Implement memory queue loss (if match_ids are fixed)
- [ ] Switch to albert-base-v2 (better MPS compatibility)
- [ ] Test with simplified model first (validate training loop)
- [ ] Enable MPS mixed precision
- [ ] Train for 50 steps, check if gap > 0.2

### **Next:**
- [ ] Gradually scale up model complexity
- [ ] Monitor MPS memory usage
- [ ] Compare performance: queue vs gradient accumulation
- [ ] Optimize data loading (might be bottleneck on MPS)

---

## 🎯 **Success Criteria (Adjusted for Batch Size 64)**

### **Acceptable Performance**

With batch size 64 + memory queue:

1. **Similarity metrics**:
   - ✅ Diagonal similarity: **0.4-0.7** (vs 0.6-0.9 for large batches)
   - ✅ Off-diagonal similarity: **0.0-0.2**
   - ✅ Gap: **> 0.2** (vs > 0.3 for large batches)

2. **Training dynamics**:
   - ✅ Loss decreases (even if slower than papers report)
   - ✅ No NaN/Inf values
   - ✅ Gap increases over time (even if gradually)

3. **Feature statistics**:
   - ✅ Feature variance > 0.3
   - ✅ No warnings about dimensional collapse
   - ✅ Temperature adjusts (if learnable)

### **Timeline Expectations**

With your hardware:
- **100 steps**: Should see gap > 0.1
- **500 steps**: Should see gap > 0.2
- **2000 steps**: Should see gap > 0.3 (if everything is working)

Compare to papers: **You'll need 2-3x more steps** due to smaller batch size.

---

## 💡 **If Memory Queue Doesn't Help**

Alternative approaches for small batches:

### Option A: Momentum Contrast (MoCo)
```python
# Use EMA encoder for queue
self.momentum_encoder = copy.deepcopy(encoder)
self.m = 0.999  # Momentum coefficient

# Update momentum encoder
for param_q, param_k in zip(encoder.parameters(), self.momentum_encoder.parameters()):
    param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
```

### Option B: SwAV / Clustering
```python
# Cluster embeddings instead of using InfoNCE
from sklearn.cluster import KMeans

# Every N steps:
cluster_assignments = KMeans(n_clusters=1000).fit_predict(all_embeddings)
# Use cluster IDs as targets
```

### Option C: Barlow Twins (No Negatives!)
```python
# Doesn't require large batches
# Decorrelates dimensions instead of using negatives
# See: src/training/losses/barlow_twins_loss.py (you already have this!)
```

---

## 🚀 **Quick Start for Today**

### Step 1: Verify Your Setup
```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available()); print('MPS built:', torch.backends.mps.is_built())"
```

### Step 2: Run Diagnostics
```bash
python debug_scripts/diagnose_match_ids.py  # Check match_ids
python debug_scripts/minimal_training_test.py  # Test loss function
```

### Step 3: If Diagnostics Pass
```python
# Quick test with your real data but simple model
from src.training.losses.barlow_twins_loss import BarlowTwinsLoss

# Try Barlow Twins instead of InfoNCE
# It's designed for small batches!
loss_fn = BarlowTwinsLoss(embedding_dim=768, lambda_param=0.005)
```

---

## 🎓 **Research on Small-Batch Contrastive Learning**

Papers to look at:
1. **Barlow Twins** (ICML 2021) - Works with batch size 256
2. **BYOL** (NeurIPS 2020) - No negatives needed
3. **SimSiam** (CVPR 2021) - Simple, works with small batches
4. **MoCo v2** (arXiv 2020) - Memory queue for small batches
5. **SigLIP** (ICCV 2023) - Better than CLIP for small batches

**Key insight**: Recent methods (2021-2023) are designed for smaller batches than CLIP!

---

## ✅ **Bottom Line for M4 Pro**

1. **Fix match_ids** → CRITICAL regardless of batch size
2. **Increase temperature to 0.3** → Easy win for small batches
3. **Add memory queue** → Gets you 30x more negatives
4. **Consider Barlow Twins** → Designed for small batches
5. **Be patient** → Will take 2-3x more steps than papers report

Your hardware CAN train multimodal models, just need the right techniques!

Good luck! 🍀
