
# Advanced Multimodal Training Strategies

This document outlines strategies to significantly improve performance in multimodal contrastive learning systems, focusing on image-text retrieval tasks. While our baseline system with semantic batch sampling can achieve reasonable results, this guide details potential improvements to push performance beyond 0.8+ accuracy toward state-of-the-art levels.

## Current Setup

Our baseline multimodal system uses:
- **Models**: ViT-base for vision, MobileBERT for text
- **Training**: 20 epochs with multistage approach
- **Data**: Flickr30k dataset with semantic group batch sampling
- **Hardware**: Apple Silicon (MPS)
- **Loss**: Standard contrastive loss with in-batch negatives

## Improvement Pathways

### 1. Better Pretrained Models

**Why**: Model capacity and pretraining significantly impact performance ceiling.

**What & Where**:
- **Text Models**: 
  ```python
  # In src/utils/argument_configs.py - Update choices
  parser.add_argument(
      "--text_model",
      choices=["transformer-base", ..., "roberta-large", "deberta-v3-large"],
  )
  
  # In demos/multimodal_training_demo.py - Command line
  --text_model roberta-base  # For better performance
  ```
  
- **Vision Models**:
  ```python
  # In src/models/model_factory.py - Add support for larger models
  if vision_model == "vit-large":
      model = timm.create_model("vit_large_patch16_224", pretrained=True)
  elif vision_model == "eva02-large":
      model = timm.create_model("eva02_large_patch14_336", pretrained=True)
  
  # In demos/multimodal_training_demo.py - Command line
  --vision_model vit-large
  ```

### 2. Data Augmentation

**Why**: Augmentation creates diverse training examples, improving generalization and reducing overfitting.

**What & Where**:
- **Image Augmentation**: 
  ```python
  # In src/data/multimodal_dataset.py - EnhancedMultimodalDataset
  from torchvision import transforms
  
  # Add strong augmentations in __init__ method
  self.transform_train = transforms.Compose([
      transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
      transforms.RandomGrayscale(p=0.1),
  ])
  
  # Use in __getitem__ method for training split
  if self.split == "train" and self.transform_train:
      image = self.transform_train(image)
  ```
  
- **Text Augmentation**:
  ```python
  # In src/data/multimodal_data_utils.py - Create a new text augmentation function
  def augment_text(text, p=0.3):
      """Apply simple text augmentations"""
      import random
      import nltk
      from nltk.corpus import wordnet
      
      if random.random() > p:
          return text
          
      words = text.split()
      if len(words) <= 3:
          return text
          
      # Simple word replacement or deletion
      for i in range(len(words)):
          if random.random() < 0.1:
              if random.random() < 0.5:
                  # Delete word
                  words[i] = ""
              else:
                  # Try synonym replacement
                  try:
                      synonyms = wordnet.synsets(words[i])
                      if synonyms and random.random() < 0.5:
                          words[i] = synonyms[0].lemmas()[0].name()
                  except:
                      pass
      
      return " ".join([w for w in words if w])
  ```
  
- **MixUp and CutMix**:
  ```python
  # In src/training/multimodal_trainer.py - Add mixup to training loop
  
  # Add to __init__
  self.mixup_alpha = 0.2  # Controls interpolation strength
  
  # In training step, before forward pass
  if self.training and self.mixup_alpha > 0:
      # Apply mixup to batch
      lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
      batch_size = images.size(0)
      index = torch.randperm(batch_size).to(images.device)
      
      # Mix images
      mixed_images = lam * images + (1 - lam) * images[index]
      
      # Update batch
      images = mixed_images
  ```

### 3. Advanced Training Techniques

**Why**: Training dynamics heavily influence final model performance.

**What & Where**:
- **Longer Training**:
  ```bash
  # Command line
  --num_epochs 40
  ```
  
- **Better Learning Rate Schedule**:
  ```python
  # In src/training/multimodal_trainer.py - Update optimizer configuration
  
  # In __init__ method
  self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
      self.optimizer,
      max_lr=learning_rate,
      total_steps=self.num_epochs * len(self.train_dataloader),
      pct_start=0.1,  # Spend 10% of steps warming up
      div_factor=25,  # Initial LR is max_lr/25
      final_div_factor=10000,  # End with very small LR
  )
  ```
  
- **Curriculum Learning**:
  ```python
  # In src/data/multimodal_dataset.py - Add curriculum learning
  
  # In EnhancedMultimodalDataset.__init__
  self.curriculum_weights = None
  if split == "train":
      # Initialize difficulty scores based on caption complexity
      self.difficulty_scores = [
          min(1.0, 0.3 + 0.7 * (len(item.get("captions", [""])[0].split()) / 20))
          for item in self.dataset
      ]
  
  # Add method to update curriculum weights
  def update_curriculum(self, epoch, total_epochs):
      """Update curriculum sampling weights based on training progress"""
      if self.split != "train":
          return
          
      # Start with easier examples, gradually include harder ones
      progress = min(1.0, epoch / (total_epochs * 0.7))
      threshold = 0.3 + progress * 0.7
      
      # Update sampling weights
      self.curriculum_weights = [
          1.0 if score <= threshold else 0.2
          for score in self.difficulty_scores
      ]
  ```
  
- **Larger Projection Dimension**:
  ```bash
  # Command line
  --projection_dim 512
  ```

### 4. Loss Function Improvements

**Why**: Better loss functions create more useful embeddings by improving the signal-to-noise ratio in training.

**What & Where**:
- **Memory Queue Contrastive Loss**:
  ```bash
  # Command line
  --loss_type memory_queue --queue_size 16384
  ```
  
- **Hard Negative Mining**:
  ```python
  # In src/training/multimodal_trainer.py - Add hard negative mining
  
  # In train_multistage method, for the final stage
  # Update loss function to hard negative mining for last stage
  if stage == 3:  # Final stage
      logger.info("Switching to hard negative mining for final stage")
      self.loss_fn = HardNegativeMiningContrastiveLoss(
          temperature=0.07,
          hard_negative_factor=2.0,
          mining_strategy="semi-hard", 
          dim=self.model.vision_dim
      )
  ```
  
- **Supervised Contrastive Learning** (if labels available):
  ```python
  # In src/training/contrastive_learning.py - Modify loss function
  
  # Add label support to contrastive loss forward method
  labels = torch.tensor([
      self.label_to_id.get(item.get("category", "unknown"), 0)
      for item in batch
  ]).to(device)
  
  # Use SupervisedContrastiveLoss from existing implementation
  ```

### 5. Ensemble Techniques

**Why**: Ensembles reduce variance and improve generalization by combining multiple perspectives.

**What & Where**:
- **Multiple Initializations**:
  ```python
  # In demos/multimodal_training_demo.py - Add ensemble capability
  
  # Train multiple models with different seeds
  trained_models = []
  for seed in [42, 43, 44]:
      set_seed(seed)
      model = create_multimodal_model(args, device)
      trainer = MultimodalTrainer(model=model, ...)
      trainer.train_multistage()
      trained_models.append(model)
  
  # Use ensemble for inference
  def ensemble_predict(models, image, text):
      """Combine predictions from multiple models"""
      similarities = []
      for model in models:
          with torch.no_grad():
              image_features = model.encode_image(image)
              text_features = model.encode_text(text)
              sim = (image_features @ text_features.T).cpu()
              similarities.append(sim)
              
      # Average similarities
      return torch.stack(similarities).mean(dim=0)
  ```
  
- **Model Distillation**:
  ```python
  # In src/training/losses.py - Add distillation loss
  
  class DistillationLoss(nn.Module):
      """Knowledge distillation loss"""
      def __init__(self, temp=2.0, alpha=0.5):
          super().__init__()
          self.temp = temp
          self.alpha = alpha
          
      def forward(self, student_logits, teacher_logits, labels=None):
          """
          Compute distillation loss between student and teacher
          """
          # Soften probabilities
          soft_targets = F.softmax(teacher_logits / self.temp, dim=1)
          soft_prob = F.log_softmax(student_logits / self.temp, dim=1)
          
          # Distillation loss
          dist_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * self.temp**2
          
          if labels is not None and self.alpha > 0:
              # Add cross entropy loss if labels available
              ce_loss = F.cross_entropy(student_logits, labels)
              return self.alpha * ce_loss + (1 - self.alpha) * dist_loss
          
          return dist_loss
  ```

## Implementation Prioritization

For practical implementation, prioritize these improvements in this order:

1. **Quick Wins (Immediate Impact)**:
   - Increase training epochs to 30-40
   - Switch to memory queue loss
   - Use all 5 captions per image
   - Increase projection dimension to 512

2. **Medium Effort (Strong Impact)**:
   - Add image augmentations
   - Implement OneCycle learning rate schedule
   - Try roberta-base instead of mobilebert

3. **Higher Effort (Maximum Impact)**:
   - Implement text augmentations
   - Add curriculum learning
   - Try larger vision models (hardware permitting)
   - Implement ensembles

## Estimated Performance Impact

| Strategy | Complexity | Est. Accuracy Gain |
|----------|------------|-------------------|
| Longer Training | Low | +5-10% |
| Memory Queue Loss | Low | +3-7% |
| Better Pretrained Models | Medium | +7-12% |
| Image Augmentations | Medium | +3-6% |
| Text Augmentations | High | +2-4% |
| Better LR Schedule | Low | +2-5% |
| Curriculum Learning | High | +1-3% |
| Ensembles | High | +3-5% |

With these improvements implemented strategically, reaching accuracy above 0.85 is feasible even on limited hardware.

