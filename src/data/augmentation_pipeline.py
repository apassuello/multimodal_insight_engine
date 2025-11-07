"""
Augmentation pipeline for multimodal data.

This module provides comprehensive augmentation capabilities for multimodal
datasets, with configurable strategies for both image and text modalities.
"""

import logging
import random
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageOps
from torchvision.transforms import functional as F

logger = logging.getLogger(__name__)

class MultimodalAugmentationPipeline:
    """
    Advanced augmentation pipeline for multimodal data with configurable strategies
    for both image and text modalities.
    """

    def __init__(
        self,
        image_augs: Optional[List[Callable]] = None,
        text_augs: Optional[List[Callable]] = None,
        image_aug_prob: float = 0.5,
        text_aug_prob: float = 0.3,
        consistency_mode: str = "matched",
        image_size: int = 224,
        severity: str = "medium",
        random_erasing_prob: float = 0.0,
        random_erasing_scale: Tuple[float, float] = (0.02, 0.33),
        color_jitter_prob: float = 0.0,
        random_resized_crop: bool = True,
        debug_mode: bool = False
    ):
        """
        Initialize the multimodal augmentation pipeline.
        
        Args:
            image_augs: Optional custom list of image augmentations
            text_augs: Optional custom list of text augmentations
            image_aug_prob: Probability of applying image augmentations
            text_aug_prob: Probability of applying text augmentations
            consistency_mode: Mode for maintaining consistency ("matched", "independent", or "paired")
            image_size: Target image size
            severity: Augmentation severity ("light", "medium", or "heavy")
            random_erasing_prob: Probability of random erasing
            random_erasing_scale: Scale range for random erasing
            color_jitter_prob: Probability of applying color jitter
            random_resized_crop: Whether to use random resized crop
            debug_mode: Whether to log detailed augmentation information
        """
        self.image_aug_prob = image_aug_prob
        self.text_aug_prob = text_aug_prob
        self.consistency_mode = consistency_mode
        self.image_size = image_size
        self.debug_mode = debug_mode
        self.severity = severity

        # Set augmentation intensity based on severity
        if severity == "light":
            color_strength = 0.3
            distortion_strength = 0.1
            erasing_prob = random_erasing_prob if random_erasing_prob > 0 else 0.1
            crop_scale = (0.8, 1.0)
        elif severity == "medium":
            color_strength = 0.5
            distortion_strength = 0.2
            erasing_prob = random_erasing_prob if random_erasing_prob > 0 else 0.2
            crop_scale = (0.7, 1.0)
        else:  # heavy
            color_strength = 0.7
            distortion_strength = 0.3
            erasing_prob = random_erasing_prob if random_erasing_prob > 0 else 0.3
            crop_scale = (0.6, 1.0)

        # Define default image augmentations if none provided
        if image_augs is None:
            # Common visual transforms
            base_transforms = []

            # Random resized crop (respects aspect ratio while zooming in)
            if random_resized_crop:
                base_transforms.append(
                    T.RandomResizedCrop(
                        image_size,
                        scale=crop_scale,
                        ratio=(3/4, 4/3)
                    )
                )
            else:
                # If not using random crop, use resize and random crop
                base_transforms.append(T.Resize(int(image_size * 1.1)))
                base_transforms.append(T.RandomCrop(image_size))

            # Horizontal flip (not always semantically valid)
            base_transforms.append(T.RandomHorizontalFlip(p=0.5))

            # Color jitter transforms
            color_jitter = T.ColorJitter(
                brightness=color_strength,
                contrast=color_strength,
                saturation=color_strength,
                hue=color_strength/2  # Hue usually has smaller range
            )

            # Apply color jitter with specified probability
            cj_prob = color_jitter_prob if color_jitter_prob > 0 else 0.8
            base_transforms.append(T.RandomApply([color_jitter], p=cj_prob))

            # Random grayscale conversion
            base_transforms.append(T.RandomGrayscale(p=distortion_strength))

            # Random posterization
            base_transforms.append(RandomPosterize(bits=4, p=distortion_strength))

            # Gaussian blur
            base_transforms.append(
                T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=distortion_strength)
            )

            # Random erasing (acts like localized occlusion/dropout)
            if erasing_prob > 0:
                base_transforms.append(
                    T.RandomErasing(
                        p=erasing_prob,
                        scale=random_erasing_scale,
                        ratio=(0.3, 3.3),
                        value=0
                    )
                )

            # Combine all transforms
            self.image_augs = T.Compose(base_transforms)
        else:
            self.image_augs = image_augs

        # Default text augmentations if none provided
        if text_augs is None:
            self.text_augs = [
                DropWords(prob=distortion_strength),
                ShuffleWords(prob=distortion_strength/2),
                ReplaceWithSynonym(prob=distortion_strength/3),
                ChangeWordOrder(prob=distortion_strength/2),
                AddMisspelling(prob=distortion_strength/3)
            ]
        else:
            self.text_augs = text_augs

        # For counting augmentations (debug only)
        if self.debug_mode:
            self.image_aug_counter = 0
            self.text_aug_counter = 0
            self.total_samples = 0

    def augment_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply image augmentations with probability.
        
        Args:
            image: Input image tensor [C, H, W]
            
        Returns:
            Augmented image tensor [C, H, W]
        """
        if random.random() < self.image_aug_prob:
            # If tensor, convert to PIL for augmentation
            if isinstance(image, torch.Tensor):
                # Save original image format
                if image.dtype == torch.float and image.max() <= 1.0:
                    # Normalized image - denormalize first
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

                    # Clone to avoid modifying original
                    img = image.clone().cpu()
                    img = img * std + mean
                    img = img.clamp(0, 1) * 255
                    img = img.to(torch.uint8).permute(1, 2, 0).numpy()
                    img = Image.fromarray(img)
                else:
                    # Already in the right format
                    img = F.to_pil_image(image)

                # Apply augmentations
                img = self.image_augs(img)

                # Convert back to tensor
                augmented = F.to_tensor(img)

                # Re-normalize if original was normalized
                if image.dtype == torch.float and image.max() <= 1.0:
                    augmented = F.normalize(augmented, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                if self.debug_mode:
                    self.image_aug_counter += 1

                return augmented
            else:
                # Already a PIL Image or other format - apply directly
                augmented = self.image_augs(image)

                if self.debug_mode:
                    self.image_aug_counter += 1

                return augmented

        return image

    def augment_text(self, text_data: Union[str, Dict]) -> Union[str, Dict]:
        """
        Apply text augmentations with probability.
        
        Args:
            text_data: Input text or text data dictionary
            
        Returns:
            Augmented text or text data dictionary
        """
        if not self.text_augs or random.random() >= self.text_aug_prob:
            return text_data

        # Extract the text from different input formats
        if isinstance(text_data, str):
            text = text_data
            is_raw_text = True
        elif isinstance(text_data, dict) and "raw_text" in text_data:
            text = text_data["raw_text"]
            is_raw_text = False
        else:
            # Can't augment this format
            return text_data

        # Apply text augmentations
        for aug in self.text_augs:
            if random.random() < aug.prob:
                text = aug(text)

                if self.debug_mode:
                    self.text_aug_counter += 1

        # Return in the same format as input
        if is_raw_text:
            return text
        else:
            # Update the raw text in the dictionary
            updated_data = text_data.copy()
            updated_data["raw_text"] = text

            # If we need to update tokenized data, we'd do it here
            # For now, we'll assume the model will retokenize the raw text
            return updated_data

    def __call__(
        self,
        batch: Dict[str, Union[torch.Tensor, Dict]]
    ) -> Dict[str, Union[torch.Tensor, Dict]]:
        """
        Apply augmentations to a batch of multimodal data.
        
        Args:
            batch: Dictionary with 'image' and 'text' keys
            
        Returns:
            Augmented batch with same structure
        """
        result = {}

        if self.debug_mode:
            self.total_samples += 1

            # Log augmentation rate every 100 batches
            if self.total_samples % 100 == 0:
                img_rate = self.image_aug_counter / self.total_samples
                txt_rate = self.text_aug_counter / self.total_samples
                logger.info(f"Augmentation rates - Image: {img_rate:.2f}, Text: {txt_rate:.2f}")

        # Determine consistent augmentation decision in matched mode
        if self.consistency_mode == "matched":
            # Apply both or neither
            apply_both = random.random() < max(self.image_aug_prob, self.text_aug_prob)

        # Augment images if present
        if "image" in batch:
            # Handle different data formats
            if isinstance(batch["image"], torch.Tensor):
                if batch["image"].dim() == 3:  # Single image: [C, H, W]
                    # In matched mode, decision is already made
                    if self.consistency_mode == "matched" and apply_both:
                        result["image"] = self.augment_image(batch["image"])
                    elif self.consistency_mode != "matched":
                        # Independent decision
                        result["image"] = self.augment_image(batch["image"])
                    else:
                        # Matched mode but decided not to augment
                        result["image"] = batch["image"]
                else:  # Batch of images: [B, C, H, W]
                    # Apply augmentation to each image separately
                    augmented = []
                    for img in batch["image"]:
                        if self.consistency_mode == "matched" and apply_both:
                            augmented.append(self.augment_image(img))
                        elif self.consistency_mode != "matched":
                            augmented.append(self.augment_image(img))
                        else:
                            augmented.append(img)

                    result["image"] = torch.stack(augmented)
            else:
                # Not a tensor - pass through
                result["image"] = batch["image"]

        # Augment text if present
        if "text" in batch:
            # Handle different text data formats
            if isinstance(batch["text"], str):
                # Raw text
                if self.consistency_mode == "matched" and apply_both:
                    result["text"] = self.augment_text(batch["text"])
                elif self.consistency_mode != "matched":
                    result["text"] = self.augment_text(batch["text"])
                else:
                    result["text"] = batch["text"]
            elif isinstance(batch["text"], dict):
                # Dictionary format with tokenized data
                if "raw_text" in batch["text"]:
                    # Has raw text - augment it
                    if self.consistency_mode == "matched" and apply_both:
                        result["text"] = self.augment_text(batch["text"])
                    elif self.consistency_mode != "matched":
                        result["text"] = self.augment_text(batch["text"])
                    else:
                        result["text"] = batch["text"]
                else:
                    # No raw text, can't augment
                    result["text"] = batch["text"]
            elif isinstance(batch["text"], list):
                # List of texts
                augmented = []
                for txt in batch["text"]:
                    if self.consistency_mode == "matched" and apply_both:
                        augmented.append(self.augment_text(txt))
                    elif self.consistency_mode != "matched":
                        augmented.append(self.augment_text(txt))
                    else:
                        augmented.append(txt)

                result["text"] = augmented
            else:
                # Unknown format - pass through
                result["text"] = batch["text"]

        # Copy any other keys unchanged
        for k, v in batch.items():
            if k not in result:
                result[k] = v

        return result


# Additional image augmentations

class RandomPosterize(nn.Module):
    """
    Apply posterization to an image with a given probability.
    Reduces the number of bits for each color channel.
    """

    def __init__(self, bits: int = 4, p: float = 0.5):
        """
        Initialize the posterize transform.
        
        Args:
            bits: Number of bits to keep for each channel
            p: Probability of applying the transform
        """
        super().__init__()
        self.bits = bits
        self.p = p

    def forward(self, img):
        """Apply the transform to the image."""
        if random.random() < self.p:
            return ImageOps.posterize(img, self.bits)
        return img


# Text augmentation classes

class TextAugmentation:
    """Base class for text augmentation transforms."""

    def __init__(self, prob: float = 0.5):
        """
        Initialize text augmentation with probability.
        
        Args:
            prob: Probability of applying this augmentation
        """
        self.prob = prob

    def __call__(self, text: str) -> str:
        """
        Apply the augmentation to the text.
        
        Args:
            text: Input text to augment
            
        Returns:
            Augmented text
        """
        raise NotImplementedError


class DropWords(TextAugmentation):
    """Randomly drop words from the text."""

    def __init__(self, prob: float = 0.5, drop_prob: float = 0.1, max_drops: int = 3):
        """
        Initialize the word dropping transform.
        
        Args:
            prob: Probability of applying this augmentation
            drop_prob: Probability of dropping each word
            max_drops: Maximum number of words to drop
        """
        super().__init__(prob)
        self.drop_prob = drop_prob
        self.max_drops = max_drops

    def __call__(self, text: str) -> str:
        words = text.split()

        # Don't drop if we have very few words
        if len(words) <= 3:
            return text

        drops = 0
        result = []

        for word in words:
            if drops < self.max_drops and random.random() < self.drop_prob:
                drops += 1
                continue
            result.append(word)

        # Ensure we don't drop all words
        if not result:
            # Keep a random word
            idx = random.randint(0, len(words) - 1)
            result = [words[idx]]

        return " ".join(result)


class ShuffleWords(TextAugmentation):
    """Shuffle some words in the text."""

    def __init__(self, prob: float = 0.3, window_size: int = 3):
        """
        Initialize the word shuffling transform.
        
        Args:
            prob: Probability of applying this augmentation
            window_size: Size of windows to shuffle within
        """
        super().__init__(prob)
        self.window_size = window_size

    def __call__(self, text: str) -> str:
        words = text.split()

        # Don't shuffle if we have very few words
        if len(words) <= 3:
            return text

        # Choose a random window to shuffle
        if len(words) <= self.window_size:
            window_start = 0
            window_end = len(words)
        else:
            window_start = random.randint(0, len(words) - self.window_size)
            window_end = window_start + self.window_size

        # Get the window words and shuffle them
        window = words[window_start:window_end]
        random.shuffle(window)

        # Reconstruct the text
        result = words[:window_start] + window + words[window_end:]
        return " ".join(result)


class ReplaceWithSynonym(TextAugmentation):
    """Replace words with simple synonyms."""

    def __init__(self, prob: float = 0.3, replace_prob: float = 0.2):
        """
        Initialize the synonym replacement transform.
        
        Args:
            prob: Probability of applying this augmentation
            replace_prob: Probability of replacing each word
        """
        super().__init__(prob)
        self.replace_prob = replace_prob

        # Simple synonym dictionary for common words
        self.synonyms = {
            "small": ["tiny", "little", "compact"],
            "big": ["large", "huge", "enormous"],
            "happy": ["glad", "joyful", "content"],
            "sad": ["unhappy", "upset", "gloomy"],
            "beautiful": ["pretty", "lovely", "attractive"],
            "ugly": ["unattractive", "plain", "hideous"],
            "fast": ["quick", "rapid", "swift"],
            "slow": ["gradual", "unhurried", "sluggish"],
            "good": ["great", "excellent", "fine"],
            "bad": ["poor", "terrible", "awful"],
            "person": ["individual", "human", "man", "woman"],
            "car": ["vehicle", "automobile", "ride"],
            "house": ["home", "residence", "dwelling"],
            "dog": ["canine", "pooch", "pup"],
            "cat": ["feline", "kitty", "kitten"],
            "walk": ["stroll", "wander", "hike"],
            "run": ["jog", "sprint", "dash"],
            "eat": ["consume", "devour", "munch"],
            "drink": ["sip", "gulp", "quaff"],
            "see": ["observe", "view", "witness"],
            "hear": ["listen", "perceive", "detect"],
            "red": ["crimson", "scarlet", "ruby"],
            "blue": ["azure", "cobalt", "navy"],
            "green": ["emerald", "olive", "jade"],
            "orange": ["amber", "tangerine", "copper"],
            "purple": ["violet", "lavender", "indigo"],
            "yellow": ["golden", "amber", "lemon"],
            "white": ["pale", "ivory", "cream"],
            "black": ["dark", "ebony", "jet"]
        }

    def __call__(self, text: str) -> str:
        words = text.split()
        result = []

        for word in words:
            # Convert to lowercase for matching
            word_lower = word.lower()

            if word_lower in self.synonyms and random.random() < self.replace_prob:
                # Choose a random synonym
                synonyms = self.synonyms[word_lower]
                synonym = random.choice(synonyms)

                # Match case of original word
                if word.istitle():
                    synonym = synonym.title()
                elif word.isupper():
                    synonym = synonym.upper()

                result.append(synonym)
            else:
                result.append(word)

        return " ".join(result)


class ChangeWordOrder(TextAugmentation):
    """Change the order of phrases in the text."""

    def __init__(self, prob: float = 0.2):
        """
        Initialize the word order change transform.
        
        Args:
            prob: Probability of applying this augmentation
        """
        super().__init__(prob)

    def __call__(self, text: str) -> str:
        # Check if the text has a common conjunction that can be used for swapping
        conjunctions = [" and ", " but ", " or ", " while ", " as "]

        for conj in conjunctions:
            if conj in text:
                parts = text.split(conj, 1)
                if len(parts) == 2 and len(parts[0].split()) >= 2 and len(parts[1].split()) >= 2:
                    # Swap the parts
                    return parts[1] + conj + parts[0]

        # If no suitable conjunction found, try to reorder phrases with commas
        if ", " in text:
            comma_parts = text.split(", ")
            if len(comma_parts) >= 2:
                # Move the last part to the beginning
                last_part = comma_parts.pop()
                return last_part + ", " + ", ".join(comma_parts)

        # No suitable pattern for reordering found
        return text


class AddMisspelling(TextAugmentation):
    """Add simple misspellings to text."""

    def __init__(self, prob: float = 0.2, char_prob: float = 0.05):
        """
        Initialize the misspelling transform.
        
        Args:
            prob: Probability of applying this augmentation
            char_prob: Probability of modifying each character
        """
        super().__init__(prob)
        self.char_prob = char_prob

        # Common character swaps for misspellings
        self.char_swaps = {
            'a': 'ae', 'e': 'ea', 'i': 'ie', 'o': 'ou', 'u': 'uo',
            's': 'ss', 't': 'tt', 'l': 'll', 'c': 'k', 'k': 'c'
        }

    def __call__(self, text: str) -> str:
        words = text.split()
        result = []

        for word in words:
            # Only consider words of length 4 or more for misspelling
            if len(word) >= 4 and random.random() < self.char_prob:
                # Choose a random position to modify
                pos = random.randint(1, len(word) - 2)  # Avoid first and last letters

                # Get the character at that position
                char = word[pos].lower()

                # Apply a random transformation
                transform_type = random.randint(0, 3)

                if transform_type == 0 and char in self.char_swaps:
                    # Swap with common misspelling
                    new_word = word[:pos] + self.char_swaps[char] + word[pos+1:]
                elif transform_type == 1:
                    # Duplicate a character
                    new_word = word[:pos] + word[pos] + word[pos:]
                elif transform_type == 2 and pos+1 < len(word):
                    # Swap adjacent characters
                    new_word = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
                else:
                    # Skip a character
                    new_word = word[:pos] + word[pos+1:]

                result.append(new_word)
            else:
                result.append(word)

        return " ".join(result)


def extract_file_metadata(file_path=__file__):
    """
    Extract structured metadata about this module.
    
    Args:
        file_path: Path to the source file (defaults to current file)
        
    Returns:
        dict: Structured metadata about the module's purpose and components
    """
    import os

    return {
        "filename": os.path.basename(file_path),
        "module_purpose": "Provides comprehensive augmentation capabilities for multimodal datasets",
        "key_classes": [
            {
                "name": "MultimodalAugmentationPipeline",
                "purpose": "Configurable pipeline for applying augmentations to both image and text data",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, image_augs: Optional[List[Callable]] = None, text_augs: Optional[List[Callable]] = None, image_aug_prob: float = 0.5, text_aug_prob: float = 0.3, consistency_mode: str = 'matched', image_size: int = 224, severity: str = 'medium', random_erasing_prob: float = 0.0, random_erasing_scale: Tuple[float, float] = (0.02, 0.33), color_jitter_prob: float = 0.0, random_resized_crop: bool = True, debug_mode: bool = False)",
                        "brief_description": "Initialize the augmentation pipeline with configurable parameters"
                    },
                    {
                        "name": "augment_image",
                        "signature": "augment_image(self, image: torch.Tensor) -> torch.Tensor",
                        "brief_description": "Apply image augmentations with probability"
                    },
                    {
                        "name": "augment_text",
                        "signature": "augment_text(self, text_data: Union[str, Dict]) -> Union[str, Dict]",
                        "brief_description": "Apply text augmentations with probability"
                    },
                    {
                        "name": "__call__",
                        "signature": "__call__(self, batch: Dict[str, Union[torch.Tensor, Dict]]) -> Dict[str, Union[torch.Tensor, Dict]]",
                        "brief_description": "Apply augmentations to a batch of multimodal data"
                    }
                ],
                "inheritance": "object"
            },
            {
                "name": "TextAugmentation",
                "purpose": "Base class for text augmentation transforms",
                "inheritance": "object"
            },
            {
                "name": "DropWords",
                "purpose": "Randomly drop words from text",
                "inheritance": "TextAugmentation"
            },
            {
                "name": "ShuffleWords",
                "purpose": "Shuffle some words within a window",
                "inheritance": "TextAugmentation"
            },
            {
                "name": "ReplaceWithSynonym",
                "purpose": "Replace words with simple synonyms",
                "inheritance": "TextAugmentation"
            },
            {
                "name": "ChangeWordOrder",
                "purpose": "Change the order of phrases in text",
                "inheritance": "TextAugmentation"
            },
            {
                "name": "AddMisspelling",
                "purpose": "Add simple misspellings to text",
                "inheritance": "TextAugmentation"
            },
            {
                "name": "RandomPosterize",
                "purpose": "Apply posterization to images",
                "inheritance": "nn.Module"
            }
        ],
        "external_dependencies": ["torch", "torchvision", "PIL", "random", "numpy"],
        "complexity_score": 8  # High complexity due to multiple augmentation types and configurations
    }
