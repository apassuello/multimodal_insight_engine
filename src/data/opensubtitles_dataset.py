import os
import random
from typing import List, Optional, Tuple


class OpenSubtitlesDataset:
    """
    Dataset class for the OpenSubtitles parallel corpus.
    
    This class handles loading and preprocessing parallel text data from the
    OpenSubtitles corpus for machine translation tasks.
    """

    def __init__(
        self,
        data_dir: str = "data/os",
        src_lang: str = "de",
        tgt_lang: str = "en",
        max_examples: Optional[int] = None,
        random_seed: int = 42
    ):
        """
        Initialize the OpenSubtitles dataset.
        
        Args:
            data_dir: Directory containing the OpenSubtitles data
            src_lang: Source language code
            tgt_lang: Target language code
            max_examples: Maximum number of examples to use (None = use all)
            random_seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_examples = max_examples

        # Set random seed for reproducibility
        random.seed(random_seed)

        # Load the data
        self.src_data, self.tgt_data = self.load_data()

    def load_data(self) -> Tuple[List[str], List[str]]:
        """
        Load and preprocess the parallel data.
        
        This function tries multiple possible file structures for OpenSubtitles data.
        
        Returns:
            Tuple containing lists of source and target sentences
        """
        # Try multiple possible file structures
        possible_patterns = [
            # Pattern 1: Direct files in data/os directory
            (f"{self.data_dir}/{self.src_lang}.txt",
             f"{self.data_dir}/{self.tgt_lang}.txt"),

            # Pattern 2: Files in language pair subdirectory (src-tgt order)
            (f"{self.data_dir}/{self.src_lang}-{self.tgt_lang}/{self.src_lang}.txt",
             f"{self.data_dir}/{self.src_lang}-{self.tgt_lang}/{self.tgt_lang}.txt"),

            # Pattern 2b: Files in language pair subdirectory (tgt-src order)
            (f"{self.data_dir}/{self.tgt_lang}-{self.src_lang}/{self.src_lang}.txt",
             f"{self.data_dir}/{self.tgt_lang}-{self.src_lang}/{self.tgt_lang}.txt"),

            # Pattern 3: Files with language pair in name (src-tgt order)
            (f"{self.data_dir}/OpenSubtitles.{self.src_lang}-{self.tgt_lang}.{self.src_lang}",
             f"{self.data_dir}/OpenSubtitles.{self.src_lang}-{self.tgt_lang}.{self.tgt_lang}"),

            # Pattern 3b: Files with language pair in name (tgt-src order)
            (f"{self.data_dir}/OpenSubtitles.{self.tgt_lang}-{self.src_lang}.{self.src_lang}",
             f"{self.data_dir}/OpenSubtitles.{self.tgt_lang}-{self.src_lang}.{self.tgt_lang}"),

            # Pattern 4: Direct language files with OpenSubtitles prefix
            (f"{self.data_dir}/OpenSubtitles.{self.src_lang}",
             f"{self.data_dir}/OpenSubtitles.{self.tgt_lang}"),
        ]

        # Try each pattern until we find files that exist
        src_file, tgt_file = None, None
        for src_pattern, tgt_pattern in possible_patterns:
            if os.path.exists(src_pattern) and os.path.exists(tgt_pattern):
                src_file, tgt_file = src_pattern, tgt_pattern
                print(f"Found OpenSubtitles files using pattern: {src_pattern.split('/')[-1]}")
                break

        # If no pattern matched, use a small synthetic dataset for testing
        if src_file is None or tgt_file is None:
            print(f"Warning: Could not find OpenSubtitles data files for {self.src_lang}-{self.tgt_lang} "
                  f"in directory {self.data_dir}. Using synthetic data for testing.")

            # A small German-English synthetic dataset for testing
            if self.src_lang == "de" and self.tgt_lang == "en":
                return [
                    "Hallo Welt", "Wie geht es dir?", "Danke, mir geht es gut",
                    "Tschüss", "Auf Wiedersehen", "Bis morgen", "Guten Tag",
                    "Ich spreche ein bisschen Deutsch", "Können Sie mir helfen?",
                    "Wo ist der Bahnhof?", "Wie spät ist es?", "Entschuldigung"
                ], [
                    "Hello world", "How are you?", "Thank you, I'm fine",
                    "Goodbye", "Farewell", "See you tomorrow", "Good day",
                    "I speak a little German", "Can you help me?",
                    "Where is the train station?", "What time is it?", "Excuse me"
                ]
            # A small English dataset for testing
            elif self.src_lang == "en" or self.tgt_lang == "en":
                en_data = [
                    "Hello world", "How are you?", "Thank you, I'm fine",
                    "Goodbye", "Farewell", "See you tomorrow", "Good day",
                    "I speak a little English", "Can you help me?",
                    "Where is the train station?", "What time is it?", "Excuse me",
                    "My name is John", "I live in New York", "The weather is nice today",
                    "I would like to order a coffee", "How much does this cost?",
                    "I'll have the steak, please", "The meeting is at 2 PM",
                    "Could you repeat that?", "I don't understand", "Let's go to the movies",
                    "I'm learning a new language", "This is my first time here",
                    "I need to buy a ticket", "Do you accept credit cards?",
                    "Where is the restroom?", "Turn left at the corner",
                    "Can you recommend a good restaurant?", "I'm allergic to nuts",
                    "What's your favorite movie?", "I love this song",
                    "I'll be back in ten minutes", "Happy birthday!",
                    "The book is on the table", "She lives next door",
                    "He works at a hospital", "They arrived yesterday",
                    "We're going to the beach", "I'm sorry I'm late",
                    "That's a beautiful painting", "How was your trip?",
                    "I had a wonderful time", "It's going to rain tomorrow",
                    "This food is delicious", "I'll call you later",
                    "What's the WIFI password?", "Can I have the bill, please?",
                    "I need to catch my flight", "When does the store open?",
                    "I'd like to make a reservation", "Is this seat taken?",
                    "I'm looking for the hotel", "Could you take our picture?",
                    "What do you recommend?", "I'm not feeling well",
                    "I need to see a doctor", "Where can I buy souvenirs?",
                    "What time is the concert?", "How far is it from here?",
                    "I'd like to rent a car", "Can I try this on?",
                ]
                return en_data, en_data
            # Default synthetic data for any other language pair
            else:
                return ["Sample text 1", "Sample text 2"], ["Sample text 1", "Sample text 2"]

        # Read data files
        print(f"Loading source data from: {src_file}")
        with open(src_file, 'r', encoding='utf-8') as f:
            src_data = [line.strip() for line in f if line.strip()]

        print(f"Loading target data from: {tgt_file}")
        with open(tgt_file, 'r', encoding='utf-8') as f:
            tgt_data = [line.strip() for line in f if line.strip()]

        # Ensure same length
        if len(src_data) != len(tgt_data):
            print(f"Warning: Source and target files have different lengths. "
                  f"Source: {len(src_data)}, Target: {len(tgt_data)}")
            min_len = min(len(src_data), len(tgt_data))
            src_data = src_data[:min_len]
            tgt_data = tgt_data[:min_len]

        # Filter out empty lines and lines that are too long or short
        filtered_pairs = []
        for src, tgt in zip(src_data, tgt_data):
            # Skip if either is empty
            if not src or not tgt:
                continue

            # Skip if either is too long (optional, adjust as needed)
            if len(src.split()) > 100 or len(tgt.split()) > 100:
                continue

            filtered_pairs.append((src, tgt))

        # Shuffle and limit
        random.shuffle(filtered_pairs)
        if self.max_examples is not None and self.max_examples < len(filtered_pairs):
            filtered_pairs = filtered_pairs[:self.max_examples]

        # Unzip the pairs
        src_data, tgt_data = zip(*filtered_pairs) if filtered_pairs else ([], [])

        print(f"Loaded {len(src_data)} parallel sentences")

        return list(src_data), list(tgt_data)

def extract_file_metadata(file_path=__file__):
    """
    Extract structured metadata about this module.
    
    Args:
        file_path: Path to the source file (defaults to current file)
        
    Returns:
        dict: Structured metadata about the module's purpose and components
    """
    return {
        "filename": os.path.basename(file_path),
        "module_purpose": "Provides a dataset class for loading and preprocessing OpenSubtitles parallel corpus data for machine translation",
        "key_classes": [
            {
                "name": "OpenSubtitlesDataset",
                "purpose": "Handles loading and preprocessing parallel text data from the OpenSubtitles corpus",
                "key_methods": [
                    {
                        "name": "__init__",
                        "signature": "__init__(self, data_dir: str = 'data/os', src_lang: str = 'de', tgt_lang: str = 'en', max_examples: Optional[int] = None, random_seed: int = 42)",
                        "brief_description": "Initialize the dataset with source/target languages and processing options"
                    },
                    {
                        "name": "load_data",
                        "signature": "load_data(self) -> Tuple[List[str], List[str]]",
                        "brief_description": "Load and preprocess parallel corpora with support for multiple file patterns"
                    }
                ],
                "inheritance": "object",
                "dependencies": ["os", "random", "typing"]
            }
        ],
        "external_dependencies": [],
        "complexity_score": 4  # Moderate complexity for handling different file formats and synthetic data generation
    }
