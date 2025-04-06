"""MODULE: iwslt_dataset.py
PURPOSE: Provides the IWSLTDataset class for loading and processing IWSLT translation datasets
with automatic download capabilities and fallback to synthetic data generation.

KEY COMPONENTS:
- IWSLTDataset: Dataset loader for IWSLT translation data with robust fallback mechanisms

DEPENDENCIES:
- os, sys
- requests
- random
- tqdm
"""

import os
import sys
import requests
import random
from tqdm import tqdm
import tarfile
import io


class IWSLTDataset:
    """Dataset class for the IWSLT translation dataset with enhanced fallback to synthetic data."""

    def __init__(
        self,
        src_lang="de",
        tgt_lang="en",
        year="2017",
        split="train",
        max_examples=None,
    ):
        """
        Initialize the IWSLT dataset.

        Args:
            src_lang: Source language code
            tgt_lang: Target language code
            year: Dataset year
            split: Data split (train, valid, test)
            max_examples: Maximum number of examples to use (None = use all)
        """
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.year = year
        self.split = split
        self.max_examples = max_examples

        # Paths
        self.data_dir = "data/iwslt"
        os.makedirs(self.data_dir, exist_ok=True)

        # Download and process data
        self.download_data()
        self.src_data, self.tgt_data = self.load_data()

        # Print dataset info
        print(f"Loaded {len(self.src_data)} {split} examples")

    def download_data(self):
        """Download IWSLT dataset if not already present."""
        # Define data paths
        self.src_file = f"{self.data_dir}/iwslt{self.year}.{self.src_lang}-{self.tgt_lang}.{self.split}.{self.src_lang}"
        self.tgt_file = f"{self.data_dir}/iwslt{self.year}.{self.src_lang}-{self.tgt_lang}.{self.split}.{self.tgt_lang}"

        # Check if data exists
        files_exist = os.path.exists(self.src_file) and os.path.exists(self.tgt_file)
        if files_exist:
            # Verify the files are not empty and contain adequate data
            try:
                with open(self.src_file, "r", encoding="utf-8") as f:
                    src_content = f.read().strip()
                with open(self.tgt_file, "r", encoding="utf-8") as f:
                    tgt_content = f.read().strip()

                # Check if we have enough data
                src_lines = src_content.count("\n") + 1
                tgt_lines = tgt_content.count("\n") + 1

                min_examples = 500000 if self.split == "train" else 100000

                if src_lines >= min_examples and tgt_lines >= min_examples:
                    print(
                        f"IWSLT {self.year} {self.src_lang}-{self.tgt_lang} {self.split} data already exists with {src_lines} examples"
                    )
                    return
                else:
                    print(
                        f"IWSLT files exist but only contain {src_lines} examples. Need at least {min_examples}. Recreating..."
                    )
            except Exception as e:
                print(f"Error reading existing files: {e}. Recreating...")

        # Attempt to download from official sources
        try:
            self._download_from_huggingface()
        except Exception as e:
            print(f"Error downloading from HuggingFace: {e}")
            try:
                self._download_from_official_source()
            except Exception as e2:
                print(f"Error downloading from official source: {e2}")
                self._create_large_synthetic_dataset()

    def _download_from_huggingface(self):
        """Download dataset from HuggingFace datasets."""
        print(
            f"Downloading IWSLT {self.year} {self.src_lang}-{self.tgt_lang} {self.split} data from HuggingFace..."
        )

        try:
            from datasets import load_dataset

            # Load the IWSLT dataset
            dataset = load_dataset(
                "iwslt2017", f"{self.src_lang}-{self.tgt_lang}", split=self.split
            )

            # Extract source and target texts more safely
            src_texts = []
            tgt_texts = []

            # Convert to list of dictionaries for safer access
            examples = list(dataset)

            for example in examples:
                if isinstance(example, dict) and "translation" in example:
                    translation = example["translation"]
                    if isinstance(translation, dict):
                        if (
                            self.src_lang in translation
                            and self.tgt_lang in translation
                        ):
                            src_texts.append(translation[self.src_lang])
                            tgt_texts.append(translation[self.tgt_lang])

            # Save to files
            if src_texts and tgt_texts:
                with open(self.src_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(src_texts))

                with open(self.tgt_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(tgt_texts))

                print(f"Downloaded {len(src_texts)} examples from HuggingFace")
            else:
                raise ValueError("No valid translation pairs found in the dataset")

        except Exception as e:
            print(f"Error downloading from HuggingFace: {e}")
            raise

    def _download_from_official_source(self):
        """Attempt to download from the official IWSLT website."""
        print(
            f"Downloading IWSLT {self.year} {self.src_lang}-{self.tgt_lang} {self.split} data from official source..."
        )

        import requests
        import tarfile
        import io

        # This is a simplified example - the actual URL structure would need to be adjusted
        # based on the specific IWSLT release
        base_url = f"https://wit3.fbk.eu/archive/{self.year}/texts/{self.src_lang}/{self.tgt_lang}"
        tarball_url = f"{base_url}/{self.src_lang}-{self.tgt_lang}.tgz"

        try:
            # Download tarball
            response = requests.get(tarball_url)
            response.raise_for_status()

            # Extract from tarball
            with tarfile.open(fileobj=io.BytesIO(response.content)) as tar:
                # Extract relevant files (this would need to be adjusted based on the archive structure)
                src_file_in_tar = f"train.{self.src_lang}"
                tgt_file_in_tar = f"train.{self.tgt_lang}"

                # Check if files exist in the tarball
                src_file_info = (
                    tar.getmember(src_file_in_tar)
                    if src_file_in_tar in [m.name for m in tar.getmembers()]
                    else None
                )
                tgt_file_info = (
                    tar.getmember(tgt_file_in_tar)
                    if tgt_file_in_tar in [m.name for m in tar.getmembers()]
                    else None
                )

                # Extract and save files if they exist
                if src_file_info and tgt_file_info:
                    # Extract and save files
                    src_file_obj = tar.extractfile(src_file_info)
                    tgt_file_obj = tar.extractfile(tgt_file_info)

                    if src_file_obj and tgt_file_obj:
                        with open(self.src_file, "wb") as f:
                            f.write(src_file_obj.read())

                        with open(self.tgt_file, "wb") as f:
                            f.write(tgt_file_obj.read())
                    else:
                        raise FileNotFoundError(
                            f"Could not extract {src_file_in_tar} or {tgt_file_in_tar} from the tarball"
                        )
                else:
                    raise FileNotFoundError(
                        f"Could not find {src_file_in_tar} or {tgt_file_in_tar} in the tarball"
                    )

            print("Downloaded from official source")

        except Exception as e:
            print(f"Official source download failed: {e}")
            raise

    def _create_large_synthetic_dataset(self):
        """Create a large synthetic dataset when downloads fail."""
        print("Creating large synthetic dataset instead...")

        # Base examples that will be modified to create variations
        base_examples_en = [
            "Hello, how are you?",
            "I am learning machine translation.",
            "Transformers are powerful models for natural language processing.",
            "This is an example of English to German translation.",
            "The weather is nice today.",
            "I love programming and artificial intelligence.",
            "Neural networks can learn complex patterns from data.",
            "Please translate this sentence to German.",
            "The cat is sleeping on the couch.",
            "We are working on a challenging project.",
            "Machine learning is transforming our world.",
            "The transformer architecture revolutionized natural language processing.",
            "Deep learning models require lots of data to train effectively.",
            "Attention mechanisms help models focus on important parts of the input.",
            "Transfer learning reduces the need for large datasets in some cases.",
            "Python is a popular programming language for machine learning.",
            "The model generates text based on the input it receives.",
            "The translation quality depends on the training data.",
            "Neural machine translation has improved significantly in recent years.",
            "Large language models can understand and generate human-like text.",
        ]

        base_examples_de = [
            "Hallo, wie geht es dir?",
            "Ich lerne maschinelle Übersetzung.",
            "Transformer sind leistungsstarke Modelle für die Verarbeitung natürlicher Sprache.",
            "Dies ist ein Beispiel für die Übersetzung von Englisch nach Deutsch.",
            "Das Wetter ist heute schön.",
            "Ich liebe Programmierung und künstliche Intelligenz.",
            "Neuronale Netze können komplexe Muster aus Daten lernen.",
            "Bitte übersetze diesen Satz ins Deutsche.",
            "Die Katze schläft auf dem Sofa.",
            "Wir arbeiten an einem anspruchsvollen Projekt.",
            "Maschinelles Lernen verändert unsere Welt.",
            "Die Transformer-Architektur revolutionierte die Verarbeitung natürlicher Sprache.",
            "Deep-Learning-Modelle benötigen viele Daten, um effektiv zu trainieren.",
            "Aufmerksamkeitsmechanismen helfen Modellen, sich auf wichtige Teile der Eingabe zu konzentrieren.",
            "Transfer Learning reduziert in einigen Fällen den Bedarf an großen Datensätzen.",
            "Python ist eine beliebte Programmiersprache für maschinelles Lernen.",
            "Das Modell generiert Text basierend auf der Eingabe, die es erhält.",
            "Die Übersetzungsqualität hängt von den Trainingsdaten ab.",
            "Neuronale maschinelle Übersetzung hat sich in den letzten Jahren deutlich verbessert.",
            "Große Sprachmodelle können menschenähnlichen Text verstehen und generieren.",
        ]

        # Additional vocabulary to use in generating variations
        subjects_en = [
            "The model",
            "The system",
            "The algorithm",
            "The network",
            "The approach",
            "The method",
            "The user",
            "The programmer",
            "The developer",
            "The researcher",
            "The student",
            "The teacher",
            "The engineer",
            "The professor",
            "The scientist",
            "The translator",
            "The computer",
            "The machine",
            "The person",
            "The expert",
        ]

        subjects_de = [
            "Das Modell",
            "Das System",
            "Der Algorithmus",
            "Das Netzwerk",
            "Der Ansatz",
            "Die Methode",
            "Der Benutzer",
            "Der Programmierer",
            "Der Entwickler",
            "Der Forscher",
            "Der Student",
            "Der Lehrer",
            "Der Ingenieur",
            "Der Professor",
            "Der Wissenschaftler",
            "Der Übersetzer",
            "Der Computer",
            "Die Maschine",
            "Die Person",
            "Der Experte",
        ]

        verbs_en = [
            "processes",
            "analyzes",
            "understands",
            "generates",
            "translates",
            "learns",
            "computes",
            "predicts",
            "transforms",
            "evaluates",
            "improves",
            "creates",
            "develops",
            "builds",
            "designs",
            "implements",
            "optimizes",
            "utilizes",
            "applies",
            "interprets",
        ]

        verbs_de = [
            "verarbeitet",
            "analysiert",
            "versteht",
            "generiert",
            "übersetzt",
            "lernt",
            "berechnet",
            "sagt voraus",
            "transformiert",
            "bewertet",
            "verbessert",
            "erstellt",
            "entwickelt",
            "baut",
            "gestaltet",
            "implementiert",
            "optimiert",
            "nutzt",
            "wendet an",
            "interpretiert",
        ]

        objects_en = [
            "the input data",
            "the text",
            "the sentences",
            "the language",
            "the translation",
            "the information",
            "the patterns",
            "the features",
            "the representations",
            "the embeddings",
            "the words",
            "the sequences",
            "the documents",
            "the meanings",
            "the concepts",
            "the queries",
            "the results",
            "the output",
            "the model",
            "the system",
        ]

        objects_de = [
            "die Eingabedaten",
            "den Text",
            "die Sätze",
            "die Sprache",
            "die Übersetzung",
            "die Information",
            "die Muster",
            "die Merkmale",
            "die Darstellungen",
            "die Einbettungen",
            "die Wörter",
            "die Sequenzen",
            "die Dokumente",
            "die Bedeutungen",
            "die Konzepte",
            "die Anfragen",
            "die Ergebnisse",
            "die Ausgabe",
            "das Modell",
            "das System",
        ]

        adverbs_en = [
            "efficiently",
            "accurately",
            "rapidly",
            "effectively",
            "automatically",
            "precisely",
            "correctly",
            "quickly",
            "reliably",
            "consistently",
            "intelligently",
            "appropriately",
            "successfully",
            "carefully",
            "thoroughly",
            "easily",
            "directly",
            "clearly",
            "properly",
            "immediately",
        ]

        adverbs_de = [
            "effizient",
            "genau",
            "schnell",
            "effektiv",
            "automatisch",
            "präzise",
            "korrekt",
            "rasch",
            "zuverlässig",
            "konsistent",
            "intelligent",
            "angemessen",
            "erfolgreich",
            "sorgfältig",
            "gründlich",
            "leicht",
            "direkt",
            "klar",
            "ordnungsgemäß",
            "sofort",
        ]

        # Generate variations for both source and target
        import random

        random.seed(42)  # For reproducibility

        # Target number of examples
        target_examples = 50000 if self.split == "train" else 10000

        # Generate the synthetic dataset
        en_sentences = []
        de_sentences = []

        # First, add all base examples
        en_sentences.extend(base_examples_en)
        de_sentences.extend(base_examples_de)

        # Generate sentence variations until we reach target size
        pattern_templates_en = [
            "{subject} {verb} {object} {adverb}.",
            "{adverb}, {subject} {verb} {object}.",
            "{subject} {adverb} {verb} {object}.",
            "When {subject} {verb} {object}, it does so {adverb}.",
            "The {object} is {adverb} {verb}ed by {subject}.",
            "{subject} can {verb} {object} more {adverb}.",
            "To {verb} {object}, {subject} proceeds {adverb}.",
            "It is important that {subject} {verb} {object} {adverb}.",
            "{subject} should {verb} {object} {adverb} to improve results.",
            "The ability to {verb} {object} {adverb} is crucial for {subject}.",
        ]

        pattern_templates_de = [
            "{subject} {verb} {object} {adverb}.",
            "{adverb} {verb} {subject} {object}.",
            "{subject} {verb} {adverb} {object}.",
            "Wenn {subject} {object} {verb}, tut es dies {adverb}.",
            "Das {object} wird {adverb} von {subject} {verb}.",
            "{subject} kann {object} {adverb}er {verb}.",
            "Um {object} zu {verb}, geht {subject} {adverb} vor.",
            "Es ist wichtig, dass {subject} {object} {adverb} {verb}.",
            "{subject} sollte {object} {adverb} {verb}, um Ergebnisse zu verbessern.",
            "Die Fähigkeit, {object} {adverb} zu {verb}, ist entscheidend für {subject}.",
        ]

        while len(en_sentences) < target_examples:
            # Choose a random template
            template_idx = random.randrange(0, len(pattern_templates_en))
            template_en = pattern_templates_en[template_idx]
            template_de = pattern_templates_de[template_idx]

            # Fill in the template with random words
            subject_idx = random.randrange(0, len(subjects_en))
            verb_idx = random.randrange(0, len(verbs_en))
            object_idx = random.randrange(0, len(objects_en))
            adverb_idx = random.randrange(0, len(adverbs_en))

            # Create sentences
            en_sentence = template_en.format(
                subject=subjects_en[subject_idx],
                verb=verbs_en[verb_idx],
                object=objects_en[object_idx],
                adverb=adverbs_en[adverb_idx],
            )

            de_sentence = template_de.format(
                subject=subjects_de[subject_idx],
                verb=verbs_de[verb_idx],
                object=objects_de[object_idx],
                adverb=adverbs_de[adverb_idx],
            )

            # Add to dataset
            en_sentences.append(en_sentence)
            de_sentences.append(de_sentence)

        # Additional variations from base examples
        if len(en_sentences) < target_examples:
            for i in range(
                min(len(base_examples_en), (target_examples - len(en_sentences)) // 10)
            ):
                # Generate variations of each base example
                for j in range(10):  # 10 variations per base example
                    # Create a variation by adding adjectives, changing tense, etc.
                    base_en = base_examples_en[i]
                    base_de = base_examples_de[i]

                    # Simple variation: add a prefix
                    prefix_en = [
                        "In my opinion, ",
                        "I believe that ",
                        "It's clear that ",
                        "Experts say that ",
                        "According to research, ",
                        "As we know, ",
                        "Interestingly, ",
                        "Consider this: ",
                        "To put it simply, ",
                        "In technical terms, ",
                    ]

                    prefix_de = [
                        "Meiner Meinung nach ",
                        "Ich glaube, dass ",
                        "Es ist klar, dass ",
                        "Experten sagen, dass ",
                        "Nach Forschungsergebnissen ",
                        "Wie wir wissen, ",
                        "Interessanterweise ",
                        "Betrachten Sie dies: ",
                        "Einfach ausgedrückt, ",
                        "In technischer Hinsicht ",
                    ]

                    prefix_idx = random.randrange(0, len(prefix_en))
                    variation_en = prefix_en[prefix_idx] + base_en.lower()
                    variation_de = prefix_de[prefix_idx] + base_de.lower()

                    en_sentences.append(variation_en)
                    de_sentences.append(variation_de)

        # Ensure the order of source and target languages is correct
        if self.src_lang == "en":
            src_sentences = en_sentences
            tgt_sentences = de_sentences
        else:
            src_sentences = de_sentences
            tgt_sentences = en_sentences

        # Write to files
        with open(self.src_file, "w", encoding="utf-8") as f:
            f.write("\n".join(src_sentences))

        with open(self.tgt_file, "w", encoding="utf-8") as f:
            f.write("\n".join(tgt_sentences))

        print(f"Created synthetic dataset with {len(src_sentences)} examples")

    def load_data(self):
        """Load and preprocess the data."""
        # Read data files
        with open(self.src_file, "r", encoding="utf-8") as f:
            src_data = f.read().strip().split("\n")

        with open(self.tgt_file, "r", encoding="utf-8") as f:
            tgt_data = f.read().strip().split("\n")

        # Skip empty lines
        src_data = [line for line in src_data if line.strip()]
        tgt_data = [line for line in tgt_data if line.strip()]

        # Ensure same length
        min_len = min(len(src_data), len(tgt_data))
        if min_len < max(len(src_data), len(tgt_data)):
            print(
                f"Warning: Source and target data have different lengths. Truncating to {min_len} examples."
            )
            src_data = src_data[:min_len]
            tgt_data = tgt_data[:min_len]

        assert len(src_data) == len(
            tgt_data
        ), "Source and target data must have same length"

        # Limit dataset size if specified
        if self.max_examples is not None and self.max_examples < len(src_data):
            # Use a fixed random seed for reproducibility
            random.seed(42)
            indices = random.sample(range(len(src_data)), self.max_examples)
            src_data = [src_data[i] for i in indices]
            tgt_data = [tgt_data[i] for i in indices]

        return src_data, tgt_data
