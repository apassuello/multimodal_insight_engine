class OpenSubtitlesDataset:
    """Mock OpenSubtitlesDataset for testing."""
    def __init__(self, src_lang="de", tgt_lang="en", max_examples=None):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_examples = max_examples
        self.src_data = ["Tschüss", "Auf Wiedersehen"]
        self.tgt_data = ["Goodbye", "Farewell"] 