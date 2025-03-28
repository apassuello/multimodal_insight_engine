import pytest
from unittest.mock import Mock, patch
from src.data.combined_translation_dataset import CombinedTranslationDataset

# Mock the OpenSubtitlesDataset import
with patch('src.data.combined_translation_dataset.OpenSubtitlesDataset', Mock):
    from src.data.combined_translation_dataset import CombinedTranslationDataset

@pytest.fixture
def mock_europarl_dataset():
    """Mock EuroparlDataset for testing."""
    mock = Mock()
    mock.src_data = ["Hallo", "Guten Tag", "Wie geht es dir?"]
    mock.tgt_data = ["Hello", "Good day", "How are you?"]
    return mock

@pytest.fixture
def mock_opensubtitles_dataset():
    """Mock OpenSubtitlesDataset for testing."""
    mock = Mock()
    mock.src_data = ["Tschüss", "Auf Wiedersehen"]
    mock.tgt_data = ["Goodbye", "Farewell"]
    return mock

def test_combined_dataset_initialization(mock_europarl_dataset, mock_opensubtitles_dataset):
    """Test basic initialization of CombinedTranslationDataset."""
    with patch('src.data.combined_translation_dataset.EuroparlDataset', return_value=mock_europarl_dataset), \
         patch('src.data.combined_translation_dataset.OpenSubtitlesDataset', return_value=mock_opensubtitles_dataset):
        
        # Test with default parameters
        dataset = CombinedTranslationDataset()
        assert dataset.src_lang == "de"
        assert dataset.tgt_lang == "en"
        assert len(dataset.src_data) == 5  # 3 from europarl + 2 from opensubtitles
        assert len(dataset.tgt_data) == 5
        
        # Test with custom parameters
        dataset = CombinedTranslationDataset(
            src_lang="fr",
            tgt_lang="es",
            datasets={"europarl": 2, "opensubtitles": 1}
        )
        assert dataset.src_lang == "fr"
        assert dataset.tgt_lang == "es"
        assert len(dataset.src_data) == 3  # 2 from europarl + 1 from opensubtitles
        assert len(dataset.tgt_data) == 3

def test_combined_dataset_custom_datasets(mock_europarl_dataset):
    """Test initialization with only one dataset."""
    with patch('src.data.combined_translation_dataset.EuroparlDataset', return_value=mock_europarl_dataset):
        dataset = CombinedTranslationDataset(
            datasets={"europarl": 3}
        )
        assert len(dataset.src_data) == 3
        assert len(dataset.tgt_data) == 3
        assert all(src in mock_europarl_dataset.src_data for src in dataset.src_data)
        assert all(tgt in mock_europarl_dataset.tgt_data for tgt in dataset.tgt_data)

def test_combined_dataset_invalid_dataset():
    """Test initialization with invalid dataset name."""
    with pytest.raises(ValueError, match="Unknown dataset: invalid_dataset"):
        CombinedTranslationDataset(
            datasets={"invalid_dataset": 100}
        )

def test_combined_dataset_empty_datasets():
    """Test initialization with empty datasets dictionary."""
    dataset = CombinedTranslationDataset(datasets={})
    assert len(dataset.src_data) == 0
    assert len(dataset.tgt_data) == 0 