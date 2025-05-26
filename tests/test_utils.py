"""
Unit tests for utility functions.
"""
import pytest
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, mock_open
import json

from pparser.utils.helpers import (
    ensure_dir, clean_text, safe_filename, chunk_text,
    merge_bboxes, calculate_overlap, validate_bbox,
    save_json, load_json, get_file_hash, format_file_size
)


class TestDirectoryHelpers:
    """Test directory utility functions."""
    
    def test_ensure_dir_creates_directory(self, temp_dir):
        """Test that ensure_dir creates a directory."""
        test_path = temp_dir / "new_dir"
        assert not test_path.exists()
        
        ensure_dir(test_path)
        assert test_path.exists()
        assert test_path.is_dir()
    
    def test_ensure_dir_with_existing_directory(self, temp_dir):
        """Test that ensure_dir works with existing directory."""
        test_path = temp_dir / "existing_dir"
        test_path.mkdir()
        
        # Should not raise an error
        ensure_dir(test_path)
        assert test_path.exists()
    
    def test_ensure_dir_creates_parent_directories(self, temp_dir):
        """Test that ensure_dir creates parent directories."""
        test_path = temp_dir / "parent" / "child" / "grandchild"
        
        ensure_dir(test_path)
        assert test_path.exists()
        assert test_path.is_dir()


class TestTextHelpers:
    """Test text processing utility functions."""
    
    def test_clean_text_removes_extra_whitespace(self):
        """Test that clean_text removes extra whitespace."""
        text = "  Hello    world  \n\n  Test  "
        result = clean_text(text)
        assert result == "Hello world Test"
    
    def test_clean_text_preserves_single_spaces(self):
        """Test that clean_text preserves single spaces."""
        text = "Hello world test"
        result = clean_text(text)
        assert result == "Hello world test"
    
    def test_clean_text_handles_empty_string(self):
        """Test that clean_text handles empty strings."""
        assert clean_text("") == ""
        assert clean_text("   ") == ""
    
    def test_clean_text_handles_none(self):
        """Test that clean_text handles None input."""
        assert clean_text(None) == ""
    
    def test_safe_filename_removes_invalid_characters(self):
        """Test that safe_filename removes invalid characters."""
        filename = "test<>:|?*file.txt"
        result = safe_filename(filename)
        assert result == "test_file.txt"
    
    def test_safe_filename_handles_spaces(self):
        """Test that safe_filename handles spaces."""
        filename = "test file name.txt"
        result = safe_filename(filename)
        assert result == "test_file_name.txt"
    
    def test_safe_filename_preserves_valid_characters(self):
        """Test that safe_filename preserves valid characters."""
        filename = "test-file_123.txt"
        result = safe_filename(filename)
        assert result == "test-file_123.txt"
    
    def test_chunk_text_splits_correctly(self):
        """Test that chunk_text splits text correctly."""
        text = "This is a test. " * 100  # Long text
        chunks = chunk_text(text, chunk_size=50)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 50
    
    def test_chunk_text_preserves_content(self):
        """Test that chunk_text preserves all content."""
        text = "This is a test sentence with multiple words."
        chunks = chunk_text(text, chunk_size=20, overlap=5)
        
        # Reconstruct text and check content is preserved
        reconstructed = chunks[0]
        for i in range(1, len(chunks)):
            # Remove overlap
            overlap_text = chunks[i][:5] if len(chunks[i]) > 5 else chunks[i]
            if overlap_text in reconstructed[-10:]:
                # Find the overlap and add the new part
                new_part = chunks[i][len(overlap_text):]
                reconstructed += new_part
            else:
                reconstructed += chunks[i]
        
        # Check that important content is preserved
        assert "test sentence" in reconstructed
        assert "multiple words" in reconstructed


class TestBoundingBoxHelpers:
    """Test bounding box utility functions."""
    
    def test_validate_bbox_valid_input(self):
        """Test validate_bbox with valid input."""
        bbox = [10, 20, 50, 80]
        result = validate_bbox(bbox)
        assert result == [10, 20, 50, 80]
    
    def test_validate_bbox_corrects_order(self):
        """Test validate_bbox corrects coordinate order."""
        bbox = [50, 80, 10, 20]  # x1 > x0, y1 > y0
        result = validate_bbox(bbox)
        assert result == [10, 20, 50, 80]
    
    def test_validate_bbox_invalid_input(self):
        """Test validate_bbox with invalid input."""
        assert validate_bbox([1, 2, 3]) is None  # Wrong length
        assert validate_bbox(None) is None
        assert validate_bbox("invalid") is None
    
    def test_merge_bboxes_single_box(self):
        """Test merge_bboxes with single bounding box."""
        bbox = [10, 20, 50, 80]
        result = merge_bboxes([bbox])
        assert result == bbox
    
    def test_merge_bboxes_multiple_boxes(self):
        """Test merge_bboxes with multiple bounding boxes."""
        bboxes = [
            [10, 20, 30, 40],
            [25, 35, 60, 70],
            [5, 15, 20, 50]
        ]
        result = merge_bboxes(bboxes)
        assert result == [5, 15, 60, 70]  # Min x0, y0, max x1, y1
    
    def test_merge_bboxes_empty_list(self):
        """Test merge_bboxes with empty list."""
        assert merge_bboxes([]) is None
    
    def test_calculate_overlap_no_overlap(self):
        """Test calculate_overlap with non-overlapping boxes."""
        bbox1 = [0, 0, 10, 10]
        bbox2 = [20, 20, 30, 30]
        assert calculate_overlap(bbox1, bbox2) == 0.0
    
    def test_calculate_overlap_partial_overlap(self):
        """Test calculate_overlap with partial overlap."""
        bbox1 = [0, 0, 20, 20]
        bbox2 = [10, 10, 30, 30]
        
        # Overlap area is 10x10 = 100
        # Union area is 20x20 + 20x20 - 100 = 700
        expected = 100 / 700
        result = calculate_overlap(bbox1, bbox2)
        assert abs(result - expected) < 0.001
    
    def test_calculate_overlap_complete_overlap(self):
        """Test calculate_overlap with complete overlap."""
        bbox1 = [10, 10, 20, 20]
        bbox2 = [10, 10, 20, 20]
        assert calculate_overlap(bbox1, bbox2) == 1.0


class TestFileHelpers:
    """Test file utility functions."""
    
    def test_save_and_load_json(self, temp_dir):
        """Test saving and loading JSON files."""
        data = {"test": "data", "number": 42, "list": [1, 2, 3]}
        file_path = temp_dir / "test.json"
        
        # Save data
        save_json(data, file_path)
        assert file_path.exists()
        
        # Load data
        loaded_data = load_json(file_path)
        assert loaded_data == data
    
    def test_load_json_nonexistent_file(self, temp_dir):
        """Test loading nonexistent JSON file."""
        file_path = temp_dir / "nonexistent.json"
        assert load_json(file_path) is None
    
    def test_load_json_invalid_json(self, temp_dir):
        """Test loading invalid JSON file."""
        file_path = temp_dir / "invalid.json"
        file_path.write_text("invalid json content")
        
        assert load_json(file_path) is None
    
    def test_get_file_hash(self, temp_dir):
        """Test getting file hash."""
        file_path = temp_dir / "test.txt"
        content = "test content"
        file_path.write_text(content)
        
        hash1 = get_file_hash(file_path)
        hash2 = get_file_hash(file_path)
        
        # Same file should have same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hash length
    
    def test_get_file_hash_different_files(self, temp_dir):
        """Test that different files have different hashes."""
        file1 = temp_dir / "test1.txt"
        file2 = temp_dir / "test2.txt"
        
        file1.write_text("content 1")
        file2.write_text("content 2")
        
        hash1 = get_file_hash(file1)
        hash2 = get_file_hash(file2)
        
        assert hash1 != hash2
    
    def test_get_file_hash_nonexistent_file(self, temp_dir):
        """Test getting hash of nonexistent file."""
        file_path = temp_dir / "nonexistent.txt"
        assert get_file_hash(file_path) is None
    
    def test_format_file_size(self):
        """Test file size formatting."""
        assert format_file_size(512) == "512 B"
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1536) == "1.5 KB"
        assert format_file_size(1024 * 1024) == "1.0 MB"
        assert format_file_size(1024 * 1024 * 1024) == "1.0 GB"
    
    def test_format_file_size_zero(self):
        """Test formatting zero file size."""
        assert format_file_size(0) == "0 B"
    
    def test_format_file_size_negative(self):
        """Test formatting negative file size."""
        assert format_file_size(-100) == "0 B"
