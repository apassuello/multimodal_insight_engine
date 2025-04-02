#!/usr/bin/env python3
"""
Metadata Compilation Script

This script recursively searches through the project's source code directory,
finds all Python files with an `extract_file_metadata` function, calls each function,
and compiles the metadata into a comprehensive JSON document that represents
the entire project structure and functionality.
"""

import os
import sys
import json
import importlib.util
import inspect
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import importlib
import logging
import types

# Configuration
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "project_metadata.json")

# Add parent directory to path to allow imports to work
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Create a NullHandler for logging
if not hasattr(logging, 'NullHandler'):
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass
    logging.NullHandler = NullHandler

def is_python_file(file_path: str) -> bool:
    """Check if a file is a Python source file."""
    return file_path.endswith('.py') and not file_path.endswith('__init__.py')

def extract_metadata_from_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract metadata from a Python file if it has an extract_file_metadata function.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Metadata dictionary or None if no metadata function exists
    """
    try:
        # Read the module source code
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Find the extract_file_metadata function
        if 'def extract_file_metadata(' not in source_code:
            return None
        
        # Extract the metadata function content
        metadata_func_code = ""
        in_func = False
        indentation = 0
        
        for line in source_code.split('\n'):
            if 'def extract_file_metadata(' in line:
                in_func = True
                indentation = len(line) - len(line.lstrip())
                metadata_func_code += line + '\n'
            elif in_func:
                if line.strip() and not line.startswith(' ' * indentation):
                    # End of function reached
                    in_func = False
                else:
                    metadata_func_code += line + '\n'
        
        # Create the extract_file_metadata function in our namespace
        namespace = {
            '__file__': file_path,
            'os': os,
        }
        
        # Add common imports that might be needed
        try:
            import torch
            namespace['torch'] = torch
        except ImportError:
            pass
            
        try:
            import numpy as np
            namespace['np'] = np
            namespace['numpy'] = np
        except ImportError:
            pass
            
        # Execute the function definition in the namespace
        exec(metadata_func_code, namespace)
        
        # Check if the function was successfully defined
        if 'extract_file_metadata' in namespace:
            # Call the function with the file path
            metadata = namespace['extract_file_metadata'](file_path)
            
            # Add module path to the metadata
            rel_path = os.path.relpath(file_path, os.path.dirname(SRC_DIR))
            metadata['module_path'] = rel_path
            
            return metadata
        
        return None
    except Exception as e:
        print(f"Error extracting metadata from {file_path}: {e}")
        return None

def collect_directory_structure(root_dir: str) -> Dict[str, Any]:
    """
    Build a dictionary representing the directory structure.
    
    Args:
        root_dir: Root directory to start from
        
    Returns:
        Dictionary with the directory structure
    """
    structure = {
        'name': os.path.basename(root_dir),
        'type': 'directory',
        'children': []
    }
    
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        
        # Skip __pycache__ and hidden directories
        if item.startswith('__') or item.startswith('.'):
            continue
            
        if os.path.isdir(item_path):
            # Recursively process subdirectories
            subdir_structure = collect_directory_structure(item_path)
            structure['children'].append(subdir_structure)
        elif is_python_file(item_path):
            # Add file information
            structure['children'].append({
                'name': item,
                'type': 'file',
                'path': os.path.relpath(item_path, os.path.dirname(SRC_DIR))
            })
    
    return structure

def compile_project_metadata() -> Dict[str, Any]:
    """
    Compile metadata for the entire project.
    
    Returns:
        Dictionary with project metadata
    """
    # Get all Python files in the project
    python_files = []
    for root, _, files in os.walk(SRC_DIR):
        for file in files:
            if is_python_file(os.path.join(root, file)):
                python_files.append(os.path.join(root, file))
    
    # Extract metadata from each file
    all_metadata = []
    modules_with_metadata = 0
    modules_without_metadata = 0
    
    for file_path in python_files:
        metadata = extract_metadata_from_file(file_path)
        if metadata is not None:
            all_metadata.append(metadata)
            modules_with_metadata += 1
        else:
            modules_without_metadata += 1
            # Create a minimal metadata entry for files without the function
            rel_path = os.path.relpath(file_path, os.path.dirname(SRC_DIR))
            all_metadata.append({
                'filename': os.path.basename(file_path),
                'module_path': rel_path,
                'module_purpose': 'No metadata function available',
                'has_metadata_function': False
            })
    
    # Get directory structure
    directory_structure = collect_directory_structure(SRC_DIR)
    
    # Build complete project metadata
    project_metadata = {
        'project_name': 'MultiModal Insight Engine',
        'directory_structure': directory_structure,
        'modules': all_metadata,
        'summary': {
            'total_modules': len(python_files),
            'modules_with_metadata': modules_with_metadata,
            'modules_without_metadata': modules_without_metadata,
        }
    }
    
    return project_metadata

def generate_complexity_summary(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a summary of module complexity scores.
    
    Args:
        metadata: The complete project metadata
        
    Returns:
        Dictionary with complexity summary
    """
    complexity_scores = []
    
    for module in metadata['modules']:
        if 'complexity_score' in module:
            complexity_scores.append({
                'module': module.get('module_path', module.get('filename', 'unknown')),
                'score': module['complexity_score']
            })
    
    # Sort by complexity score (descending)
    complexity_scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Calculate average complexity
    avg_complexity = sum(item['score'] for item in complexity_scores) / len(complexity_scores) if complexity_scores else 0
    
    return {
        'average_complexity': avg_complexity,
        'modules_by_complexity': complexity_scores
    }

def generate_dependency_summary(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a summary of module dependencies.
    
    Args:
        metadata: The complete project metadata
        
    Returns:
        Dictionary with dependency summary
    """
    all_dependencies = set()
    dependency_count = {}
    
    for module in metadata['modules']:
        if 'external_dependencies' in module:
            module_deps = module['external_dependencies']
            for dep in module_deps:
                all_dependencies.add(dep)
                dependency_count[dep] = dependency_count.get(dep, 0) + 1
    
    # Sort dependencies by usage count
    sorted_dependencies = [
        {'name': dep, 'usage_count': count}
        for dep, count in sorted(dependency_count.items(), key=lambda x: x[1], reverse=True)
    ]
    
    return {
        'total_external_dependencies': len(all_dependencies),
        'dependencies_by_usage': sorted_dependencies
    }

def main():
    """Main function to compile and save project metadata."""
    print(f"Compiling metadata from source files in {SRC_DIR}...")
    project_metadata = compile_project_metadata()
    
    # Add complexity summary
    project_metadata['complexity_summary'] = generate_complexity_summary(project_metadata)
    
    # Add dependency summary
    project_metadata['dependency_summary'] = generate_dependency_summary(project_metadata)
    
    # Save metadata to JSON file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(project_metadata, f, indent=2)
    
    print(f"Metadata compilation complete!")
    print(f"Total modules: {project_metadata['summary']['total_modules']}")
    print(f"Modules with metadata: {project_metadata['summary']['modules_with_metadata']}")
    print(f"Modules without metadata: {project_metadata['summary']['modules_without_metadata']}")
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()