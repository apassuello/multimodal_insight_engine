#!/usr/bin/env python3
"""
Metadata Visualization Script

This script takes the JSON metadata compiled by compile_metadata.py
and generates a user-friendly Markdown report with visualizations and summaries.
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configuration
METADATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "project_metadata.json")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "project_architecture.md")

def load_metadata() -> Dict[str, Any]:
    """Load project metadata from JSON file."""
    try:
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Metadata file {METADATA_FILE} not found.")
        print("Please run compile_metadata.py first to generate the metadata.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {METADATA_FILE}.")
        sys.exit(1)

def generate_directory_tree(structure: Dict[str, Any], indent: int = 0) -> str:
    """
    Generate a text-based tree representation of the directory structure.
    
    Args:
        structure: Directory structure dictionary
        indent: Current indentation level
        
    Returns:
        String representation of the tree
    """
    result = []
    prefix = "    " * indent
    
    if structure['type'] == 'directory':
        result.append(f"{prefix}📁 {structure['name']}/")
        for child in structure['children']:
            result.append(generate_directory_tree(child, indent + 1))
    else:
        result.append(f"{prefix}📄 {structure['name']}")
    
    return "\n".join(result)

def generate_module_table(modules: List[Dict[str, Any]]) -> str:
    """
    Generate a markdown table of modules.
    
    Args:
        modules: List of module metadata
        
    Returns:
        Markdown table string
    """
    table = "| Module | Purpose | Complexity |\n"
    table += "|--------|---------|------------|\n"
    
    for module in sorted(modules, key=lambda x: x.get('module_path', '')):
        path = module.get('module_path', 'unknown')
        purpose = module.get('module_purpose', 'No description available')
        complexity = module.get('complexity_score', 'N/A')
        
        # Truncate long purposes
        if len(purpose) > 100:
            purpose = purpose[:97] + "..."
        
        table += f"| `{path}` | {purpose} | {complexity} |\n"
    
    return table

def generate_class_summary(modules: List[Dict[str, Any]]) -> str:
    """
    Generate a summary of key classes in the project.
    
    Args:
        modules: List of module metadata
        
    Returns:
        Markdown summary string
    """
    result = []
    
    for module in modules:
        if 'key_classes' not in module:
            continue
            
        for cls in module.get('key_classes', []):
            module_path = module.get('module_path', 'unknown').replace('src/', '')
            result.append(f"### {cls['name']} (`{module_path}`)")
            result.append(f"{cls.get('purpose', 'No description available')}")
            
            # Add inheritance information if available
            if 'inheritance' in cls and cls['inheritance'] != 'object':
                result.append(f"\n**Inherits from:** `{cls['inheritance']}`")
            
            # Add dependencies
            if 'dependencies' in cls and cls['dependencies']:
                result.append("\n**Dependencies:**")
                for dep in cls['dependencies']:
                    result.append(f"- `{dep}`")
            
            # Add key methods
            if 'key_methods' in cls and cls['key_methods']:
                result.append("\n**Key Methods:**")
                for method in cls['key_methods']:
                    result.append(f"- `{method['name']}`: {method.get('brief_description', 'No description')}")
            
            result.append("\n")
    
    return "\n".join(result)

def generate_complexity_chart(metadata: Dict[str, Any]) -> str:
    """
    Generate a text-based bar chart of module complexity.
    
    Args:
        metadata: Project metadata
        
    Returns:
        Text chart string
    """
    if 'complexity_summary' not in metadata or 'modules_by_complexity' not in metadata['complexity_summary']:
        return "No complexity data available."
    
    complex_modules = metadata['complexity_summary']['modules_by_complexity']
    if not complex_modules:
        return "No complexity data available."
    
    # Take top 10 most complex modules
    top_modules = complex_modules[:10]
    
    result = ["**Top 10 Most Complex Modules:**\n"]
    result.append("```")
    
    # Find max for scaling
    max_score = max(item['score'] for item in top_modules)
    max_name_length = max(len(item['module']) for item in top_modules)
    
    # Generate bars
    for item in top_modules:
        module_name = item['module'].ljust(max_name_length)
        score = item['score']
        bar_length = int((score / max_score) * 30)  # Scale to 30 chars max
        bar = "█" * bar_length
        result.append(f"{module_name} | {bar} ({score})")
    
    result.append("```")
    result.append(f"\n**Average Module Complexity:** {metadata['complexity_summary'].get('average_complexity', 'N/A'):.2f}")
    
    return "\n".join(result)

def generate_dependency_chart(metadata: Dict[str, Any]) -> str:
    """
    Generate a summary of external dependencies.
    
    Args:
        metadata: Project metadata
        
    Returns:
        Markdown summary string
    """
    if 'dependency_summary' not in metadata or 'dependencies_by_usage' not in metadata['dependency_summary']:
        return "No dependency data available."
    
    dependencies = metadata['dependency_summary']['dependencies_by_usage']
    if not dependencies:
        return "No external dependencies found."
    
    result = ["**External Dependencies:**\n"]
    result.append("| Library | Usage Count |\n")
    result.append("|---------|-------------|\n")
    
    for dep in dependencies:
        result.append(f"| {dep['name']} | {dep['usage_count']} |")
    
    return "\n".join(result)

def generate_report(metadata: Dict[str, Any]) -> str:
    """
    Generate a comprehensive markdown report from the metadata.
    
    Args:
        metadata: Project metadata
        
    Returns:
        Markdown report string
    """
    project_name = metadata.get('project_name', 'Project')
    
    report = [
        f"# {project_name} Architecture",
        "## Project Overview",
        "This document provides an overview of the project structure, key components, and architecture.",
        f"Generated from {len(metadata['modules'])} source files.",
        "",
        "## Directory Structure",
        "```",
        generate_directory_tree(metadata['directory_structure']),
        "```",
        "",
        "## Module Summary",
        generate_module_table(metadata['modules']),
        "",
        "## Complexity Analysis",
        generate_complexity_chart(metadata),
        "",
        "## Dependencies",
        generate_dependency_chart(metadata),
        "",
        "## Key Components",
        generate_class_summary(metadata['modules']),
        "",
        "## Statistics",
        f"- Total Python modules: {metadata['summary']['total_modules']}",
        f"- Modules with metadata: {metadata['summary']['modules_with_metadata']}",
        f"- Modules without metadata: {metadata['summary']['modules_without_metadata']}",
        f"- External dependencies: {metadata.get('dependency_summary', {}).get('total_external_dependencies', 'N/A')}",
        "",
        "---",
        "Generated automatically from source code metadata"
    ]
    
    return "\n".join(report)

def main():
    """Main function to generate and save the markdown report."""
    print(f"Loading metadata from {METADATA_FILE}...")
    metadata = load_metadata()
    
    print(f"Generating markdown report...")
    report = generate_report(metadata)
    
    # Save report to markdown file
    with open(OUTPUT_FILE, 'w') as f:
        f.write(report)
    
    print(f"Report generation complete!")
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()