import ast
import os
import re
from typing import Dict, List, Any, Optional, Tuple

def extract_docstring_info(docstring: str) -> Dict[str, Any]:
    """Extract structured information from a module-level docstring."""
    if not docstring:
        return {}
    
    info = {
        "module_purpose": "",
        "key_components": [],
        "dependencies": [],
        "special_notes": "",
    }
    
    # Look for purpose in docstring
    purpose_match = re.search(r'PURPOSE:\s*(.*?)(?:\n|$)', docstring)
    if purpose_match:
        info["module_purpose"] = purpose_match.group(1).strip()
    
    # Look for key components
    components_match = re.search(r'KEY COMPONENTS:(.*?)(?:DEPENDENCIES|SPECIAL NOTES|$)', 
                               docstring, re.DOTALL)
    if components_match:
        components_text = components_match.group(1).strip()
        for line in components_text.split('\n'):
            component = line.strip()
            if component.startswith('-'):
                info["key_components"].append(component[1:].strip())
    
    # Look for dependencies
    dependencies_match = re.search(r'DEPENDENCIES:\s*(.*?)(?:SPECIAL NOTES|$)', 
                                 docstring, re.DOTALL)
    if dependencies_match:
        dependencies_text = dependencies_match.group(1).strip()
        info["dependencies"] = [dep.strip() for dep in re.split(r',|\s+', dependencies_text) if dep.strip()]
    
    # Look for special notes
    notes_match = re.search(r'SPECIAL NOTES:\s*(.*?)(?:\n\n|$)', docstring, re.DOTALL)
    if notes_match:
        info["special_notes"] = notes_match.group(1).strip()
    
    return info

def extract_class_info(node: ast.ClassDef) -> Dict[str, Any]:
    """Extract information about a class from its AST node."""
    class_info = {
        "name": node.name,
        "purpose": "",
        "key_methods": [],
        "inheritance": "",
        "dependencies": []
    }
    
    # Extract docstring
    docstring = ast.get_docstring(node)
    if docstring:
        # First line is usually the purpose
        class_info["purpose"] = docstring.strip().split('\n')[0].strip()
    
    # Extract inheritance
    if node.bases:
        base_names = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_names.append(f"{base.attr}")
        class_info["inheritance"] = ", ".join(base_names)
    
    # Extract methods
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and not item.name.startswith('_') or item.name == '__init__':
            method_info = {
                "name": item.name,
                "signature": f"{item.name}({format_args(item.args)})",
                "brief_description": ""
            }
            
            # Extract method docstring
            method_docstring = ast.get_docstring(item)
            if method_docstring:
                # First line is usually the brief description
                method_info["brief_description"] = method_docstring.strip().split('\n')[0].strip()
            
            class_info["key_methods"].append(method_info)
    
    return class_info

def format_args(args: ast.arguments) -> str:
    """Format function arguments as a string."""
    arg_parts = []
    
    # Add self parameter
    if args.args and args.args[0].arg == 'self':
        arg_parts.append('self')
        args_to_process = args.args[1:]
    else:
        args_to_process = args.args
    
    # Process regular args
    for arg in args_to_process:
        if hasattr(arg, 'annotation') and arg.annotation:
            if isinstance(arg.annotation, ast.Name):
                arg_parts.append(f"{arg.arg}: {arg.annotation.id}")
            elif isinstance(arg.annotation, ast.Attribute):
                arg_parts.append(f"{arg.arg}: {arg.annotation.attr}")
            else:
                arg_parts.append(arg.arg)
        else:
            arg_parts.append(arg.arg)
    
    # Process defaults
    defaults_offset = len(args_to_process) - len(args.defaults)
    for i, default in enumerate(args.defaults):
        arg_idx = i + defaults_offset
        if arg_idx < len(arg_parts):
            if isinstance(default, ast.Constant):
                default_value = repr(default.value)
            else:
                default_value = "..."
            arg_parts[arg_idx] = f"{arg_parts[arg_idx]} = {default_value}"
    
    return ", ".join(arg_parts)

def extract_function_info(node: ast.FunctionDef) -> Dict[str, Any]:
    """Extract information about a function from its AST node."""
    function_info = {
        "name": node.name,
        "signature": f"{node.name}({format_args(node.args)})",
        "brief_description": ""
    }
    
    # Extract docstring
    docstring = ast.get_docstring(node)
    if docstring:
        # First line is usually the brief description
        function_info["brief_description"] = docstring.strip().split('\n')[0].strip()
    
    return function_info

def analyze_file(file_path: str) -> Dict[str, Any]:
    """Analyze a Python file and extract metadata."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return {"error": "Could not parse file"}
    
    metadata = {
        "filename": os.path.basename(file_path),
        "module_purpose": "",
        "key_classes": [],
        "key_functions": [],
        "external_dependencies": [],
        "complexity_score": 5  # Default moderate complexity
    }
    
    # Extract module docstring
    module_docstring = ast.get_docstring(tree)
    if module_docstring:
        docstring_info = extract_docstring_info(module_docstring)
        if docstring_info.get("module_purpose"):
            metadata["module_purpose"] = docstring_info["module_purpose"]
        if docstring_info.get("dependencies"):
            metadata["external_dependencies"] = [
                dep for dep in docstring_info["dependencies"] 
                if not dep.startswith('.')
            ]
    
    # Extract classes and functions
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_info = extract_class_info(node)
            metadata["key_classes"].append(class_info)
        elif isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
            # Skip extract_file_metadata itself
            if node.name != "extract_file_metadata":
                function_info = extract_function_info(node)
                metadata["key_functions"].append(function_info)
    
    # Estimate complexity
    if len(metadata["key_classes"]) > 3 or len(metadata["key_functions"]) > 5:
        metadata["complexity_score"] = 7  # Higher complexity
    elif len(metadata["key_classes"]) <= 1 and len(metadata["key_functions"]) <= 2:
        metadata["complexity_score"] = 3  # Lower complexity
    
    return metadata

def generate_metadata_function(metadata: Dict[str, Any]) -> str:
    """Generate the extract_file_metadata function code."""
    function_code = [
        "\n\ndef extract_file_metadata(file_path=__file__):",
        '    """',
        '    Extract structured metadata about this module.',
        '    ',
        '    Args:',
        '        file_path: Path to the source file (defaults to current file)',
        '        ',
        '    Returns:',
        '        dict: Structured metadata about the module\'s purpose and components',
        '    """',
        '    return {',
        f'        "filename": os.path.basename(file_path),',
    ]
    
    # Add module purpose
    purpose = metadata.get("module_purpose", "")
    if not purpose and len(metadata["key_classes"]) > 0:
        purpose = f"Contains {', '.join(c['name'] for c in metadata['key_classes'][:3])}"
        if len(metadata["key_classes"]) > 3:
            purpose += " and other classes"
    
    function_code.append(f'        "module_purpose": "{purpose}",')
    
    # Add key classes
    if metadata["key_classes"]:
        function_code.append('        "key_classes": [')
        for class_info in metadata["key_classes"]:
            function_code.append('            {')
            function_code.append(f'                "name": "{class_info["name"]}",')
            function_code.append(f'                "purpose": "{class_info["purpose"]}",')
            
            if class_info["key_methods"]:
                function_code.append('                "key_methods": [')
                for method in class_info["key_methods"]:
                    function_code.append('                    {')
                    function_code.append(f'                        "name": "{method["name"]}",')
                    function_code.append(f'                        "signature": "{method["signature"]}",')
                    function_code.append(f'                        "brief_description": "{method["brief_description"]}"')
                    function_code.append('                    },')
                function_code.append('                ],')
            
            if class_info["inheritance"]:
                function_code.append(f'                "inheritance": "{class_info["inheritance"]}",')
            
            # Simplified dependencies
            function_code.append(f'                "dependencies": ["torch", "typing"]')
            function_code.append('            },')
        function_code.append('        ],')
    
    # Add key functions
    if metadata["key_functions"]:
        function_code.append('        "key_functions": [')
        for func_info in metadata["key_functions"]:
            function_code.append('            {')
            function_code.append(f'                "name": "{func_info["name"]}",')
            function_code.append(f'                "signature": "{func_info["signature"]}",')
            function_code.append(f'                "brief_description": "{func_info["brief_description"]}"')
            function_code.append('            },')
        function_code.append('        ],')
    
    # Add external dependencies
    function_code.append(f'        "external_dependencies": {metadata["external_dependencies"]},')
    
    # Add complexity score
    function_code.append(f'        "complexity_score": {metadata["complexity_score"]}  # {"Low" if metadata["complexity_score"] <= 3 else "Moderate" if metadata["complexity_score"] <= 6 else "High"} complexity')
    
    # Close function
    function_code.append('    }')
    
    return '\n'.join(function_code)

def append_metadata_function(file_path: str) -> Tuple[bool, str]:
    """Analyze a file and append the extract_file_metadata function to it."""
    try:
        # Check if file already has extract_file_metadata
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if "def extract_file_metadata" in content:
            return False, "File already has extract_file_metadata function"
        
        # Make sure the file imports os
        if "import os" not in content:
            content = content.split('\n')
            # Add import os after other imports
            import_inserted = False
            for i, line in enumerate(content):
                if line.startswith('import ') or line.startswith('from '):
                    if i + 1 < len(content) and (not content[i+1].startswith('import ') and not content[i+1].startswith('from ')):
                        content.insert(i + 1, 'import os')
                        import_inserted = True
                        break
            
            if not import_inserted:
                # Just add it at the top
                content.insert(0, 'import os')
            
            content = '\n'.join(content)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Analyze the file
        metadata = analyze_file(file_path)
        if "error" in metadata:
            return False, metadata["error"]
        
        # Generate function code
        function_code = generate_metadata_function(metadata)
        
        # Append to file
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(function_code)
        
        return True, "Successfully added extract_file_metadata function"
    
    except Exception as e:
        return False, f"Error: {str(e)}"

def process_directory(directory: str):
    """Process all Python files in a directory recursively."""
    results = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                success, message = append_metadata_function(file_path)
                results.append((file_path, success, message))
    
    return results

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python add_metadata.py <directory_or_file_path>")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if os.path.isdir(path):
        results = process_directory(path)
        
        # Print summary
        success_count = sum(1 for _, success, _ in results if success)
        print(f"Processed {len(results)} files, {success_count} successful, {len(results) - success_count} failed")
        
        # Print failures
        failures = [(path, msg) for path, success, msg in results if not success]
        if failures:
            print("\nFailures:")
            for path, msg in failures:
                print(f"  {path}: {msg}")
    else:
        success, message = append_metadata_function(path)
        print(f"{'Success' if success else 'Failed'}: {message}")