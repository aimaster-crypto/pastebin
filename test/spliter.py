import re
import os

def extract_class_name(java_code):
    """
    Extract the class or interface name from Java code using regex.
    Assumes the declaration is 'public class ClassName' or 'public interface InterfaceName'.
    Returns None if no class or interface name is found.
    """
    match = re.search(r'public\s+(?:class|interface)\s+(\w+)', java_code)
    return match.group(1) if match else None

def save_to_file(code, filename, output_dir="output"):
    """
    Save the code to a file in the specified output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(code)
    print(f"Saved file: {filepath}")

def split_java_code(input_data, input_type='file'):
    """
    Split Java code blocks from input_data (file or string) and save to separate .java files.
    input_type: 'file' if input_data is a file path, 'string' if input_data is raw text.
    """
    # Read input based on type
    if input_type == 'file':
        with open(input_data, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        content = input_data

    # Regex to find ```java ... ``` blocks
    pattern = r'```java\s*([\s\S]*?)\s*```'
    java_blocks = re.findall(pattern, content)

    if not java_blocks:
        print("No Java code blocks found.")
        return

    # Process each Java code block
    for i, java_code in enumerate(java_blocks):
        java_code = java_code.strip()
        class_name = extract_class_name(java_code)
        
        # Determine filename
        if class_name:
            filename = f"{class_name}.java"
        else:
            filename = f"JavaCode_{i+1}.java"
        
        # Save the code to a file
        save_to_file(java_code, filename)

# Example usage
if __name__ == "__main__":
    FILE_PATH = ""
    data = open(FILE_PATH, "r").read()
    split_java_code(data, input_type='string')
