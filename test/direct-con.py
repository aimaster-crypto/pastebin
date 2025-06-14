import os
import ollama

# Change this to match your local model name if different
MODEL_NAME = "llama3.2:3b"

def convert_soap_to_rest(java_code):
    prompt = f"""
You are a senior Java developer.

Convert the following Spring SOAP @Endpoint class into a Spring Boot REST controller.

Requirements:
- Replace @Endpoint with @RestController
- Replace @PayloadRoot methods with @PostMapping or @GetMapping as appropriate
- Use idiomatic REST method names and paths
- Preserve method logic and parameters
- Remove all SOAP-specific annotations
- Keep class and method structure similar where possible

Here is the Java code:
{java_code}
Return only valid Java code in your response.
"""
    response = ollama.chat(model=MODEL_NAME, messages=[
        {"role": "user", "content": prompt}
    ])
    return response['message']['content']

def convert_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".java"):
            input_path = os.path.join(input_dir, file)
            with open(input_path, "r") as f:
                java_code = f.read()

            print(f"Converting {file}...")
            rest_code = convert_soap_to_rest(java_code)

            output_file = os.path.join(output_dir, file.replace(".java", "RestController.java"))
            with open(output_file, "w") as f:
                f.write(rest_code)

if __name__ == "__main__":
    convert_directory("soap_endpoints", "rest_controllers")
