import os
import sys
import requests
from zeep import Client

OLLAMA_MODEL = "llama3"  # Or codellama, mistral, etc.
OLLAMA_URL = "http://localhost:11434/api/generate"

def parse_wsdl(wsdl_path):
    client = Client(wsdl_path)
    service = list(client.wsdl.services.values())[0]
    port = list(service.ports.values())[0]
    operations = port.binding._operations

    op_list = []
    for name, op in operations.items():
        input_params = [p.name for p in op.input.body.parts]
        op_list.append({
            "operation": name,
            "params": input_params
        })
    return op_list

def read_soap_impl_files(folder_path):
    combined = ""
    for fname in os.listdir(folder_path):
        if fname.endswith(".java"):
            with open(os.path.join(folder_path, fname), "r") as f:
                combined += f"\n\n// FILE: " + fname + "\n" + f.read()
    return combined

def query_ollama(prompt, model=OLLAMA_MODEL):
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False}
    )
    if response.status_code == 200:
        return response.json()["response"]
    else:
        raise Exception(f"Ollama error {response.status_code}:\n{response.text}")

def build_prompt_1(operations, soap_code):
    prompt = """You are a software engineer helping migrate a SOAP service to REST.

Below is a list of operations extracted from the WSDL, and the current Java implementation.

**Your task:**
Generate a complete OpenAPI 3.0 YAML specification for the service.
Use proper JSON schemas, HTTP methods, and meaningful type definitions.

WSDL Operations:
"""
    for op in operations:
        prompt += f"- {op['operation']}({', '.join(op['params'])})\n"
    prompt += "\nJava SOAP Implementation:\n" + soap_code + "\n\nReturn ONLY the OpenAPI 3.0 YAML."
    return prompt

def build_prompt_2(openapi_yaml):
    return f"""You are a Spring Boot developer.

Below is an OpenAPI 3.0 specification.

**Your task:**
Generate a Java Spring Boot REST controller class that implements this API.
Use @RestController, @RequestBody, etc., and return dummy data or TODOs if needed.
Use only Java code in your output.

OpenAPI Spec:
{openapi_yaml}
"""

def main(wsdl_file, impl_folder):
    print("📦 Parsing WSDL...")
    operations = parse_wsdl(wsdl_file)

    print("📄 Reading SOAP implementation files...")
    soap_code = read_soap_impl_files(impl_folder)

    print("🧠 Step 1: Generating OpenAPI spec with Ollama...")
    prompt1 = build_prompt_1(operations, soap_code)
    openapi_yaml = query_ollama(prompt1)

    with open("openapi.yaml", "w") as f:
        f.write(openapi_yaml)

    print("🧠 Step 2: Generating Spring Boot controller from OpenAPI...")
    prompt2 = build_prompt_2(openapi_yaml)
    controller_code = query_ollama(prompt2)

    with open("SoapRestController.java", "w") as f:
        f.write(controller_code)

    print("✅ Done! Generated `openapi.yaml` and `SoapRestController.java`.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python soap_to_rest_2step_ollama.py <path-to-wsdl> <path-to-soap-impl-folder>")
    else:
        main(sys.argv[1], sys.argv[2])
