#!/usr/bin/env python3
"""
SOAP to REST Controller Converter using Ollama
This script analyzes SOAP endpoints and generates Spring Boot REST controllers
while preserving downstream service calls.
"""

import requests
import json
import os
import sys
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
import argparse

class SOAPToRESTConverter:
    def __init__(self, ollama_host: str = "http://localhost:11434", model: str = "llama2"):
        self.ollama_host = ollama_host
        self.model = model
        self.ollama_url = f"{ollama_host}/api/generate"
    
    def test_ollama_connection(self) -> bool:
        """Test if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags")
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def parse_wsdl(self, wsdl_content: str) -> Dict:
        """Parse WSDL content to extract service information"""
        try:
            root = ET.fromstring(wsdl_content)
            
            # Extract namespaces
            namespaces = {}
            for prefix, uri in root.attrib.items():
                if prefix.startswith('xmlns'):
                    ns_name = prefix.split(':')[-1] if ':' in prefix else 'default'
                    namespaces[ns_name] = uri
            
            # Extract service information
            services = []
            operations = []
            
            # Find all operations
            for elem in root.iter():
                if 'operation' in elem.tag.lower():
                    op_name = elem.get('name')
                    if op_name:
                        operations.append({
                            'name': op_name,
                            'element': elem
                        })
            
            return {
                'namespaces': namespaces,
                'operations': operations,
                'raw_content': wsdl_content
            }
        except ET.ParseError as e:
            print(f"Error parsing WSDL: {e}")
            return {}
    
    def analyze_soap_endpoint(self, soap_code: str, wsdl_context: str = "") -> str:
        """Analyze SOAP endpoint code to understand its structure"""
        wsdl_section = f"\n\nWSDL Context for reference:\n{wsdl_context}\n" if wsdl_context else ""
        
        analysis_prompt = f"""
        Analyze this Java SOAP endpoint code with @Endpoint annotation and extract the following information:
        1. Service name and class name
        2. All @PayloadRoot annotated methods and their operations
        3. Request/Response object types and their XML mappings
        4. Input parameters, their types, and XML element names
        5. Return types and response XML structure
        6. Any downstream service calls (database calls, external API calls, etc.)
        7. Business logic flow and transformations
        8. Exception handling patterns
        9. Any @Autowired dependencies
        10. Namespace URIs and local parts from @PayloadRoot
        
        SOAP Endpoint Code:
        {soap_code}
        {wsdl_section}
        
        Please provide a detailed structured analysis in the following format:
        - Service Name: 
        - Class Name: 
        - Endpoint Methods: (list each @PayloadRoot method with namespace and localPart)
        - Request Objects: (JAXB classes used for requests)
        - Response Objects: (JAXB classes used for responses)
        - Parameters: (detailed parameter analysis with XML mappings)
        - Return Types: (detailed return type analysis)
        - Downstream Calls: (all external service calls, DB operations, etc.)
        - Dependencies: (all @Autowired services)
        - Business Logic: (step-by-step flow for each operation)
        - Exception Handling: (how errors are handled and returned)
        - XML Namespaces: (all namespaces used)
        """
        
        return self.call_ollama(analysis_prompt)
    
    def generate_rest_controller(self, soap_analysis: str, soap_code: str, wsdl_info: Dict = None) -> str:
        """Generate Spring Boot REST controller based on SOAP analysis"""
        wsdl_section = ""
        if wsdl_info:
            wsdl_section = f"""
            
        WSDL Information for additional context:
        - Available Operations: {[op['name'] for op in wsdl_info.get('operations', [])]}
        - Namespaces: {wsdl_info.get('namespaces', {})}
        """
        
        conversion_prompt = f"""
        Based on the following SOAP @Endpoint analysis, generate a complete Spring Boot REST controller that:
        
        CRITICAL REQUIREMENTS:
        1. Convert each @PayloadRoot method to appropriate REST endpoints
        2. Map SOAP operations to REST HTTP methods (GET for queries, POST for creation, PUT for updates, DELETE for removal)
        3. Convert JAXB request/response objects to JSON-friendly DTOs or keep them if they work with Jackson
        4. Preserve ALL downstream service calls EXACTLY as they are
        5. Maintain the same business logic flow
        6. Convert XML-based error handling to REST-appropriate error responses
        7. Use proper Spring Boot REST annotations
        8. Handle request/response transformation between SOAP XML and REST JSON
        9. Preserve all @Autowired dependencies
        10. Maintain the same validation logic
        
        SOAP Analysis:
        {soap_analysis}
        {wsdl_section}
        
        Original SOAP Endpoint Code:
        {soap_code}
        
        Generate a complete Spring Boot REST controller with:
        
        1. CLASS STRUCTURE:
        - @RestController annotation
        - @RequestMapping for base path (derive from service name)
        - Keep all @Autowired dependencies from original
        - Proper package declaration and imports
        
        2. ENDPOINT METHODS:
        - Convert each @PayloadRoot method to @GetMapping, @PostMapping, etc.
        - Use @RequestBody for complex request objects
        - Use @PathVariable for ID-based operations
        - Use @RequestParam for simple query parameters
        - Return ResponseEntity with appropriate HTTP status codes
        
        3. REQUEST/RESPONSE HANDLING:
        - Convert JAXB objects to DTOs if needed for JSON serialization
        - Preserve all data fields and their mappings
        - Handle XML to JSON transformation gracefully
        
        4. BUSINESS LOGIC PRESERVATION:
        - Keep ALL downstream service calls identical
        - Preserve the exact same business logic flow
        - Maintain all validation rules
        - Keep the same transaction boundaries
        
        5. ERROR HANDLING:
        - Convert SOAP faults to REST exceptions
        - Use @ExceptionHandler for proper error responses
        - Return appropriate HTTP status codes
        - Preserve error message content
        
        6. DOCUMENTATION:
        - Add brief JavaDoc comments for each endpoint
        - Include parameter descriptions
        - Note the original SOAP operation mapping
        
        Generate ONLY the complete Java REST controller code without any explanations or markdown formatting.
        """
        
        return self.call_ollama(conversion_prompt)
    
    def call_ollama(self, prompt: str) -> str:
        """Make API call to Ollama"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 4000
            }
        }
        
        try:
            response = requests.post(
                self.ollama_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama: {e}")
            return ""
    
    def process_soap_file(self, file_path: str, wsdl_url: str = None) -> Dict[str, str]:
        """Process a SOAP endpoint file with optional WSDL context"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soap_code = f.read()
            
            # Validate that this is a SOAP endpoint file
            if "@Endpoint" not in soap_code:
                return {"error": "File does not contain @Endpoint annotation. Please provide a valid SOAP endpoint Java file."}
            
            wsdl_info = None
            wsdl_context = ""
            
            # Fetch WSDL if URL provided
            if wsdl_url:
                print(f"Fetching WSDL from: {wsdl_url}")
                try:
                    wsdl_response = requests.get(wsdl_url, timeout=30)
                    wsdl_response.raise_for_status()
                    wsdl_content = wsdl_response.text
                    wsdl_info = self.parse_wsdl(wsdl_content)
                    wsdl_context = wsdl_content[:2000]  # First 2000 chars for context
                    print("WSDL fetched successfully")
                except Exception as e:
                    print(f"Warning: Could not fetch WSDL: {e}")
                    print("Proceeding with SOAP file analysis only...")
            
            print(f"Analyzing SOAP endpoint: {file_path}")
            analysis = self.analyze_soap_endpoint(soap_code, wsdl_context)
            
            if not analysis:
                return {"error": "Failed to analyze SOAP endpoint"}
            
            print("Generating REST controller...")
            rest_controller = self.generate_rest_controller(analysis, soap_code, wsdl_info)
            
            if not rest_controller:
                return {"error": "Failed to generate REST controller"}
            
            result = {
                "analysis": analysis,
                "rest_controller": rest_controller,
                "original_soap": soap_code
            }
            
            if wsdl_info:
                result["wsdl_info"] = wsdl_info
                result["wsdl_content"] = wsdl_content
            
            return result
            
        except FileNotFoundError:
            return {"error": f"File not found: {file_path}"}
        except Exception as e:
            return {"error": f"Error processing file: {e}"}
    
    def process_wsdl_url(self, wsdl_url: str) -> Dict[str, str]:
        """Process a WSDL URL"""
        try:
            response = requests.get(wsdl_url, timeout=30)
            response.raise_for_status()
            
            wsdl_content = response.text
            wsdl_info = self.parse_wsdl(wsdl_content)
            
            if not wsdl_info:
                return {"error": "Failed to parse WSDL"}
            
            # Generate analysis prompt for WSDL
            wsdl_prompt = f"""
            Analyze this WSDL content and generate Spring Boot REST controllers for all operations:
            
            {wsdl_content}
            
            For each operation, create appropriate REST endpoints maintaining the same functionality.
            """
            
            rest_controllers = self.call_ollama(wsdl_prompt)
            
            return {
                "wsdl_info": wsdl_info,
                "rest_controllers": rest_controllers,
                "original_wsdl": wsdl_content
            }
            
        except requests.exceptions.RequestException as e:
            return {"error": f"Error fetching WSDL: {e}"}
    
    def save_output(self, result: Dict[str, str], output_dir: str, base_name: str):
        """Save generated REST controller to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save REST controller
        if "rest_controller" in result:
            controller_file = os.path.join(output_dir, f"{base_name}RestController.java")
            with open(controller_file, 'w', encoding='utf-8') as f:
                f.write(result["rest_controller"])
            print(f"✅ REST controller saved to: {controller_file}")
        
        if "rest_controllers" in result:
            controller_file = os.path.join(output_dir, f"{base_name}RestControllers.java")
            with open(controller_file, 'w', encoding='utf-8') as f:
                f.write(result["rest_controllers"])
            print(f"✅ REST controllers saved to: {controller_file}")
        
        # Save analysis
        if "analysis" in result:
            analysis_file = os.path.join(output_dir, f"{base_name}_endpoint_analysis.txt")
            with open(analysis_file, 'w', encoding='utf-8') as f:
                f.write(result["analysis"])
            print(f"📋 Analysis saved to: {analysis_file}")
        
        # Save original SOAP endpoint for reference
        if "original_soap" in result:
            original_file = os.path.join(output_dir, f"{base_name}_original_endpoint.java")
            with open(original_file, 'w', encoding='utf-8') as f:
                f.write(result["original_soap"])
            print(f"📄 Original SOAP endpoint saved to: {original_file}")
        
        # Save WSDL if available
        if "wsdl_content" in result:
            wsdl_file = os.path.join(output_dir, f"{base_name}_service.wsdl")
            with open(wsdl_file, 'w', encoding='utf-8') as f:
                f.write(result["wsdl_content"])
            print(f"🔗 WSDL saved to: {wsdl_file}")
        
        # Save WSDL analysis if available
        if "wsdl_info" in result:
            wsdl_info_file = os.path.join(output_dir, f"{base_name}_wsdl_analysis.json")
            with open(wsdl_info_file, 'w', encoding='utf-8') as f:
                json.dump(result["wsdl_info"], f, indent=2)
            print(f"🔍 WSDL analysis saved to: {wsdl_info_file}")
        
        # Create a summary file
        summary_file = os.path.join(output_dir, f"{base_name}_conversion_summary.md")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"# SOAP to REST Conversion Summary\n\n")
            f.write(f"**Service:** {base_name}\n")
            f.write(f"**Conversion Date:** {os.popen('date').read().strip()}\n\n")
            f.write(f"## Files Generated:\n")
            f.write(f"- `{base_name}RestController.java` - Main REST controller\n")
            f.write(f"- `{base_name}_endpoint_analysis.txt` - Detailed analysis\n")
            f.write(f"- `{base_name}_original_endpoint.java` - Original SOAP endpoint\n")
            if "wsdl_content" in result:
                f.write(f"- `{base_name}_service.wsdl` - WSDL definition\n")
                f.write(f"- `{base_name}_wsdl_analysis.json` - WSDL analysis\n")
            f.write(f"\n## Notes:\n")
            f.write(f"- All downstream service calls have been preserved\n")
            f.write(f"- Business logic remains unchanged\n")
            f.write(f"- Review the generated controller for any manual adjustments needed\n")
            f.write(f"- Test thoroughly before deploying to production\n")
        
        print(f"📝 Conversion summary saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert SOAP @Endpoint Java files to REST controllers using Ollama")
    parser.add_argument("--input", "-i", required=True, help="Input SOAP endpoint Java file")
    parser.add_argument("--wsdl", "-w", help="WSDL URL for additional context")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama host URL")
    parser.add_argument("--model", default="codellama", help="Ollama model to use (recommended: codellama, llama2, or mistral)")
    parser.add_argument("--name", "-n", help="Base name for output files")
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = SOAPToRESTConverter(args.ollama_host, args.model)
    
    # Test Ollama connection
    print("🔗 Testing Ollama connection...")
    if not converter.test_ollama_connection():
        print(f"❌ Error: Cannot connect to Ollama at {args.ollama_host}")
        print("Please ensure Ollama is running and accessible")
        print("You can start Ollama with: ollama serve")
        sys.exit(1)
    
    print(f"✅ Connected to Ollama at {args.ollama_host} using model: {args.model}")
    
    # Determine base name
    base_name = args.name
    if not base_name:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        # Remove common suffixes
        for suffix in ["Endpoint", "Service", "Impl"]:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file does not exist: {args.input}")
        sys.exit(1)
    
    if not args.input.endswith('.java'):
        print(f"⚠️  Warning: Input file does not have .java extension")
    
    # Process SOAP endpoint file
    print(f"🔄 Processing SOAP endpoint file: {args.input}")
    if args.wsdl:
        print(f"🔄 Using WSDL context from: {args.wsdl}")
        result = converter.process_soap_file(args.input, args.wsdl)
    else:
        print("ℹ️  Processing without WSDL context (consider providing --wsdl for better results)")
        result = converter.process_soap_file(args.input)
    
    # Handle errors
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        sys.exit(1)
    
    # Save results
    print(f"💾 Saving results to: {args.output}")
    converter.save_output(result, args.output, base_name)
    
    print(f"\n🎉 Conversion completed successfully!")
    print(f"📁 Check the '{args.output}' directory for all generated files")
    print(f"\n📋 Next steps:")
    print(f"   1. Review the generated {base_name}RestController.java")
    print(f"   2. Add any missing imports or dependencies")
    print(f"   3. Test the REST endpoints")
    print(f"   4. Update your application configuration if needed")

if __name__ == "__main__":
    main()
