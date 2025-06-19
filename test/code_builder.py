#!/usr/bin/env python3
"""
Automated JUnit 5 Test Generator for Multi-Module Spring Boot Projects
Uses CodeLlama via Ollama API to generate comprehensive test classes
"""

import os
import json
import requests
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SpringBootTestGenerator:
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "codellama:7b"):
        """
        Initialize the test generator
        
        Args:
            ollama_url: Ollama server URL
            model: CodeLlama model to use (codellama:7b, codellama:13b, codellama:34b)
        """
        self.ollama_url = ollama_url
        self.model = model
        self.api_url = f"{ollama_url}/api/generate"
        self.processed_classes = set()
        
    def find_modules(self, project_root: Path) -> List[Path]:
        """Find all modules in a multi-module project (directories with pom.xml)"""
        modules = []
        
        for item in project_root.rglob("pom.xml"):
            module_dir = item.parent
            # Skip target directories and nested dependencies
            if "target" not in str(module_dir) and ".m2" not in str(module_dir):
                modules.append(module_dir)
        
        # Sort by depth to process parent modules first
        modules.sort(key=lambda x: len(x.parts))
        
        logger.info(f"Found {len(modules)} modules: {[m.name for m in modules]}")
        return modules
    
    def find_java_classes(self, module_path: Path) -> List[Path]:
        """Find all Java source files in a module's src/main/java directory"""
        java_src_dir = module_path / "src" / "main" / "java"
        
        if not java_src_dir.exists():
            logger.warning(f"No src/main/java directory found in {module_path}")
            return []
        
        java_files = list(java_src_dir.rglob("*.java"))
        logger.info(f"Found {len(java_files)} Java files in {module_path.name}")
        
        return java_files
    
    def extract_class_info(self, java_file: Path) -> Dict[str, str]:
        """Extract class information from Java file"""
        try:
            content = java_file.read_text(encoding='utf-8')
            
            # Extract package name
            package_match = re.search(r'package\s+([^;]+);', content)
            package_name = package_match.group(1).strip() if package_match else ""
            
            # Extract class name
            class_match = re.search(r'(?:public\s+)?(?:abstract\s+)?(?:final\s+)?class\s+(\w+)', content)
            class_name = class_match.group(1) if class_match else java_file.stem
            
            # Determine class type based on annotations and naming
            class_type = self.determine_class_type(content, class_name)
            
            return {
                "package": package_name,
                "class_name": class_name,
                "class_type": class_type,
                "content": content,
                "file_path": str(java_file)
            }
        except Exception as e:
            logger.error(f"Error extracting class info from {java_file}: {e}")
            return {}
    
    def determine_class_type(self, content: str, class_name: str) -> str:
        """Determine the type of Spring Boot class based on content and naming"""
        content_lower = content.lower()
        class_name_lower = class_name.lower()
        
        if "@restcontroller" in content_lower or "@controller" in content_lower:
            return "controller"
        elif "@service" in content_lower or "service" in class_name_lower:
            return "service"
        elif "@repository" in content_lower or "repository" in class_name_lower:
            return "repository"
        elif "@component" in content_lower:
            return "component"
        elif "@configuration" in content_lower:
            return "configuration"
        elif "entity" in class_name_lower or "@entity" in content_lower:
            return "entity"
        else:
            return "general"
    
    def generate_test_prompt(self, class_info: Dict[str, str]) -> str:
        """Generate the prompt for CodeLlama to create JUnit 5 tests"""
        class_type = class_info["class_type"]
        class_name = class_info["class_name"]
        package = class_info["package"]
        content = class_info["content"]
        
        base_prompt = f"""
Generate a complete JUnit 5 test class for the following Spring Boot {class_type} class.

Requirements:
1. Use JUnit 5 annotations (@Test, @BeforeEach, @AfterEach, etc.)
2. Include appropriate Spring Boot test annotations based on class type:
   - For Controllers: @WebMvcTest, @MockMvc
   - For Services: @ExtendWith(MockitoExtension.class), @Mock, @InjectMocks
   - For Repositories: @DataJpaTest
   - For Components: @SpringBootTest
3. Mock all dependencies using @Mock or @MockBean
4. Test all public methods
5. Include positive and negative test scenarios
6. Use proper assertions (assertThat, assertEquals, etc.)
7. Follow naming convention: {class_name}Test
8. Use package: {package}

Original class to test:
```java
{content}
```

Generate ONLY the complete test class code without any explanations:
"""
        
        return base_prompt.strip()
    
    def call_ollama(self, prompt: str) -> Optional[str]:
        """Call Ollama API to generate test code"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent code generation
                    "top_p": 0.9,
                    "num_predict": 2000  # Allow longer responses
                }
            }
            
            logger.info("Calling Ollama API...")
            response = requests.post(
                self.api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120  # 2 minute timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to Ollama failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error calling Ollama: {e}")
            return None
    
    def clean_generated_code(self, code: str) -> str:
        """Clean and format the generated test code"""
        # Remove markdown code blocks if present
        code = re.sub(r'```java\s*', '', code)
        code = re.sub(r'```\s*$', '', code)
        
        # Remove any explanatory text before the class
        lines = code.split('\n')
        class_start = -1
        
        for i, line in enumerate(lines):
            if 'package ' in line or 'import ' in line or 'class ' in line:
                class_start = i
                break
        
        if class_start > 0:
            code = '\n'.join(lines[class_start:])
        
        return code.strip()
    
    def save_test_class(self, module_path: Path, class_info: Dict[str, str], test_code: str):
        """Save the generated test class to the appropriate test directory"""
        package = class_info["package"]
        class_name = class_info["class_name"]
        test_class_name = f"{class_name}Test"
        
        # Create test directory structure
        test_base_dir = module_path / "src" / "test" / "java"
        
        if package:
            package_dirs = package.split('.')
            test_dir = test_base_dir / Path(*package_dirs)
        else:
            test_dir = test_base_dir
        
        # Create directories if they don't exist
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Save test file
        test_file_path = test_dir / f"{test_class_name}.java"
        
        try:
            test_file_path.write_text(test_code, encoding='utf-8')
            logger.info(f"✅ Generated test: {test_file_path}")
        except Exception as e:
            logger.error(f"Failed to save test file {test_file_path}: {e}")
    
    def generate_tests_for_module(self, module_path: Path):
        """Generate tests for all Java classes in a module"""
        logger.info(f"\n📦 Processing module: {module_path.name}")
        
        java_files = self.find_java_classes(module_path)
        
        if not java_files:
            logger.warning(f"No Java files found in {module_path}")
            return
        
        for java_file in java_files:
            if str(java_file) in self.processed_classes:
                continue
                
            logger.info(f"🔍 Processing class: {java_file.name}")
            
            class_info = self.extract_class_info(java_file)
            if not class_info:
                continue
            
            # Check if test already exists
            test_name = f"{class_info['class_name']}Test.java"
            test_dir = module_path / "src" / "test" / "java"
            
            if class_info["package"]:
                package_dirs = class_info["package"].split('.')
                test_path = test_dir / Path(*package_dirs) / test_name
            else:
                test_path = test_dir / test_name
            
            if test_path.exists():
                logger.info(f"⏭️ Test already exists: {test_path}")
                continue
            
            # Generate test using Ollama
            prompt = self.generate_test_prompt(class_info)
            test_code = self.call_ollama(prompt)
            
            if test_code:
                cleaned_code = self.clean_generated_code(test_code)
                self.save_test_class(module_path, class_info, cleaned_code)
                self.processed_classes.add(str(java_file))
            else:
                logger.error(f"❌ Failed to generate test for {java_file.name}")
    
    def generate_tests_for_project(self, project_root: Path):
        """Generate tests for entire multi-module project"""
        logger.info(f"🚀 Starting test generation for project: {project_root.absolute()}")
        
        # Check if Ollama is accessible
        if not self.check_ollama_connection():
            logger.error("❌ Cannot connect to Ollama. Please ensure Ollama is running.")
            return
        
        # Find all modules
        modules = self.find_modules(project_root)
        
        if not modules:
            logger.info("⚠️ No modules found. Treating as single-module project.")
            modules = [project_root]
        
        # Process each module
        for module in modules:
            try:
                self.generate_tests_for_module(module)
            except Exception as e:
                logger.error(f"Error processing module {module}: {e}")
        
        logger.info(f"\n✅ Test generation completed! Processed {len(self.processed_classes)} classes.")
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama server is accessible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

def main():
    parser = argparse.ArgumentParser(description="Generate JUnit 5 tests for Spring Boot project using Ollama")
    parser.add_argument("project_path", nargs="?", default=".", 
                       help="Path to Spring Boot project root (default: current directory)")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                       help="Ollama server URL (default: http://localhost:11434)")
    parser.add_argument("--model", default="codellama:7b",
                       help="CodeLlama model to use (default: codellama:7b)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    project_path = Path(args.project_path).resolve()
    
    if not project_path.exists():
        logger.error(f"Project path does not exist: {project_path}")
        return
    
    generator = SpringBootTestGenerator(args.ollama_url, args.model)
    generator.generate_tests_for_project(project_path)

if __name__ == "__main__":
    main()
