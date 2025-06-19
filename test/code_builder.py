#!/usr/bin/env python3
"""
Advanced Spring Boot Test Generator - Edition 2
Validates test execution with 'mvn test -Dtest=ClassName' and regenerates failed tests
"""

import os
import json
import requests
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedSpringTestGenerator:
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "codellama:7b"):
        """
        Initialize the advanced test generator with validation capabilities
        """
        self.ollama_url = ollama_url
        self.model = model
        self.api_url = f"{ollama_url}/api/generate"
        self.processed_classes = set()
        self.existing_tests = set()
        self.generated_tests = set()
        self.failed_tests = set()
        self.passed_tests = set()
        self.regenerated_tests = set()
        
    def find_modules(self, project_root: Path) -> List[Path]:
        """Find all modules in a multi-module project"""
        modules = []
        
        for item in project_root.rglob("pom.xml"):
            module_dir = item.parent
            if "target" not in str(module_dir) and ".m2" not in str(module_dir):
                modules.append(module_dir)
        
        modules.sort(key=lambda x: len(x.parts))
        logger.info(f"Found {len(modules)} modules: {[m.name for m in modules]}")
        return modules
    
    def find_existing_tests(self, module_path: Path) -> Dict[str, Path]:
        """Find all existing test files in a module and return mapping"""
        existing_tests = {}
        test_dir = module_path / "src" / "test" / "java"
        
        if test_dir.exists():
            for test_file in test_dir.rglob("*Test.java"):
                test_name = test_file.stem
                if test_name.endswith("Test"):
                    original_class = test_name[:-4]
                    existing_tests[original_class] = test_file
                    logger.debug(f"Found existing test: {test_file}")
        
        return existing_tests
    
    def find_java_classes(self, module_path: Path) -> List[Path]:
        """Find all Java source files in a module"""
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
            
            # Determine class type
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
        """Determine the type of Spring Boot class"""
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
    
    def run_maven_test(self, module_path: Path, test_class_name: str, timeout: int = 120) -> Tuple[bool, str]:
        """
        Run maven test for a specific test class
        Returns: (success: bool, output: str)
        """
        try:
            cmd = ["mvn", "test", f"-Dtest={test_class_name}", "-q", "--batch-mode"]
            
            logger.info(f"🧪 Running: {' '.join(cmd)} in {module_path}")
            
            result = subprocess.run(
                cmd,
                cwd=module_path,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            success = result.returncode == 0
            output = result.stdout + result.stderr
            
            if success:
                logger.info(f"✅ Test passed: {test_class_name}")
                self.passed_tests.add(test_class_name)
            else:
                logger.warning(f"❌ Test failed: {test_class_name}")
                self.failed_tests.add(test_class_name)
                logger.debug(f"Error output: {output}")
            
            return success, output
            
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Test timeout: {test_class_name}")
            self.failed_tests.add(test_class_name)
            return False, "Test execution timeout"
        except Exception as e:
            logger.error(f"Error running test {test_class_name}: {e}")
            self.failed_tests.add(test_class_name)
            return False, str(e)
    
    def analyze_test_failure(self, test_output: str) -> Dict[str, str]:
        """Analyze test failure output to provide better context for regeneration"""
        failure_info = {
            "compilation_errors": [],
            "runtime_errors": [],
            "missing_dependencies": [],
            "suggestions": []
        }
        
        lines = test_output.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            
            # Compilation errors
            if "compilation failure" in line_lower or "cannot find symbol" in line_lower:
                failure_info["compilation_errors"].append(line.strip())
            
            # Runtime errors
            elif any(error in line_lower for error in ["nullpointerexception", "illegalargumentexception", "assertionerror"]):
                failure_info["runtime_errors"].append(line.strip())
            
            # Missing dependencies
            elif any(missing in line_lower for error in ["could not autowire", "no qualifying bean", "unsatisfied dependency"]):
                failure_info["missing_dependencies"].append(line.strip())
        
        # Generate suggestions based on errors
        if failure_info["compilation_errors"]:
            failure_info["suggestions"].append("Fix import statements and class references")
        if failure_info["missing_dependencies"]:
            failure_info["suggestions"].append("Add proper @MockBean annotations for dependencies")
        if failure_info["runtime_errors"]:
            failure_info["suggestions"].append("Initialize mocks properly and add null checks")
        
        return failure_info
    
    def generate_improved_test_prompt(self, class_info: Dict[str, str], failure_info: Optional[Dict] = None) -> str:
        """Generate improved prompt based on failure analysis"""
        class_type = class_info["class_type"]
        class_name = class_info["class_name"]
        package = class_info["package"]
        content = class_info["content"]
        
        base_prompt = f"""
Generate a robust and working JUnit 5 test class for the following Spring Boot {class_type} class.

CRITICAL REQUIREMENTS:
1. The test MUST compile and run successfully
2. Use proper Spring Boot test annotations for {class_type} type
3. Mock ALL external dependencies properly
4. Include proper imports and setup
5. Handle edge cases and null scenarios
6. Use correct assertion methods

Class Type Specific Requirements:
"""
        
        if class_type == "controller":
            base_prompt += """
- Use @WebMvcTest(ClassName.class) annotation
- Include @MockBean for service dependencies
- Use MockMvc for HTTP testing
- Test all endpoints with proper HTTP methods
- Include request/response validation
"""
        elif class_type == "service":
            base_prompt += """
- Use @ExtendWith(MockitoExtension.class)
- Use @Mock for repository/external dependencies
- Use @InjectMocks for the service under test
- Test business logic thoroughly
- Mock all external calls
"""
        elif class_type == "repository":
            base_prompt += """
- Use @DataJpaTest for JPA repository tests
- Use @Autowired for the repository
- Include proper entity setup
- Test CRUD operations
"""
        else:
            base_prompt += """
- Use @SpringBootTest for integration testing
- Use @MockBean for external dependencies
- Include proper component scanning
"""
        
        if failure_info and failure_info.get("suggestions"):
            base_prompt += f"""
PREVIOUS TEST FAILURES - MUST ADDRESS:
{chr(10).join(f"- {suggestion}" for suggestion in failure_info["suggestions"])}

COMMON ISSUES TO AVOID:
- Missing import statements
- Incorrect annotation usage
- Uninitialized mocks
- Missing @MockBean for Spring dependencies
- Incorrect test class structure
"""
        
        base_prompt += f"""
Generate a complete, working test class with:
- Package: {package}
- Class name: {class_name}Test
- All necessary imports
- Proper setup and teardown
- Multiple test methods covering different scenarios
- Exception handling tests

Original class to test:
```java
{content}
```

Generate ONLY the complete test class code that will compile and run successfully:
"""
        
        return base_prompt.strip()
    
    def call_ollama(self, prompt: str) -> Optional[str]:
        """Call Ollama API with enhanced parameters for better code generation"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.05,  # Very low for consistent code
                    "top_p": 0.8,
                    "top_k": 40,
                    "num_predict": 3000,
                    "repeat_penalty": 1.1
                }
            }
            
            logger.info("🤖 Calling Ollama API for test generation...")
            response = requests.post(
                self.api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=200
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return None
    
    def clean_generated_code(self, code: str) -> str:
        """Clean and format the generated test code"""
        # Remove markdown code blocks
        code = re.sub(r'```java\s*', '', code)
        code = re.sub(r'```\s*$', '', code)
        
        # Remove any explanatory text
        lines = code.split('\n')
        java_start = -1
        
        for i, line in enumerate(lines):
            if any(keyword in line for keyword in ['package ', 'import ', '@', 'class ']):
                java_start = i
                break
        
        if java_start > 0:
            code = '\n'.join(lines[java_start:])
        
        return code.strip()
    
    def save_test_class(self, module_path: Path, class_info: Dict[str, str], test_code: str) -> Path:
        """Save the generated test class and return the file path"""
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
        
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file_path = test_dir / f"{test_class_name}.java"
        
        try:
            test_file_path.write_text(test_code, encoding='utf-8')
            logger.info(f"💾 Saved test: {test_file_path}")
            return test_file_path
        except Exception as e:
            logger.error(f"Failed to save test file {test_file_path}: {e}")
            raise
    
    def validate_and_regenerate_test(self, module_path: Path, class_info: Dict[str, str], 
                                   test_file_path: Path, max_attempts: int = 3) -> bool:
        """
        Validate test execution and regenerate if it fails
        Returns True if test passes, False if all attempts fail
        """
        class_name = class_info["class_name"]
        test_class_name = f"{class_name}Test"
        
        for attempt in range(1, max_attempts + 1):
            logger.info(f"🔄 Validation attempt {attempt}/{max_attempts} for {test_class_name}")
            
            # Run the test
            success, output = self.run_maven_test(module_path, test_class_name)
            
            if success:
                logger.info(f"✅ Test validation successful: {test_class_name}")
                return True
            
            if attempt < max_attempts:
                logger.warning(f"❌ Test failed on attempt {attempt}, regenerating...")
                
                # Analyze failure
                failure_info = self.analyze_test_failure(output)
                
                # Generate improved test
                improved_prompt = self.generate_improved_test_prompt(class_info, failure_info)
                new_test_code = self.call_ollama(improved_prompt)
                
                if new_test_code:
                    cleaned_code = self.clean_generated_code(new_test_code)
                    
                    # Save the regenerated test
                    try:
                        test_file_path.write_text(cleaned_code, encoding='utf-8')
                        logger.info(f"🔄 Regenerated test (attempt {attempt + 1}): {test_file_path}")
                        self.regenerated_tests.add(test_class_name)
                        
                        # Wait a bit before next validation
                        time.sleep(2)
                    except Exception as e:
                        logger.error(f"Failed to save regenerated test: {e}")
                        break
                else:
                    logger.error(f"Failed to generate improved test for {class_name}")
                    break
            else:
                logger.error(f"❌ All validation attempts failed for {test_class_name}")
                logger.debug(f"Final error output: {output}")
        
        return False
    
    def process_class_with_validation(self, module_path: Path, java_file: Path, 
                                    class_info: Dict[str, str], force_regenerate: bool = False) -> bool:
        """Process a single class with full validation cycle"""
        class_name = class_info["class_name"]
        test_class_name = f"{class_name}Test"
        
        # Check if test exists
        existing_tests = self.find_existing_tests(module_path)
        test_exists = class_name in existing_tests
        
        if test_exists and not force_regenerate:
            # Validate existing test
            logger.info(f"🔍 Validating existing test: {test_class_name}")
            success, _ = self.run_maven_test(module_path, test_class_name)
            
            if success:
                logger.info(f"✅ Existing test passes: {test_class_name}")
                return True
            else:
                logger.warning(f"⚠️ Existing test fails, will regenerate: {test_class_name}")
        
        # Generate new test
        logger.info(f"🔧 Generating test for: {class_name}")
        
        prompt = self.generate_improved_test_prompt(class_info)
        test_code = self.call_ollama(prompt)
        
        if not test_code:
            logger.error(f"❌ Failed to generate test for {class_name}")
            return False
        
        cleaned_code = self.clean_generated_code(test_code)
        test_file_path = self.save_test_class(module_path, class_info, cleaned_code)
        
        self.generated_tests.add(test_class_name)
        
        # Validate and potentially regenerate
        return self.validate_and_regenerate_test(module_path, class_info, test_file_path)
    
    def generate_tests_for_module(self, module_path: Path, force_regenerate: bool = False, 
                                max_workers: int = 2):
        """Generate and validate tests for all Java classes in a module"""
        logger.info(f"\n📦 Processing module: {module_path.name}")
        
        java_files = self.find_java_classes(module_path)
        
        if not java_files:
            logger.warning(f"No Java files found in {module_path}")
            return
        
        # Extract class info for all files
        classes_to_process = []
        for java_file in java_files:
            class_info = self.extract_class_info(java_file)
            if class_info:
                classes_to_process.append((java_file, class_info))
        
        logger.info(f"🔄 Processing {len(classes_to_process)} classes with validation")
        
        # Process classes (can be done in parallel, but limited workers to avoid overwhelming)
        success_count = 0
        failed_count = 0
        
        for java_file, class_info in classes_to_process:
            try:
                success = self.process_class_with_validation(
                    module_path, java_file, class_info, force_regenerate
                )
                
                if success:
                    success_count += 1
                    self.processed_classes.add(str(java_file))
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing {class_info.get('class_name', 'unknown')}: {e}")
                failed_count += 1
        
        logger.info(f"📊 Module {module_path.name} results: {success_count} passed, {failed_count} failed")
    
    def run_full_test_suite(self, module_path: Path) -> Tuple[int, int]:
        """Run all tests in a module and return pass/fail counts"""
        try:
            logger.info(f"🧪 Running full test suite for module: {module_path.name}")
            
            result = subprocess.run(
                ["mvn", "test", "-q", "--batch-mode"],
                cwd=module_path,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for full suite
            )
            
            output = result.stdout + result.stderr
            
            # Parse test results
            passed = len(re.findall(r'Tests run: \d+, Failures: 0, Errors: 0', output))
            failed = len(re.findall(r'Tests run: \d+, Failures: [1-9]', output))
            
            if result.returncode == 0:
                logger.info(f"✅ Full test suite passed for {module_path.name}")
            else:
                logger.warning(f"⚠️ Some tests failed in {module_path.name}")
                logger.debug(f"Test output: {output}")
            
            return passed, failed
            
        except Exception as e:
            logger.error(f"Error running full test suite for {module_path}: {e}")
            return 0, 0
    
    def generate_tests_for_project(self, project_root: Path, force_regenerate: bool = False, 
                                 validate_full_suite: bool = True):
        """Generate and validate tests for entire multi-module project"""
        logger.info(f"🚀 Starting advanced test generation for project: {project_root.absolute()}")
        
        if not self.check_ollama_connection():
            logger.error("❌ Cannot connect to Ollama. Please ensure Ollama is running.")
            return
        
        modules = self.find_modules(project_root)
        
        if not modules:
            logger.info("⚠️ No modules found. Treating as single-module project.")
            modules = [project_root]
        
        total_modules = len(modules)
        
        for i, module in enumerate(modules, 1):
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing module {i}/{total_modules}: {module.name}")
                logger.info(f"{'='*60}")
                
                self.generate_tests_for_module(module, force_regenerate)
                
                if validate_full_suite:
                    passed, failed = self.run_full_test_suite(module)
                    logger.info(f"📊 Module test suite: {passed} passed, {failed} failed")
                
            except Exception as e:
                logger.error(f"Error processing module {module}: {e}")
        
        # Final summary
        self.print_final_summary()
    
    def print_final_summary(self):
        """Print comprehensive summary of the test generation process"""
        logger.info(f"\n{'='*80}")
        logger.info("🎯 FINAL SUMMARY")
        logger.info(f"{'='*80}")
        
        logger.info(f"📊 STATISTICS:")
        logger.info(f"   • Classes processed: {len(self.processed_classes)}")
        logger.info(f"   • Tests generated: {len(self.generated_tests)}")
        logger.info(f"   • Tests passed validation: {len(self.passed_tests)}")
        logger.info(f"   • Tests failed validation: {len(self.failed_tests)}")
        logger.info(f"   • Tests regenerated: {len(self.regenerated_tests)}")
        
        if self.passed_tests:
            logger.info(f"\n✅ PASSED TESTS:")
            for test in sorted(self.passed_tests):
                logger.info(f"   • {test}")
        
        if self.failed_tests:
            logger.info(f"\n❌ FAILED TESTS:")
            for test in sorted(self.failed_tests):
                logger.info(f"   • {test}")
        
        if self.regenerated_tests:
            logger.info(f"\n🔄 REGENERATED TESTS:")
            for test in sorted(self.regenerated_tests):
                logger.info(f"   • {test}")
        
        success_rate = (len(self.passed_tests) / len(self.generated_tests) * 100) if self.generated_tests else 0
        logger.info(f"\n🎯 SUCCESS RATE: {success_rate:.1f}%")
        
        logger.info(f"\n💡 RECOMMENDATIONS:")
        if self.failed_tests:
            logger.info("   • Review failed tests manually for complex scenarios")
            logger.info("   • Consider adding custom test templates for specific patterns")
        if success_rate > 80:
            logger.info("   • Great success rate! Consider running full integration tests")
        else:
            logger.info("   • Consider reviewing class complexity and dependencies")
        
        logger.info(f"{'='*80}")
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama server is accessible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                if self.model in model_names:
                    logger.info(f"✅ Ollama connection verified. Model {self.model} available.")
                    return True
                else:
                    logger.error(f"❌ Model {self.model} not found. Available models: {model_names}")
                    return False
            return False
        except Exception as e:
            logger.error(f"❌ Cannot connect to Ollama: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(
        description="Advanced Spring Boot Test Generator - Validate & Regenerate Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python advanced_test_generator.py                    # Generate tests for current directory
  python advanced_test_generator.py /path/to/project  # Generate tests for specific project
  python advanced_test_generator.py --force           # Force regenerate all tests
  python advanced_test_generator.py --no-validate     # Skip full suite validation
  python advanced_test_generator.py --model codellama:13b  # Use larger model
        """
    )
    
    parser.add_argument("project_path", nargs="?", default=".", 
                       help="Path to Spring Boot project root")
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                       help="Ollama server URL")
    parser.add_argument("--model", default="codellama:7b",
                       help="CodeLlama model to use (codellama:7b, codellama:13b, codellama:34b)")
    parser.add_argument("--force", "-f", action="store_true",
                       help="Force regenerate all tests, even if they exist and pass")
    parser.add_argument("--no-validate", action="store_true",
                       help="Skip full test suite validation")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    project_path = Path(args.project_path).resolve()
    
    if not project_path.exists():
        logger.error(f"Project path does not exist: {project_path}")
        return 1
    
    # Check if Maven is available
    try:
        subprocess.run(["mvn", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("❌ Maven is not available. Please install Maven and ensure it's in PATH.")
        return 1
    
    generator = AdvancedSpringTestGenerator(args.ollama_url, args.model)
    generator.generate_tests_for_project(
        project_path, 
        args.force, 
        not args.no_validate
    )
    
    # Return exit code based on success rate
    if generator.generated_tests:
        success_rate = len(generator.passed_tests) / len(generator.generated_tests)
        return 0 if success_rate >= 0.5 else 1
    else:
        return 0

if __name__ == "__main__":
    exit(main())
