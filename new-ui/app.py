from flask import Flask, render_template, request, jsonify, session
import re
import time
import json
import random
from typing import Dict, Any, List
import uuid

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

def parse_agent_response(response: str) -> List[Dict[str, Any]]:
    """Parse AI agent response containing code, XML, images, and text."""
    blocks = []
    
    # Split by triple quotes to find annotated blocks
    parts = re.split(r'```(\w+)?\n(.*?)```', response, flags=re.DOTALL)
    
    for i in range(len(parts)):
        if i % 3 == 0:  # Regular text
            if parts[i].strip():
                blocks.append({"type": "text", "content": parts[i].strip()})
        elif i % 3 == 1:  # Language identifier
            lang = parts[i].strip() if parts[i] else "text"
            content = parts[i + 1] if i + 1 < len(parts) else ""
            if content.strip():
                if lang.lower() in ['xml', 'html']:
                    blocks.append({"type": "xml", "content": content.strip()})
                elif lang.lower() in ['python', 'javascript', 'java', 'cpp', 'c', 'sql', 'json']:
                    blocks.append({"type": "code", "language": lang, "content": content.strip()})
                else:
                    blocks.append({"type": "code", "language": lang, "content": content.strip()})
    
    return blocks

def ai_agent_function(query: str, codebase: str) -> str:
    """
    This is the function you'll implement to call your AI agent.
    
    Args:
        query (str): User's query/question
        codebase (str): Selected codebase from dropdown
        
    Returns:
        str: AI agent response (can contain code blocks, XML, etc.)
    """
    # TODO: Implement your AI agent logic here
    # This is a placeholder response for demonstration
    
    sample_responses = [
        f"""Here's a response to your query about: {query}

```python
def example_function():
    # This is sample code related to {codebase}
    print("Hello from {codebase}!")
    return "Sample response"
```

This code demonstrates how to handle your request in the context of {codebase}.""",
        
        f"""I understand you're asking about: {query}

```xml
<response>
    <codebase>{codebase}</codebase>
    <query>{query}</query>
    <status>processed</status>
</response>
```

The XML above shows the structured response format.""",
        
        f"""Your query "{query}" has been processed for codebase: {codebase}

```javascript
// Example JavaScript solution
function handleQuery(query, codebase) {{
    console.log(`Processing: ${{query}} for ${{codebase}}`);
    return "Response generated successfully";
}}
```

This function shows how to implement similar functionality in JavaScript."""
    ]
    
    # Simulate processing time
    time.sleep(1)
    
    # Return a random sample response
    return random.choice(sample_responses)

@app.route('/')
def index():
    """Main chat interface."""
    if 'messages' not in session:
        session['messages'] = []
    if 'selected_codebase' not in session:
        session['selected_codebase'] = 'General'
    
    codebases = [
        "General",
        "Python/Django",
        "JavaScript/React",
        "Java/Spring",
        "C++/Qt",
        "Ruby/Rails",
        "Go/Gin",
        "Rust/Actix",
        "TypeScript/Node.js",
        "PHP/Laravel"
    ]
    
    return render_template('index.html', 
                         messages=session['messages'], 
                         codebases=codebases,
                         selected_codebase=session['selected_codebase'])

@app.route('/send_message', methods=['POST'])
def send_message():
    """Handle sending a message."""
    data = request.json
    user_message = data.get('message', '').strip()
    codebase = data.get('codebase', 'General')
    
    if not user_message:
        return jsonify({'error': 'Message cannot be empty'}), 400
    
    # Initialize messages if not exists
    if 'messages' not in session:
        session['messages'] = []
    
    # Add user message
    user_msg = {
        'id': str(uuid.uuid4()),
        'role': 'user',
        'content': user_message,
        'codebase': codebase,
        'timestamp': time.time()
    }
    session['messages'].append(user_msg)
    session['selected_codebase'] = codebase
    
    # Get AI response
    ai_response = ai_agent_function(user_message, codebase)
    
    # Parse AI response
    parsed_blocks = parse_agent_response(ai_response)
    
    # Add AI message
    ai_msg = {
        'id': str(uuid.uuid4()),
        'role': 'assistant',
        'content': ai_response,
        'parsed_blocks': parsed_blocks,
        'timestamp': time.time()
    }
    session['messages'].append(ai_msg)
    
    # Save session
    session.modified = True
    
    return jsonify({
        'user_message': user_msg,
        'ai_message': ai_msg
    })

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    """Clear chat history."""
    session['messages'] = []
    session.modified = True
    return jsonify({'success': True})

@app.route('/set_codebase', methods=['POST'])
def set_codebase():
    """Set selected codebase."""
    data = request.json
    codebase = data.get('codebase', 'General')
    session['selected_codebase'] = codebase
    session.modified = True
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
