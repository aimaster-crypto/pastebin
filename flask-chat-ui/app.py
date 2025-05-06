from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

# In-memory session storage (replace with a database in production)
active_sessions = {}

@app.route('/')
def index():
    """Render the chat interface"""
    return render_template('index.html')

@app.route('/api/start', methods=['POST'])
def start_session():
    """
    Initialize a new chat session
    Expected input: 
    {
        "session_id": "unique-session-id",
        "option": "selected-model-option"
    }
    """
    try:
        data = request.json
        session_id = data.get('session_id')
        option = data.get('option')
        
        # Log the request
        logger.info(f"Starting new session: {session_id} with option: {option}")
        
        # Store session information (implement your logic here)
        active_sessions[session_id] = {
            "option": option,
            "created_at": None,  # Add any session data you need
            "history": []
        }
        
        # Return success response
        return jsonify({
            "success": True,
            "message": f"Welcome to the chat! Using option: {option}"
        })
        
    except Exception as e:
        logger.error(f"Error starting session: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Failed to start session"
        }), 500

@app.route('/api/chat', methods=['POST'])
def process_message():
    """
    Process a chat message
    Expected input:
    {
        "session_id": "unique-session-id",
        "option": "selected-model-option",
        "message": "user message"
    }
    """
    try:
        data = request.json
        session_id = data.get('session_id')
        option = data.get('option')
        user_message = data.get('message')
        
        # Log the request
        logger.info(f"Received message for session {session_id}, option {option}: {user_message}")
        
        # Check if session exists
        if session_id not in active_sessions:
            return jsonify({
                "success": False,
                "error": "Session not found"
            }), 404
            
        # Process the message (implement your logic here)
        # ...
        
        # Store message in history
        active_sessions[session_id]["history"].append({
            "role": "user",
            "content": user_message
        })
        
        # Generate a response (implement your logic here)
        response = f"Echo: {user_message}"
        
        # Store response in history
        active_sessions[session_id]["history"].append({
            "role": "assistant",
            "content": response
        })
        
        # Return the response
        return jsonify({
            "success": True,
            "response": response
        })
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Failed to process message"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(debug=True, host='0.0.0.0', port=port)