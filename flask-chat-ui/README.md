# Simple Chat Application

A lightweight chat application with a clean UI that connects to various AI models through a Flask backend.

## Features

- Clean, responsive user interface
- Model selection dropdown with visual styling
- Session-based conversation tracking
- RESTful API for chat functionality

## Project Structure

```
chat_app/
├── app.py                # Main Flask application
├── static/               # Static files directory
├── templates/            # Templates directory
│   └── index.html        # Chat UI HTML
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## API Endpoints

### Start Session
- **URL**: `/api/start`
- **Method**: `POST`
- **Data Params**:
  ```json
  {
    "session_id": "unique-session-id",
    "option": "selected-model-option"
  }
  ```
- **Success Response**:
  ```json
  {
    "success": true,
    "message": "Welcome to the chat! Using option: selected-model-option"
  }
  ```

### Send Message
- **URL**: `/api/chat`
- **Method**: `POST`
- **Data Params**:
  ```json
  {
    "session_id": "unique-session-id",
    "option": "selected-model-option",
    "message": "user message"
  }
  ```
- **Success Response**:
  ```json
  {
    "success": true,
    "response": "bot response"
  }
  ```

## Setup Instructions

### Prerequisites
- Python 3.7+
- Flask

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app.py
   ```
4. Access the application at `http://localhost:5000`

## Customization

### Adding New Models
To add a new model option:

1. Add the option to the dropdown in `templates/index.html`
2. Update the server-side logic in `app.py` to handle the new model

### Implementing Model Integration
The current implementation includes placeholder logic for actual AI model integration. To connect with real AI models:

1. Add model-specific API credentials
2. Implement the API calls in the `process_message` function
3. Handle response formatting as needed

## License

This project is released under the MIT License.