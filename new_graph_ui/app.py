import sqlite3
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# SQLite database setup
DATABASE = 'workflows.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS workflows (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                current_stage INTEGER NOT NULL,
                created TEXT NOT NULL,
                updated TEXT NOT NULL
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS stages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id TEXT NOT NULL,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                duration INTEGER DEFAULT 0,
                started_at TEXT,
                completed_at TEXT,
                updated TEXT,
                FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE CASCADE
            )
        ''')
        conn.commit()

def migrate_db():
    with get_db_connection() as conn:
        c = conn.cursor()
        try:
            c.execute("PRAGMA table_info(stages)")
            columns = [col[1] for col in c.fetchall()]
            if 'updated' not in columns:
                logger.info("Adding 'updated' column to stages table")
                c.execute('ALTER TABLE stages ADD COLUMN updated TEXT')
                c.execute('''
                    UPDATE stages 
                    SET updated = (SELECT created FROM workflows WHERE workflows.id = stages.workflow_id)
                    WHERE updated IS NULL
                ''')
                conn.commit()
                logger.info("Migration completed: added 'updated' column to stages")
            else:
                logger.info("No migration needed: 'updated' column already exists")
        except Exception as e:
            logger.error(f"Error during database migration: {str(e)}")
            raise

init_db()
migrate_db()

def serialize_workflow(workflow, stages):
    return {
        'id': workflow['id'],
        'name': workflow['name'],
        'status': workflow['status'],
        'current_stage': workflow['current_stage'],
        'created': workflow['created'],
        'updated': workflow['updated'],
        'stages': [
            {
                'id': stage['id'],
                'name': stage['name'],
                'status': stage['status'],
                'duration': stage['duration'],
                'started_at': stage['started_at'],
                'completed_at': stage['completed_at'],
                'updated': stage['updated']
            } for stage in stages
        ]
    }

@app.route('/api/workflows', methods=['GET'])
def get_workflows():
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT * FROM workflows')
            workflows = c.fetchall()
            result = {}
            for workflow in workflows:
                c.execute('SELECT * FROM stages WHERE workflow_id = ?', (workflow['id'],))
                stages = c.fetchall()
                result[workflow['id']] = serialize_workflow(workflow, stages)
            return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting workflows: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/workflows/<workflow_id>', methods=['GET'])
def get_workflow(workflow_id):
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT * FROM workflows WHERE id = ?', (workflow_id,))
            workflow = c.fetchone()
            if not workflow:
                return jsonify({'error': 'Workflow not found'}), 404
            c.execute('SELECT * FROM stages WHERE workflow_id = ?', (workflow_id,))
            stages = c.fetchall()
            return jsonify(serialize_workflow(workflow, stages))
    except Exception as e:
        logger.error(f"Error getting workflow {workflow_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/workflows', methods=['POST'])
def create_workflow():
    try:
        data = request.json
        if not data or 'name' not in data:
            return jsonify({'error': 'Name is required'}), 400
        if "id" not in data:
            workflow_id = str(uuid.uuid4())
        else:
            workflow_id = data["id"]
        created = datetime.now().isoformat()
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO workflows (id, name, status, current_stage, created, updated)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (workflow_id, data['name'], data.get('status', 'pending'), 
                  data.get('current_stage', 1), created, created))
            
            for stage in data.get('stages', []):
                c.execute('''
                    INSERT INTO stages (workflow_id, name, status, duration, started_at, completed_at, updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (workflow_id, stage['name'], stage.get('status', 'pending'), 
                      stage.get('duration', 0), stage.get('started_at'), 
                      stage.get('completed_at'), created))
            
            conn.commit()
        return jsonify({'id': workflow_id, 'message': 'Workflow created successfully'}), 201
    except Exception as e:
        logger.error(f"Error creating workflow: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/workflows/<workflow_id>', methods=['PUT'])
def update_workflow(workflow_id):
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Validate inputs
        valid_statuses = ['pending', 'running', 'completed']
        if 'status' in data and data['status'] not in valid_statuses:
            return jsonify({'error': f"Invalid status. Must be one of {valid_statuses}"}), 400
        if 'name' in data and (not data['name'] or not isinstance(data['name'], str)):
            return jsonify({'error': 'Name must be a non-empty string'}), 400
        if 'current_stage' in data and (not isinstance(data['current_stage'], int) or data['current_stage'] < 1):
            return jsonify({'error': 'Current stage must be a positive integer'}), 400

        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT * FROM workflows WHERE id = ?', (workflow_id,))
            workflow = c.fetchone()
            if not workflow:
                return jsonify({'error': 'Workflow not found'}), 404
            
            # Use existing values if not provided in the request
            name = data.get('name', workflow['name'])
            status = data.get('status', workflow['status'])
            current_stage = data.get('current_stage', workflow['current_stage'])
            
            c.execute('''
                UPDATE workflows 
                SET name = ?, status = ?, current_stage = ?, updated = ?
                WHERE id = ?
            ''', (name, status, current_stage, datetime.now().isoformat(), workflow_id))
            conn.commit()
        return jsonify({'message': 'Workflow updated successfully'})
    except Exception as e:
        logger.error(f"Error updating workflow {workflow_id}: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/workflows/<workflow_id>/stages/<int:stage_id>/status', methods=['PUT'])
def update_stage_status(workflow_id, stage_id):
    try:
        data = request.json
        status = data.get('status')
        duration = data.get('duration')

        valid_statuses = ['pending', 'running', 'completed']
        if not status or status not in valid_statuses:
            return jsonify({'error': f"Invalid status. Must be one of {valid_statuses}"}), 400

        if duration is not None and not isinstance(duration, int):
            return jsonify({'error': 'Duration must be an integer'}), 400

        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT * FROM workflows WHERE id = ?', (workflow_id,))
            if not c.fetchone():
                return jsonify({'error': 'Workflow not found'}), 404
            
            c.execute('SELECT * FROM stages WHERE id = ? AND workflow_id = ?', 
                     (stage_id, workflow_id))
            if not c.fetchone():
                return jsonify({'error': 'Stage not found'}), 404
            
            update_fields = ['status = ?', 'updated = ?']
            update_values = [status, datetime.now().isoformat()]
            
            if duration is not None:
                update_fields.append('duration = ?')
                update_values.append(duration)
            
            if status == 'running':
                update_fields.append('started_at = ?')
                update_values.append(datetime.now().isoformat())
            elif status == 'completed':
                update_fields.append('completed_at = ?')
                update_values.append(datetime.now().isoformat())
            
            update_values.append(stage_id)
            update_values.append(workflow_id)
            
            c.execute(f'''
                UPDATE stages 
                SET {', '.join(update_fields)}
                WHERE id = ? AND workflow_id = ?
            ''', update_values)
            
            if status == 'running':
                c.execute('''
                    UPDATE workflows 
                    SET current_stage = ?, status = 'running', updated = ?
                    WHERE id = ?
                ''', (stage_id, datetime.now().isoformat(), workflow_id))
            
            conn.commit()
        return jsonify({'message': 'Stage status updated successfully'})
    except Exception as e:
        logger.error(f"Error updating stage {stage_id} for workflow {workflow_id}: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/workflows/<workflow_id>', methods=['DELETE'])
def delete_workflow(workflow_id):
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT * FROM workflows WHERE id = ?', (workflow_id,))
            if not c.fetchone():
                return jsonify({'error': 'Workflow not found'}), 404
            c.execute('DELETE FROM workflows WHERE id = ?', (workflow_id,))
            conn.commit()
        return jsonify({'message': 'Workflow deleted successfully'})
    except Exception as e:
        logger.error(f"Error deleting workflow {workflow_id}: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)