import requests
import json
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkflowAPIClient:
    def __init__(self, base_url: str = "http://localhost:8080/api"):
        """Initialize the API client with the base URL."""
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, 
                     params: Optional[Dict] = None) -> Dict:
        """Helper method to make HTTP requests."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            raise APIError(f"Request failed: {str(e)}", 
                         getattr(e.response, 'status_code', None))

    def get_workflows(self) -> Dict:
        """Get all workflows."""
        return self._make_request("GET", "workflows")

    def get_workflow(self, workflow_id: str) -> Dict:
        """Get a specific workflow by ID."""
        return self._make_request("GET", f"workflows/{workflow_id}")

    def get_stages(self, workflow_id: str) -> List[Dict]:
        """Get all stages for a specific workflow."""
        workflow = self.get_workflow(workflow_id)
        return workflow.get('stages', [])

    def get_stage(self, workflow_id: str, stage_id: int) -> Dict:
        """Get a specific stage by ID from a workflow."""
        if not isinstance(stage_id, int) or stage_id < 1:
            raise ValueError("Stage ID must be a positive integer")
        
        stages = self.get_stages(workflow_id)
        for stage in stages:
            if stage.get('id') == stage_id:
                return stage
        raise ValueError(f"Stage {stage_id} not found in workflow {workflow_id}")

    def get_stage_by_name(self, workflow_id: str, stage_name: str) -> Dict:
        """Get a specific stage by name from a workflow."""
        if not stage_name or not isinstance(stage_name, str):
            raise ValueError("Stage name must be a non-empty string")
        
        stages = self.get_stages(workflow_id)
        for stage in stages:
            if stage.get('name') == stage_name:
                return stage
        raise ValueError(f"Stage '{stage_name}' not found in workflow {workflow_id}")

    def create_workflow(self, name: str, stages: List[Dict], 
                      status: str = "pending", 
                      current_stage: int = 1) -> Dict:
        """Create a new workflow."""
        if not name or not isinstance(name, str):
            raise ValueError("Name must be a non-empty string")
        valid_statuses = ['pending', 'running', 'completed']
        if status not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}")
        if not isinstance(current_stage, int) or current_stage < 1:
            raise ValueError("Current stage must be a positive integer")
        
        data = {
            "name": name,
            "status": status,
            "current_stage": current_stage,
            "stages": stages
        }
        return self._make_request("POST", "workflows", data=data)

    def update_workflow(self, workflow_id: str, name: Optional[str] = None, 
                       status: Optional[str] = None, 
                       current_stage: Optional[int] = None) -> Dict:
        """Update a workflow."""
        # Fetch current workflow to get existing values
        try:
            workflow = self.get_workflow(workflow_id)
        except APIError as e:
            if e.status_code == 404:
                raise ValueError(f"Workflow {workflow_id} not found")
            raise
        
        # Use existing values if not provided
        data = {
            "name": name if name is not None else workflow['name'],
            "status": status if status is not None else workflow['status'],
            "current_stage": current_stage if current_stage is not None else workflow['current_stage']
        }

        # Validate inputs
        valid_statuses = ['pending', 'running', 'completed']
        if data['status'] not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}")
        if not data['name'] or not isinstance(data['name'], str):
            raise ValueError("Name must be a non-empty string")
        if not isinstance(data['current_stage'], int) or data['current_stage'] < 1:
            raise ValueError("Current stage must be a positive integer")

        return self._make_request("PUT", f"workflows/{workflow_id}", data=data)

    def update_stage_status(self, workflow_id: str, stage_id: int, 
                          status: str, duration: Optional[int] = None) -> Dict:
        """Update a stage's status."""
        valid_statuses = ['pending', 'running', 'completed']
        if not status or status not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}")
        if duration is not None and not isinstance(duration, int):
            raise ValueError("Duration must be an integer")
        
        data = {"status": status}
        if duration is not None:
            data["duration"] = duration
        return self._make_request("PUT", 
                                f"workflows/{workflow_id}/stages/{stage_id}/status", 
                                data=data)

    def delete_workflow(self, workflow_id: str) -> Dict:
        """Delete a workflow."""
        return self._make_request("DELETE", f"workflows/{workflow_id}")

    def health_check(self) -> Dict:
        """Check API health status."""
        return self._make_request("GET", "health")

    def close(self):
        """Close the HTTP session."""
        self.session.close()

class APIError(Exception):
    """Custom exception for API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

# Example usage
if __name__ == "__main__":
    client = WorkflowAPIClient()
    
    try:
        # Check API health
        health = client.health_check()
        print("Health check:", health)

        # Create a new workflow
        stages = [
            {"name": "Stage 1", "status": "pending"},
            {"name": "Stage 2", "status": "pending"}
        ]
        new_workflow = client.create_workflow(
            name="Test Workflow",
            stages=stages,
            status="pending",
            current_stage=1
        )
        print("Created workflow:", new_workflow)

        # Get workflow ID
        workflow_id = new_workflow["id"]

        # Get all workflows
        workflows = client.get_workflows()
        print("All workflows:", workflows)

        # Get all stages
        stages = client.get_stages(workflow_id)
        print("All stages:", stages)

        # Get stage by ID
        stage_id = stages[0]["id"]  # Get the first stage's ID
        stage = client.get_stage(workflow_id, stage_id)
        print("Stage details (by ID):", stage)

        # Get stage by name
        stage_name = "Stage 1"
        stage_by_name = client.get_stage_by_name(workflow_id, stage_name)
        print(f"Stage details (by name '{stage_name}'):", stage_by_name)

        # Update stage status
        stage_update = client.update_stage_status(
            workflow_id=workflow_id,
            stage_id=stage_id,
            status="running",
            duration=100
        )
        print("Stage update:", stage_update)

        # Get specific workflow
        workflow = client.get_workflow(workflow_id)
        print("Workflow details:", workflow)

        # Update workflow with all required fields
        update = client.update_workflow(
            workflow_id=workflow_id,
            name="Updated Workflow",
            status="running",
            current_stage=workflow['current_stage']  # Preserve current_stage
        )
        print("Workflow update:", update)

        # Delete workflow
        delete = client.delete_workflow(workflow_id)
        print("Delete workflow:", delete)

    except APIError as e:
        print(f"API Error: {e.message} (Status: {e.status_code})")
    except ValueError as e:
        print(f"Validation Error: {str(e)}")
    finally:
        client.close()