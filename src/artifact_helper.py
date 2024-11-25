# artifact_helper.py
from typing import Dict, Any, Tuple
import numpy as np
from tools.elevation_tool import analyze_elevation
from tools.route_tool import find_route
from tools.urban_tool import urban_suitability_analysis


def get_elevation_artifact(llm_response, index: int) -> Dict[str, Any]:
    """
    Extract the artifact from an LLM response for the elevation analysis tool.
    """
    if hasattr(llm_response, 'tool_calls') and llm_response.tool_calls:
        tool_call = llm_response.tool_calls[index]
        tool_response = analyze_elevation.invoke(tool_call)
        return {
            'llm_response': llm_response,
            'content': tool_response.content,
            'artifact': tool_response.artifact,
            'tool_call': tool_call
        }
    return None


def get_route_artifact(llm_response, index: int) -> Dict[str, Any]:
    """
    Extract the artifact from an LLM response for the route finding tool.
    """
    if hasattr(llm_response, 'tool_calls') and llm_response.tool_calls:
        tool_call = llm_response.tool_calls[index]
        tool_response = find_route.invoke(tool_call)
        return {
            'llm_response': llm_response,
            'content': tool_response.content,
            'artifact': tool_response.artifact,
            'tool_call': tool_call
        }
    return None


def get_urban_suitability_artifact(llm_response, index: int) -> Dict[str, Any]:
    """
    Extract the artifact from an LLM response for the urban suitability analysis tool.
    """
    if hasattr(llm_response, 'tool_calls') and llm_response.tool_calls:
        tool_call = llm_response.tool_calls[index]
        tool_response = urban_suitability_analysis.invoke(tool_call)
        return {
            'llm_response': llm_response,
            'content': tool_response.content,
            'artifact': tool_response.artifact,
            'tool_call': tool_call
        }
    return None
