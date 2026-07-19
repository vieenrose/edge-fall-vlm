"""synthgen — synthetic fall/danger data generation for the AI-Care-2 VLM.

Pure modules (no bpy) importable anywhere: config, cameras (math), rationale, quality,
scene (sampling). Blender-only entry points live in render.py / bodies.py and only run
inside `blenderproc run`.
"""
__all__ = ["config", "cameras", "rationale", "quality", "scene"]
