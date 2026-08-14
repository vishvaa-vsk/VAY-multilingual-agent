import os
import streamlit.components.v1 as components

# Build static directory path for the custom component
parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "frontend")

_component_func = components.declare_component(
    "aurora_component",
    path=build_dir
)

def aurora_component(
    colorStops=None,
    amplitude=1.0,
    blend=0.5,
    speed=0.5,
    key=None
):
    """
    Renders the Aurora background WebGL component from React Bits.
    colorStops: list of 3 hex colors [color1, color2, color3] representing the aurora gradient
    amplitude: height intensity of the aurora effect
    blend: blending of the aurora effect with the background
    speed: animation speed of the aurora
    """
    if colorStops is None:
        # Default real Aurora Borealis colors: vibrant emerald green, cyan teal, and velvet indigo
        colorStops = ["#00FF87", "#60EFFF", "#7C3AED"]
        
    return _component_func(
        colorStops=list(colorStops),
        amplitude=float(amplitude),
        blend=float(blend),
        speed=float(speed),
        key=key,
        default=None
    )
