import os
import streamlit.components.v1 as components

# For simple deployment and runtime, we build a static directory
_RELEASE = True

parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "frontend")

_component_func = components.declare_component(
    "strands_component",
    path=build_dir
)

def strands_component(
    colors=None,
    count=3,
    speed=0.5,
    amplitude=1.0,
    waviness=1.0,
    thickness=0.7,
    glow=2.6,
    taper=3.0,
    spread=1.0,
    hueShift=0.0,
    intensity=0.6,
    saturation=1.5,
    opacity=1.0,
    scale=1.5,
    glass=False,
    refraction=1.0,
    dispersion=1.0,
    glassSize=1.0,
    status="idle",
    audio_data=None,  # Base64-encoded WAV to play back to the user
    key=None
):
    """
    Renders the Strands WebGL component.
    Returns:
        dict: containing user interaction events like audio recorded from the mic.
    """
    if colors is None:
        colors = ["#FF4242", "#7C3AED", "#06B6D4", "#EAB308"]
        
    return _component_func(
        colors=colors,
        count=count,
        speed=speed,
        amplitude=amplitude,
        waviness=waviness,
        thickness=thickness,
        glow=glow,
        taper=taper,
        spread=spread,
        hueShift=hueShift,
        intensity=intensity,
        saturation=saturation,
        opacity=opacity,
        scale=scale,
        glass=glass,
        refraction=refraction,
        dispersion=dispersion,
        glassSize=glassSize,
        status=status,
        audio_data=audio_data,
        key=key,
        default=None
    )
