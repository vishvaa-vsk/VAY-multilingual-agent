import os
import streamlit.components.v1 as components

# Declare Streamlit custom component for SpecularButton
parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "frontend")

_component_func = components.declare_component(
    "specular_button",
    path=build_dir
)

def specular_button(
    label="Get Started",
    size="lg",
    radius=18,
    tint="#ffffff",
    tintOpacity=0.0,
    blur=0,
    textColor="#f5f5f5",
    lineColor="#ffffff",
    baseColor="#525252",
    intensity=1.0,
    shineSize=10,
    shineFade=40,
    thickness=1.0,
    speed=0.35,
    followMouse=True,
    proximity=250,
    autoAnimate=False,
    disabled=False,
    fullWidth=True,
    key=None
):
    """
    Renders the SpecularButton React Bits WebGL component in Streamlit.
    Returns:
        dict: containing user interaction events (e.g. {'event': 'click', 'id': ...}) or None.
    """
    return _component_func(
        label=label,
        size=size,
        radius=radius,
        tint=tint,
        tintOpacity=tintOpacity,
        blur=blur,
        textColor=textColor,
        lineColor=lineColor,
        baseColor=baseColor,
        intensity=intensity,
        shineSize=shineSize,
        shineFade=shineFade,
        thickness=thickness,
        speed=speed,
        followMouse=followMouse,
        proximity=proximity,
        autoAnimate=autoAnimate,
        disabled=disabled,
        fullWidth=fullWidth,
        key=key,
        default=None
    )
