import os
import streamlit.components.v1 as components

# Build static directory path for the custom component
parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "frontend")

_component_func = components.declare_component(
    "galaxy_component",
    path=build_dir
)

def galaxy_component(
    focal=(0.5, 0.5),
    rotation=(1.0, 0.0),
    starSpeed=0.5,
    density=1.5,
    hueShift=0.0,
    disableAnimation=False,
    speed=1.0,
    mouseInteraction=False,
    glowIntensity=0.5,
    saturation=0.0,
    mouseRepulsion=False,
    repulsionStrength=0.0,
    twinkleIntensity=0.35,
    rotationSpeed=0.08,
    autoCenterRepulsion=0.0,
    transparent=True,
    key=None
):
    """
    Renders the pure white Galaxy background WebGL component.
    """
    return _component_func(
        focal=list(focal),
        rotation=list(rotation),
        starSpeed=starSpeed,
        density=density,
        hueShift=hueShift,
        disableAnimation=disableAnimation,
        speed=speed,
        mouseInteraction=mouseInteraction,
        glowIntensity=glowIntensity,
        saturation=saturation,
        mouseRepulsion=mouseRepulsion,
        repulsionStrength=repulsionStrength,
        twinkleIntensity=twinkleIntensity,
        rotationSpeed=rotationSpeed,
        autoCenterRepulsion=autoCenterRepulsion,
        transparent=transparent,
        key=key,
        default=None
    )
