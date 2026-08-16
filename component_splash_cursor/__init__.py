import os
import streamlit.components.v1 as components

# Build static directory path for the custom component
parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "frontend")

_component_func = components.declare_component(
    "splash_cursor_component",
    path=build_dir
)

def splash_cursor_component(
    sim_resolution=128,
    dye_resolution=1440,
    capture_resolution=512,
    density_dissipation=3.5,
    velocity_dissipation=2.0,
    pressure=0.1,
    pressure_iterations=20,
    curl=3.0,
    splat_radius=0.2,
    splat_force=6000.0,
    shading=True,
    color_update_speed=10.0,
    rainbow_mode=True,
    color="#7c3aed",
    transparent=True,
    key=None
):
    """
    Renders the interactive fluid SplashCursor WebGL component from React Bits.
    """
    return _component_func(
        SIM_RESOLUTION=sim_resolution,
        DYE_RESOLUTION=dye_resolution,
        CAPTURE_RESOLUTION=capture_resolution,
        DENSITY_DISSIPATION=density_dissipation,
        VELOCITY_DISSIPATION=velocity_dissipation,
        PRESSURE=pressure,
        PRESSURE_ITERATIONS=pressure_iterations,
        CURL=curl,
        SPLAT_RADIUS=splat_radius,
        SPLAT_FORCE=splat_force,
        SHADING=shading,
        COLOR_UPDATE_SPEED=color_update_speed,
        RAINBOW_MODE=rainbow_mode,
        COLOR=color,
        TRANSPARENT=transparent,
        key=key,
        default=None
    )
