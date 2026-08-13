import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="VAY",
    layout="wide"
)

st.markdown("""
<style>
    .block-container {
        padding: 0 !important;
    }

    header {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


html = r"""
<!DOCTYPE html>
<html>
<head>

<meta charset="UTF-8">

<style>
html, body {
    margin: 0;
    padding: 0;
    width: 100%;
    height: 600px;
    overflow: hidden;
    background: transparent;
}

#strands-container {
    position: relative;
    width: 100%;
    height: 600px;
    overflow: hidden;
}

canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    display: block;
}
</style>

</head>

<body>

<div id="strands-container"></div>

<script type="module">

import {
    Renderer,
    Program,
    Mesh,
    Color,
    Triangle
} from "https://esm.sh/ogl";


// =====================================================
// CONFIG
// =====================================================

const MAX_STRANDS = 12;
const MAX_COLORS = 8;


// =====================================================
// VERTEX SHADER
// =====================================================

const VERT = `#version 300 es

in vec2 position;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}

`;


// =====================================================
// FRAGMENT SHADER
// =====================================================

const FRAG = `#version 300 es

precision highp float;

uniform float uTime;
uniform vec2 uResolution;

uniform vec3 uColors[${MAX_COLORS}];

uniform int uColorCount;
uniform int uStrandCount;

uniform float uSpeed;
uniform float uAmplitude;
uniform float uWaviness;
uniform float uThickness;
uniform float uGlow;
uniform float uTaper;
uniform float uSpread;
uniform float uHueShift;
uniform float uIntensity;
uniform float uOpacity;
uniform float uScale;
uniform float uSaturation;

out vec4 fragColor;

const float PI = 3.14159265;


// -----------------------------------------------------
// Spectrum
// -----------------------------------------------------

vec3 spectrum(float t) {

    return 0.5 + 0.5 *
        cos(
            2.0 * PI *
            (t + vec3(0.00, 0.33, 0.67))
        );
}


// -----------------------------------------------------
// Palette
// -----------------------------------------------------

vec3 samplePalette(float t) {

    t = fract(t);

    float scaled =
        t * float(uColorCount);

    int idx =
        int(floor(scaled));

    float blend =
        fract(scaled);

    int nextIdx =
        idx + 1;

    if (nextIdx >= uColorCount)
        nextIdx = 0;

    return mix(
        uColors[idx],
        uColors[nextIdx],
        blend
    );
}


vec3 strandColor(float t) {

    if (uColorCount > 0)
        return samplePalette(t);

    return spectrum(t);
}


// =====================================================
// MAIN
// =====================================================

void main() {

    vec2 uv =
        (gl_FragCoord.xy -
        0.5 * uResolution)
        / uResolution.y;

    uv /=
        max(uScale, 0.0001);


    float e =
        0.06 +
        uIntensity * 0.94;


    float env =
        pow(
            max(
                cos(
                    uv.x *
                    PI *
                    1.3
                ),
                0.0
            ),
            uTaper
        );


    vec3 col =
        vec3(0.0);


    for (
        int i = 0;
        i < ${MAX_STRANDS};
        i++
    ) {

        if (i >= uStrandCount)
            break;


        float fi =
            float(i);


        float ph =
            fi *
            1.7 *
            uSpread;


        float freq =
            (2.0 +
            fi * 0.35)
            * uWaviness;


        float spd =
            1.4 +
            fi * 1.2;


        float tt =
            uTime *
            uSpeed;


        float w =

            sin(
                uv.x *
                freq +
                tt *
                spd +
                ph
            ) * 0.60

            +

            sin(
                uv.x *
                freq *
                1.1 -

                tt *
                spd *
                0.7 +

                ph *
                1.7
            ) * 0.40;


        float amp =
            (
                0.1 +
                0.02 * e
            )
            *
            env
            *
            uAmplitude;


        float y =
            w * amp;


        float d =
            abs(
                uv.y -
                y
            );


        float thick =
            (
                0.001 +
                0.05 * e
            )
            *
            (0.35 + env)
            *
            uThickness;


        float g =
            thick /
            (
                d +
                thick * 0.45
            );


        g =
            g * g;


        float h =
            fi /
            float(uStrandCount)

            +

            uv.x * 0.30

            +

            uTime * 0.04

            +

            uHueShift;


        col +=
            strandColor(h)
            *
            g
            *
            env;
    }


    col *=
        0.45 +
        0.7 * e;


    col =
        1.0 -
        exp(
            -col *
            uGlow
        );


    float gray =
        dot(
            col,
            vec3(
                0.2126,
                0.7152,
                0.0722
            )
        );


    col =
        max(
            mix(
                vec3(gray),
                col,
                uSaturation
            ),
            0.0
        );


    float lum =
        max(
            max(
                col.r,
                col.g
            ),
            col.b
        );


    float alpha =
        clamp(
            lum,
            0.0,
            1.0
        )
        *
        uOpacity;


    fragColor =
        vec4(
            col *
            uOpacity,
            alpha
        );
}

`;


// =====================================================
// PALETTE
// =====================================================

function buildPalette(colors) {

    const filled =
        colors && colors.length
        ? colors
        : ["#ffffff"];


    const padded = [];


    for (
        let i = 0;
        i < MAX_COLORS;
        i++
    ) {

        const hex =
            filled[i] ??
            filled[filled.length - 1];


        const c =
            new Color(hex);


        padded.push([
            c.r,
            c.g,
            c.b
        ]);
    }


    return padded;
}


// =====================================================
// STRANDS
// =====================================================

function createStrands(
    container,
    options = {}
) {

    const settings = {

        colors:
            options.colors ??
            [
                "#F97316",
                "#7C3AED",
                "#06B6D4"
            ],

        count:
            options.count ?? 3,

        speed:
            options.speed ?? 0.5,

        amplitude:
            options.amplitude ?? 1,

        waviness:
            options.waviness ?? 1,

        thickness:
            options.thickness ?? 0.7,

        glow:
            options.glow ?? 2.6,

        taper:
            options.taper ?? 3,

        spread:
            options.spread ?? 1,

        hueShift:
            options.hueShift ?? 0,

        intensity:
            options.intensity ?? 0.6,

        opacity:
            options.opacity ?? 1,

        scale:
            options.scale ?? 1.5,

        saturation:
            options.saturation ?? 2
    };


    // -------------------------------------------------
    // Renderer
    // -------------------------------------------------

    const renderer =
        new Renderer({
            alpha: true,
            premultipliedAlpha: true,
            antialias: true
        });


    const gl =
        renderer.gl;


    gl.clearColor(
        0,
        0,
        0,
        0
    );


    gl.enable(
        gl.BLEND
    );


    gl.blendFunc(
        gl.ONE,
        gl.ONE_MINUS_SRC_ALPHA
    );


    gl.canvas.style.backgroundColor =
        "black";


    container.appendChild(
        gl.canvas
    );


    // -------------------------------------------------
    // Geometry
    // -------------------------------------------------

    const geometry =
        new Triangle(gl);


    if (geometry.attributes.uv) {

        delete geometry.attributes.uv;

    }


    // -------------------------------------------------
    // Program
    // -------------------------------------------------

    const program =
        new Program(
            gl,
            {

                vertex:
                    VERT,

                fragment:
                    FRAG,

                uniforms: {

                    uTime: {
                        value: 0
                    },

                    uResolution: {
                        value: [
                            container
                                .offsetWidth,

                            container
                                .offsetHeight
                        ]
                    },

                    uColors: {
                        value:
                            buildPalette(
                                settings.colors
                            )
                    },

                    uColorCount: {
                        value:
                            Math.min(
                                settings.colors.length,
                                MAX_COLORS
                            )
                    },

                    uStrandCount: {
                        value:
                            Math.min(
                                settings.count,
                                MAX_STRANDS
                            )
                    },

                    uSpeed: {
                        value:
                            settings.speed
                    },

                    uAmplitude: {
                        value:
                            settings.amplitude
                    },

                    uWaviness: {
                        value:
                            settings.waviness
                    },

                    uThickness: {
                        value:
                            settings.thickness
                    },

                    uGlow: {
                        value:
                            settings.glow
                    },

                    uTaper: {
                        value:
                            settings.taper
                    },

                    uSpread: {
                        value:
                            settings.spread
                    },

                    uHueShift: {
                        value:
                            settings.hueShift
                    },

                    uIntensity: {
                        value:
                            settings.intensity
                    },

                    uOpacity: {
                        value:
                            settings.opacity
                    },

                    uScale: {
                        value:
                            settings.scale
                    },

                    uSaturation: {
                        value:
                            settings.saturation
                    }

                }
            }
        );


    // -------------------------------------------------
    // Mesh
    // -------------------------------------------------

    const mesh =
        new Mesh(
            gl,
            {
                geometry,
                program
            }
        );


    // -------------------------------------------------
    // Resize
    // -------------------------------------------------

    function resize() {

        const width =
            container.offsetWidth;


        const height =
            container.offsetHeight;


        renderer.setSize(
            width,
            height
        );


        program.uniforms
            .uResolution
            .value = [
                width,
                height
            ];
    }


    resize();


    window.addEventListener(
        "resize",
        resize
    );


    // -------------------------------------------------
    // Animation
    // -------------------------------------------------

    let animationId;


    function update(time) {

        animationId =
            requestAnimationFrame(
                update
            );


        program.uniforms
            .uTime
            .value =
                time * 0.001;


        program.uniforms
            .uColors
            .value =
                buildPalette(
                    settings.colors
                );


        program.uniforms
            .uColorCount
            .value =
                Math.min(
                    settings.colors.length,
                    MAX_COLORS
                );


        program.uniforms
            .uStrandCount
            .value =
                Math.min(
                    Math.max(
                        Math.round(
                            settings.count
                        ),
                        1
                    ),
                    MAX_STRANDS
                );


        program.uniforms
            .uSpeed
            .value =
                settings.speed;


        program.uniforms
            .uAmplitude
            .value =
                settings.amplitude;


        program.uniforms
            .uWaviness
            .value =
                settings.waviness;


        program.uniforms
            .uThickness
            .value =
                settings.thickness;


        program.uniforms
            .uGlow
            .value =
                settings.glow;


        program.uniforms
            .uTaper
            .value =
                settings.taper;


        program.uniforms
            .uSpread
            .value =
                settings.spread;


        program.uniforms
            .uHueShift
            .value =
                settings.hueShift;


        program.uniforms
            .uIntensity
            .value =
                settings.intensity;


        program.uniforms
            .uOpacity
            .value =
                settings.opacity;


        program.uniforms
            .uScale
            .value =
                settings.scale;


        program.uniforms
            .uSaturation
            .value =
                settings.saturation;


        renderer.render({
            scene: mesh
        });

    }


    animationId =
        requestAnimationFrame(
            update
        );
}


// =====================================================
// START
// =====================================================

const container =
    document.getElementById(
        "strands-container"
    );


createStrands(
    container,
    {

        colors: [
            "#F97316",
            "#7C3AED",
            "#06B6D4"
        ],

        count: 3,

        speed: 0.5,

        amplitude: 1,

        waviness: 1,

        thickness: 0.7,

        glow: 2.6,

        taper: 3,

        spread: 1,

        intensity: 0.6,

        saturation: 2,

        opacity: 1,

        scale: 1.5,

        hueShift: 0

    }
);

</script>

</body>
</html>
"""


components.html(
    html,
    height=600,
    scrolling=False
)