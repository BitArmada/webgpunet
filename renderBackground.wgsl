
struct Neuron{
    state: f32,
    fired: f32,
    connections: array<f32, 8>,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) cell: vec2f,
    @location(1) color: vec4f,
};

@group(0) @binding(0) var<uniform> grid: vec2f;
@group(0) @binding(1) var<storage> network: array<Neuron>;

fn inferno(x: f32) -> vec3<f32> {
    let t = clamp(x, 0.0, 1.0);

    let r = 0.0
        + 0.4727 * t
        + 2.5179 * t*t
        - 2.4596 * t*t*t
        + 1.3331 * t*t*t*t;

    let g = 0.0
        + 0.0280 * t
        + 3.3434 * t*t
        - 4.1740 * t*t*t
        + 2.1532 * t*t*t*t;

    let b = 0.0
        + 0.5855 * t
        + 0.1407 * t*t
        - 2.4060 * t*t*t
        + 2.1165 * t*t*t*t;

    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn inferno_balanced(x: f32) -> vec3<f32> {
    // Clamp input
    var t = clamp(x, 0.0, 1.0);

    // Perceptual redistribution
    // (gamma + soft log curve)
    t = pow(t, 0.75);
    t = log(1.0 + 6.0 * t) / log(7.0);

    // Inferno polynomial
    let r =
          0.4727 * t
        + 2.5179 * t*t
        - 2.4596 * t*t*t
        + 1.3331 * t*t*t*t;

    let g =
          0.0280 * t
        + 3.3434 * t*t
        - 4.1740 * t*t*t
        + 2.1532 * t*t*t*t;

    let b =
          0.5855 * t
        + 0.1407 * t*t
        - 2.4060 * t*t*t
        + 2.1165 * t*t*t*t;

    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn plasma(x: f32) -> vec3<f32> {
    let t = clamp(x, 0.0, 1.0);

    let r = 0.050 + 2.40*t - 2.20*t*t + 0.75*t*t*t;
    let g = 0.030 + 1.10*t + 2.00*t*t - 2.00*t*t*t;
    let b = 0.500 + 1.60*t - 2.80*t*t + 1.20*t*t*t;

    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn flame_colormap(x: f32) -> vec3<f32> {
    // Clamp and redistribute
    var t = clamp(x, 0.0, 1.0);

    // Strong low-end preservation, compressed highlights
    t = pow(t, 0.65);
    t = log(1.0 + 8.0 * t) / log(9.0);

    // --- Fire body (red-dominant inferno) ---
    let r =
          0.60 * t
        + 2.80 * t*t
        - 2.30 * t*t*t
        + 1.20 * t*t*t*t;

    let g =
          0.05 * t
        + 2.10 * t*t
        - 2.60 * t*t*t
        + 1.40 * t*t*t*t;

    let b =
          0.02 * t
        + 0.50 * t*t
        - 1.10 * t*t*t
        + 0.80 * t*t*t*t;

    var color = vec3<f32>(r, g, b);

    // --- Blue ionized flame tip ---
    // Activates only very near the top
    let tip = smoothstep(0.90, 1.0, t);
    let blue_tip = vec3<f32>(0.4, 0.6, 1.0);

    color = mix(color, blue_tip, tip * 0.6);

    // Fade from pure black
    color *= smoothstep(0.02, 0.15, t);

    return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn neon_colormap(x: f32) -> vec3<f32> {
    var t = clamp(x, 0.0, 1.0);

    // Perceptual redistribution so we don't lose low-end detail
    t = pow(t, 0.8);
    t = log(1.0 + 10.0 * t) / log(11.0);

    // Hue sweep (purple → blue → cyan → green → yellow → red)
    let r = 0.6 + 0.4 * sin(6.28318 * (t + 0.00));
    let g = 0.6 + 0.4 * sin(6.28318 * (t + 0.33));
    let b = 0.6 + 0.4 * sin(6.28318 * (t + 0.66));

    var color = vec3<f32>(r, g, b);

    // Fade from black
    color *= smoothstep(0.02, 0.15, t);

    // Push toward white at the top (emissive)
    let glow = smoothstep(0.75, 1.0, t);
    color = mix(color, vec3<f32>(1.0), glow);

    return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
}
fn neon_energy(x: f32) -> vec3<f32> {
    var t = clamp(x, 0.0, 1.0);

    // Single perceptual remap
    // >1 compresses low values, <1 expands them
    t = pow(t, 1.6);

    // Hue rotation (purple → blue → cyan → green → yellow → red)
    let r = 0.6 + 0.4 * sin(6.28318 * (t + 0.00));
    let g = 0.6 + 0.4 * sin(6.28318 * (t + 0.33));
    let b = 0.6 + 0.4 * sin(6.28318 * (t + 0.66));

    var color = vec3<f32>(r, g, b);

    // Fade in from black
    color *= smoothstep(0.0, 0.25, t);

    // Push to white only near the very top
    let hot = smoothstep(0.80, 1.0, t);
    color = mix(color, vec3<f32>(1.0), hot);

    return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
}


@vertex
fn vertexMain(@location(0) position: vec2f, @builtin(instance_index) instance: u32) -> VertexOutput {
    var output: VertexOutput;

    let node = network[instance];

    let i = f32(instance);
    let cell = vec2f(i % grid.x, floor(i / grid.x));

    let scale = f32(1);//f32(cellState[instance]);
    let cellOffset = cell / grid * 2;
    let gridPos = (position*scale+1) / grid - 1 + cellOffset;

    output.position = vec4f(gridPos, 0, 1);
    output.cell = cell / grid;

    var connectivity = 0f;
    for(var j = 0; j < 8; j++){
        connectivity+= node.connections[j];
    }
    connectivity/=8;
    // output.color = vec4f(node.fired,0, 0, 1);
    // output.color = vec4f(neon_energy(-min((node.state), 0)), 1);
    output.color = vec4f(0, node.fired, -min(node.state, 0), 1);
    // output.color = vec4f(flame_colormap(node.fired)+vec3(0,0,-min(node.state, 0)), 1);
    // output.color = vec4f(flame_colormap(node.fired)+vec3(0,-min(node.state, 0),0), 1);
    // output.color = vec4f(max(node.state, 0), node.fired, -min(node.state, 0), 1);
    // output.color = vec4f(node.connections[0], node.connections[1], node.connections[2], 1);
    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    return input.color;
}