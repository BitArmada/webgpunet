
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
    output.color = vec4f(0,node.fired, -min(node.state, 0), 1);
    // output.color = vec4f(max(node.state, 0), node.fired, -min(node.state, 0), 1);
    // output.color = vec4f(node.connections[0], node.connections[1], node.connections[2], 1);
    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    return input.color;
}