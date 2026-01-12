
const computeShaderCode = (LR) => {return `struct Neuron{
    state: f32,
    fired: f32,
    connections: array<f32, 8>,
}

@group(0) @binding(0) var<uniform> grid: vec2f;

@group(0) @binding(1) var<storage> networkIn: array<Neuron>;
@group(0) @binding(2) var<storage, read_write> networkOut: array<Neuron>;

fn cellIndex(cell: vec2u) -> u32 {
    return (cell.y % (u32(grid.y)) * u32(grid.x)) + (cell.x % u32(grid.x));
}

fn cellActive(x: u32, y: u32) -> Neuron {
    return networkIn[cellIndex(vec2(x, y))];
}

// fn getConnection(cell: vec2u, n: i32) -> u32 {
//     switch (n){
//         case 0: {
//             if(cell.x-1 >= 0 && cell.y-1 >= 0 && cell.x-1 < u32(grid.x) && cell.y-1 < u32(grid.y)){
//                 return cellIndex(vec2u(cell.x-1, cell.y-1));
//             }
//         }
//         default: {
//             return 4294967295;
//         }
//     }
//     return 4294967295;
// }

fn getConnection(n: i32) -> vec2f {
    switch (n){
        case 0: {return vec2f(-1,-1);}
        case 1: {return vec2f(0,-1);}
        case 2: {return vec2f(1,-1);}
        case 3: {return vec2f(1,0);}
        case 4: {return vec2f(1,1);}
        case 5: {return vec2f(0,1);}
        case 6: {return vec2f(-1,1);}
        case 7: {return vec2f(-1,0);}
        default: {return vec2f(0,0);}
    }
}

fn updateNeuron(index: u32){
    const sDecay = f32(0.976);
    const fDecay = f32(0.9);
    const activation = f32(0.2);

    networkOut[index].fired = networkIn[index].fired * fDecay;

    if(networkIn[index].state > activation){
        networkOut[index].state = 1;
        networkOut[index].fired = 1;
    }else{
        networkOut[index].state = networkIn[index].state * sDecay;
    }

    if(networkIn[index].state >= f32(1)){
        networkOut[index].state = -1;
    }
}

@compute @workgroup_size(8, 8)
fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
    let i = cellIndex(cell.xy);
    // update node
    updateNeuron(i);

    networkOut[i].connections = networkIn[i].connections;


    for(var j = 0; j < 8; j++){
        let con = getConnection(j);
        let cord = vec2f(cell.xy)+con;
        if(cord.x >= 0 && cord.y >= 0 && cord.x < grid.x && cord.y < grid.y){
            if(networkIn[i].state == 1){
                let connection = cellIndex(vec2u(cord));
                let w = ((j+3)%8); // weight of other node
                networkOut[connection].state += networkIn[i].connections[j];

                // train
                const MAXWEIGHT = 0.1;
                const tmax = 1; // max training function value
                const tmin = -0.2; // min training function value
                const LR = ${LR};
                // const rThreshold = 10.5; // rienforcement threshold
                // const fDecay = Math.pow(-tmin/(tmax-tmin), 1/rThreshold);
                networkOut[connection].connections[w] += ((networkIn[i].fired*(tmax-tmin)+tmin)) * (MAXWEIGHT-(networkIn[connection].connections[w])) * LR;
            }
        }

    }

    // let con = vec2f(1,-1);
    // let cord = vec2f(cell.xy);
    // networkOut[i].connections[0] = f32((cord.x+1)/grid.x);
    // networkOut[i].connections[1] = f32((cord.y+1)/grid.y);

}`}

const renderShaderCode = `
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
    output.color = vec4f(0, node.fired, -min(node.state, 0), 1);
    return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    return input.color;
}`

export default async function createBackground(canvas, width, LR){
    if (!navigator.gpu) throw Error("WebGPU not supported.");

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw Error("Couldn’t request WebGPU adapter.");

    const device = await adapter.requestDevice();
    if (!device) throw Error("Couldn’t request WebGPU logical device.");

    const context = canvas.getContext("webgpu");
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

    context.configure({
    device: device,
    format: canvasFormat,
    });

    // const GRID_SIZE = size??256;//256;
    const GRID_WIDTH = Math.min(width, 512);
    const GRID_HEIGHT = GRID_WIDTH;

    const UPDATE_INTERVAL = 1000/100;
    const WORKGROUP_SIZE = 8; // must be a factor of grid size
    const BUFFER_SIZE = GRID_WIDTH * GRID_HEIGHT * 10;
    let step = 0;

    const random = function(){return (Math.random()*2)-1}

    const vertices = new Float32Array([
        -1, -1,
        1, -1,
        1, 1,

        -1, -1,
        1, 1,
        -1, 1,
    ]);

    const vertexBuffer = device.createBuffer({
        label: "Cell vertices",
        size: vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(vertexBuffer, 0, vertices);

    const vertexBufferLayout = {
        arrayStride: 8,
        attributes: [{
            format: "float32x2",
            offset: 0,
            shaderLocation: 0, // Position. Matches @location(0) in the @vertex shader.
        }],
    };

    // Create the bind group layout and pipeline layout.
    const bindGroupLayout = device.createBindGroupLayout({
        label: "Cell Bind Group Layout",
        entries: [{
            binding: 0,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
            buffer: {} // Grid uniform buffer
        }, {
            binding: 1,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
            buffer: { type: "read-only-storage" } // Cell state input buffer
        }, {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage" } // Cell state output buffer
        }]
    });

    const pipelineLayout = device.createPipelineLayout({
        label: "Cell Pipeline Layout",
        bindGroupLayouts: [bindGroupLayout],
    });

    // Create the shader that will render the cells.
    const cellShaderModule = device.createShaderModule({
        label: "Cell shader",
        code: renderShaderCode
    });

    // Create a pipeline that renders the cell.
    const cellPipeline = device.createRenderPipeline({
        label: "Cell pipeline",
        layout: pipelineLayout,
        vertex: {
            module: cellShaderModule,
            entryPoint: "vertexMain",
            buffers: [vertexBufferLayout]
        },
        fragment: {
            module: cellShaderModule,
            entryPoint: "fragmentMain",
            targets: [{
                format: canvasFormat
            }]
        }
    });

    // Create the compute shader that will process the game of life simulation.
    const simulationShaderModule = device.createShaderModule({
        label: "Life simulation shader",
        code: computeShaderCode(LR??0.6)
    });

    // Create a compute pipeline that updates the game state.
    const simulationPipeline = device.createComputePipeline({
        label: "Simulation pipeline",
        layout: pipelineLayout,
        compute: {
            module: simulationShaderModule,
            entryPoint: "computeMain",
        }
    });

    // Create a uniform buffer that describes the grid.
    const uniformArray = new Float32Array([GRID_WIDTH, GRID_HEIGHT]);
    const uniformBuffer = device.createBuffer({
        label: "Grid Uniforms",
        size: uniformArray.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

    // 10 cell variables
    const networkBuffer = new Float32Array(BUFFER_SIZE);

    // Create two storage buffers to hold the cell state.
    const networkBufferStorage = [
        device.createBuffer({
            label: "Cell State A",
            size: networkBuffer.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        }),
        device.createBuffer({
            label: "Cell State B",
            size: networkBuffer.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        })
    ];

    const resultsBuffer = device.createBuffer({
        size: networkBuffer.byteLength,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // Set each cell to a random state, then copy the JavaScript array into
    // the storage buffer.
    for (let i = 0; i < networkBuffer.length; ++i) {
        if(i % 10 > 1){
            networkBuffer[i] = random() * 0.9
        }
    }
    // networkBuffer[10111] = 1;

    device.queue.writeBuffer(networkBufferStorage[0], 0, networkBuffer);

    // Create a bind group to pass the grid uniforms into the pipeline
    const bindGroups = [
        device.createBindGroup({
            label: "Cell renderer bind group A",
            layout: bindGroupLayout,
            entries: [{
                binding: 0,
                resource: { buffer: uniformBuffer }
            }, {
                binding: 1,
                resource: { buffer: networkBufferStorage[0] }
            }, {
                binding: 2,
                resource: { buffer: networkBufferStorage[1] }
            }],
        }),
        device.createBindGroup({
            label: "Cell renderer bind group B",
            layout: bindGroupLayout,
            entries: [{
                binding: 0,
                resource: { buffer: uniformBuffer }
            }, {
                binding: 1,
                resource: { buffer: networkBufferStorage[1] }
            }, {
                binding: 2,
                resource: { buffer: networkBufferStorage[0] }
            }],
        }),
    ];

    async function activate(x, y, value){

        const cell = (x + y * GRID_WIDTH) * 10

        await resultsBuffer.mapAsync(
            GPUMapMode.READ,
            0, // Offset
            BUFFER_SIZE*4, // Length
        );
        const copyArrayBuffer = resultsBuffer.getMappedRange(4*cell, 4*10);
        const data = new Float32Array(copyArrayBuffer.slice(0));
        resultsBuffer.unmap();

        data[0] = value;

        device.queue.writeBuffer(networkBufferStorage[step % 2], 4*cell, data);

    }

    async function readNetwork(){
        // read
        await resultsBuffer.mapAsync(
            GPUMapMode.READ,
            0, // Offset
            BUFFER_SIZE*4, // Length
        );
        const copyArrayBuffer = await resultsBuffer.getMappedRange(0, BUFFER_SIZE*4);
        const data = new Float32Array(copyArrayBuffer.slice(0));
        await resultsBuffer.unmap();
        return data;
    }

    function positionToIndex(x, y){
        return (x + y * GRID_WIDTH) * 10;
    }

    canvas.addEventListener('mousedown', async (event)=>{
        var rect = event.target.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = rect.bottom-event.clientY;
        await activate(Math.floor((x/canvas.width)*GRID_WIDTH), Math.floor((y/canvas.height)*GRID_HEIGHT), 1)
    })

    function update() {
        const encoder = device.createCommandEncoder();

        // Start a compute pass
        const computePass = encoder.beginComputePass();

        computePass.setPipeline(simulationPipeline);
        computePass.setBindGroup(0, bindGroups[step % 2]);
        const workgroupCount = Math.ceil(GRID_WIDTH / WORKGROUP_SIZE);
        computePass.dispatchWorkgroups(workgroupCount, workgroupCount);
        computePass.end();

        encoder.copyBufferToBuffer(
            networkBufferStorage[step % 2],
            0, // Source offset
            resultsBuffer,
            0, // Destination offset
            BUFFER_SIZE*4,
        );

        step++; // Increment the step count

        // Start a render pass
        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                loadOp: "clear",
                clearValue: { r: 0, g: 0, b: 0.0, a: 1.0 },
                storeOp: "store",
            }]
        });

        // Draw the grid.
        pass.setPipeline(cellPipeline);
        pass.setBindGroup(0, bindGroups[step % 2]); // Updated!
        pass.setVertexBuffer(0, vertexBuffer);
        pass.draw(vertices.length / 2, GRID_WIDTH * GRID_HEIGHT);

        // End the render pass and submit the command buffer
        pass.end();
        device.queue.submit([encoder.finish()]);
    }

    return {update, readNetwork, positionToIndex, activate};
}