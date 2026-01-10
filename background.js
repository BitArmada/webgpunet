
export default async function createBackground(canvas, width){
    if (!navigator.gpu) throw Error("WebGPU not supported.");

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw Error("Couldn’t request WebGPU adapter.");

    const device = await adapter.requestDevice();
    if (!device) throw Error("Couldn’t request WebGPU logical device.");

    const renderShaderCode = await loadFile('./renderBackground.wgsl');
    const computeShaderCode = await loadFile('./compute.wgsl');

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
        code: computeShaderCode
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

        //activate
        // data[cell] = value;

        // write
        // device.queue.writeBuffer(networkBufferStorage[step % 2], 0, data);

        // read
        // await resultsBuffer.mapAsync(
        //     GPUMapMode.READ,
        //     0, // Offset
        //     BUFFER_SIZE*4, // Length
        // );
        // const copyArrayBuffer = resultsBuffer.getMappedRange(0, BUFFER_SIZE*4);
        // const data = new Float32Array(copyArrayBuffer.slice(0));
        // resultsBuffer.unmap();

        // //activate
        // data[cell] = value;

        // // write
        // device.queue.writeBuffer(networkBufferStorage[step % 2], 0, data);


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

    //event stuff
    const inputKeys = ' abcdefghijklmnopqrstuvwxyz1234567890'
    var input = {}
    // generate input
    for(var i = 0; i < inputKeys.length; i++){
        const key = inputKeys[i];
        input[key] = {
            x: Math.floor((i / inputKeys.length) * GRID_WIDTH),
            y: 10,
            value: 0
        }
    }

    document.addEventListener('keydown', async (event)=>{
        input[event.key].value = 1;
        // update inputs
        // await updateInput()
        await activate(5,5,1)
        await activate(5,10,1)
        await activate(10,7, -1)
    })
    document.addEventListener('keyup', async (event)=>{
        input[event.key].value = 0;
        // update inputs
        // await updateInput()
    })

    canvas.addEventListener('mousedown', async (event)=>{
        var rect = event.target.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = rect.bottom-event.clientY;
        await activate(Math.floor((x/canvas.width)*GRID_WIDTH), Math.floor((y/canvas.height)*GRID_HEIGHT), 1)
    })

    async function updateInput(){
        var data = await readNetwork();
        for (i in input){
            data[positionToIndex(input[i].x, input[i].y)] = input[i].value;
        }

        await device.queue.writeBuffer(networkBufferStorage[step % 2], 0, data);
    }

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

    setInterval(update, UPDATE_INTERVAL);

    async function loadFile(src){
        const response = await fetch(src);
        const data = await response.text();
        return data;
    }
}