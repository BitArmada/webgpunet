struct Neuron{
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
                const LR = 0.6;
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

}