// Compilation script for VoltGrad Vulkan compute shaders
const fs = require('fs');
const path = require('path');

const files = [
    'matmul_forward',
    'matmul_backward_dx',
    'matmul_backward_dw',
    'matmul_batched_forward',
    'matmul_batched_backward_dx',
    'matmul_batched_backward_dw',
    'relu_forward',
    'relu_backward',
    'loss_mse',
    'optimizer_step',
    'repack_dbw'
];

async function main() {
    let glslang;
    try {
        const glslangModule = require('@webgpu/glslang');
        glslang = await glslangModule();
    } catch (e) {
        console.error('Error: @webgpu/glslang is required to compile shaders.');
        console.error('Run: npm install @webgpu/glslang');
        process.exit(1);
    }

    for (const name of files) {
        const compPath = path.join(__dirname, `${name}.comp`);
        const spvPath = path.join(__dirname, `${name}.spv`);
        try {
            console.log(`Compiling ${compPath}...`);
            const src = fs.readFileSync(compPath, 'utf8');
            const spv = glslang.compileGLSL(src, 'compute');
            if (!spv || spv.length === 0) {
                throw new Error(`Failed to generate SPIR-V bytecode for ${name}.comp`);
            }
            fs.writeFileSync(spvPath, Buffer.from(spv.buffer, spv.byteOffset, spv.byteLength));
            console.log(`Saved -> ${spvPath} (${spv.length * 4} bytes)`);
        } catch (compileErr) {
            console.error(`Compilation error in ${name}.comp:`, compileErr);
            process.exit(1);
        }
    }
    console.log('All shaders compiled successfully!');
}

main().catch(err => {
    console.error('Fatal shader build error:', err);
    process.exit(1);
});
