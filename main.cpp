
#include <SDL3/SDL.h>
#include <SDL3/SDL_video.h>
#include <SDL3/SDL_gpu.h>

#include "imgui_sdl3_handler.hpp"

#include <imgui.h>
#include <string>
#include <variant>
#include <vector>

#include <d3dcompiler.h>

static const std::string s_imguiVertexShader = R"(
cbuffer vertexBuffer : register(b0)
{
	float4x4 ProjectionMatrix;
};

struct VS_INPUT
{
	float2 pos : TEXCOORD0;
	float2 uv  : TEXCOORD1;
	uint col : TEXCOORD2;
};

struct PS_INPUT
{
	float4 pos : SV_POSITION;
	float4 col : COLOR0;
	float2 uv  : TEXCOORD0;
};

PS_INPUT main(VS_INPUT input)
{
	PS_INPUT output;
	output.pos = mul( ProjectionMatrix, float4(input.pos.xy, 0.f, 1.f));
	output.col.x = float(( input.col >> 0 ) & 0xFF) / 255.0;
	output.col.y = float(( input.col >> 8 ) & 0xFF) / 255.0;
	output.col.z = float(( input.col >> 16 ) & 0xFF) / 255.0;
	output.col.w = float(( input.col >> 24 ) & 0xFF) / 255.0;
	output.uv  = input.uv;
	return output;
} )";

static const std::string s_imguiFragmentShader = R"(
struct PS_INPUT
{
	float4 pos : SV_POSITION;
	float4 col : COLOR0;
	float2 uv  : TEXCOORD0;
};

sampler sampler0;
Texture2D texture0;

float4 main(PS_INPUT input) : SV_Target
{
	float4 out_col = input.col * texture0.Sample(sampler0, input.uv);
	return out_col;
} )";

std::vector<uint8_t> CompileD3DShader(::SDL_GPUShaderStage stage, const std::string& shader)
{
    ::ID3DBlob* shaderBlob = nullptr;
    D3DCompile(shader.c_str(), shader.size(), nullptr, nullptr, nullptr, "main", stage == SDL_GPU_SHADERSTAGE_VERTEX ? "vs_5_0" : "ps_5_0", 0, 0, &shaderBlob, nullptr);

    std::vector<uint8_t> result;
    result.resize(shaderBlob->GetBufferSize());

    std::memcpy(result.data(), shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize());

    shaderBlob->Release();

    return result;
}

struct ImGuiRenderPass
{
public:
    void Initialize(::SDL_GPUDevice* device, ::SDL_Window* window)
    {
        auto vertexShaderBlob = CompileD3DShader(SDL_GPU_SHADERSTAGE_VERTEX, s_imguiVertexShader);

        auto vertexShaderDesc = SDL_GPUShaderCreateInfo{};
        vertexShaderDesc.code = (const uint8_t*)vertexShaderBlob.data();
        vertexShaderDesc.codeSize = vertexShaderBlob.size();
        vertexShaderDesc.entryPointName = "main";
        vertexShaderDesc.format = SDL_GPU_SHADERFORMAT_DXBC;
        vertexShaderDesc.stage = SDL_GPU_SHADERSTAGE_VERTEX;
        vertexShaderDesc.samplerCount = 0;
        vertexShaderDesc.uniformBufferCount = 1;
        vertexShaderDesc.storageBufferCount = 0;
        vertexShaderDesc.storageTextureCount = 0;

        auto vertexShader = ::SDL_CreateGPUShader(device, &vertexShaderDesc);

        auto fragmentShaderBlob = CompileD3DShader(SDL_GPU_SHADERSTAGE_FRAGMENT, s_imguiFragmentShader);

        auto fragmentShaderDesc = SDL_GPUShaderCreateInfo{};
        fragmentShaderDesc.code = (const uint8_t*)fragmentShaderBlob.data();
        fragmentShaderDesc.codeSize = fragmentShaderBlob.size();
        fragmentShaderDesc.entryPointName = "main";
        fragmentShaderDesc.format = SDL_GPU_SHADERFORMAT_DXBC;
        fragmentShaderDesc.stage = SDL_GPU_SHADERSTAGE_FRAGMENT;
        fragmentShaderDesc.samplerCount = 1;
        fragmentShaderDesc.uniformBufferCount = 0;
        fragmentShaderDesc.storageBufferCount = 0;
        fragmentShaderDesc.storageTextureCount = 0;

        auto fragmentShader = ::SDL_CreateGPUShader(device, &fragmentShaderDesc);

        auto attachmentDesc = SDL_GPUColorAttachmentDescription{};
        attachmentDesc.format = ::SDL_GetGPUSwapchainTextureFormat(device, window);
        attachmentDesc.blendState = {
            .blendEnable = SDL_TRUE,
            .srcColorBlendFactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA,
            .dstColorBlendFactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
            .colorBlendOp = SDL_GPU_BLENDOP_ADD,
            .srcAlphaBlendFactor = SDL_GPU_BLENDFACTOR_ONE,
            .dstAlphaBlendFactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
            .alphaBlendOp = SDL_GPU_BLENDOP_ADD,
            .colorWriteMask = 0xF,
        };

        // Create the pipelines
        auto pipelineDesc = SDL_GPUGraphicsPipelineCreateInfo{};
        pipelineDesc.attachmentInfo = {
            .colorAttachmentDescriptions = &attachmentDesc,
            .colorAttachmentCount = 1,
            .hasDepthStencilAttachment = SDL_TRUE,
            .depthStencilFormat = SDL_GPU_TEXTUREFORMAT_D24_UNORM_S8_UINT
        };

        pipelineDesc.depthStencilState.depthTestEnable = SDL_TRUE;
        pipelineDesc.depthStencilState.depthWriteEnable = SDL_FALSE;
        pipelineDesc.depthStencilState.compareOp = SDL_GPU_COMPAREOP_GREATER_OR_EQUAL;

        auto vertexBindingsDesc = ::SDL_GPUVertexBinding{
            .binding = 0,
            .stride = sizeof(ImDrawVert),
            .inputRate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
            .instanceStepRate = 0,
        };

        SDL_GPUVertexAttribute vertexAttributesDesc[] = {
            ::SDL_GPUVertexAttribute{
                .location = 0,
                .binding = 0,
                .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2,
                .offset = 0
            },
            ::SDL_GPUVertexAttribute{
                .location = 1,
                .binding = 0,
                .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2,
                .offset = sizeof(float) * 2,
            },
            ::SDL_GPUVertexAttribute{
                .location = 2,
                .binding = 0,
                .format = SDL_GPU_VERTEXELEMENTFORMAT_UINT,
                .offset = sizeof(float) * 4,
            },

        };

        pipelineDesc.vertexInputState = {
            .vertexBindings = &vertexBindingsDesc,
            .vertexBindingCount = 1,
            .vertexAttributes = vertexAttributesDesc,
            .vertexAttributeCount = 3,
        };

        pipelineDesc.rasterizerState = ::SDL_GPURasterizerState{
            .fillMode = SDL_GPU_FILLMODE_FILL,
            .cullMode = SDL_GPU_CULLMODE_NONE,
            .frontFace = {},
            .depthBiasEnable = SDL_FALSE,
            .depthBiasConstantFactor = {},
            .depthBiasClamp = {},
            .depthBiasSlopeFactor = {},
        };

        pipelineDesc.primitiveType = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
        pipelineDesc.multisampleState.sampleMask = 0xFFFF;
        pipelineDesc.vertexShader = vertexShader;
        pipelineDesc.fragmentShader = fragmentShader;

        m_pipeline = ::SDL_CreateGPUGraphicsPipeline(device, &pipelineDesc);

        // Clean up shader resources
        ::SDL_ReleaseGPUShader(device, vertexShader);
        ::SDL_ReleaseGPUShader(device, fragmentShader);

        auto vertexBufferDesc = ::SDL_GPUBufferCreateInfo{
            .usageFlags = ::SDL_GPU_BUFFERUSAGE_VERTEX_BIT,
            .sizeInBytes = sizeof(ImDrawVert) * 1024 * 1024 * 64
        };

        m_vertexBuffer = ::SDL_CreateGPUBuffer(device, &vertexBufferDesc);
        ::SDL_SetGPUBufferName(device, m_vertexBuffer, "ImGui - VertexBuffer");

        auto vertexTransferBufferDesc = ::SDL_GPUTransferBufferCreateInfo{
            .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
            .sizeInBytes = vertexBufferDesc.sizeInBytes
        };

        m_vertexTransferBuffer = ::SDL_CreateGPUTransferBuffer(device, &vertexTransferBufferDesc);

        auto indexBufferDesc = ::SDL_GPUBufferCreateInfo{
            .usageFlags = ::SDL_GPU_BUFFERUSAGE_INDEX_BIT,
            .sizeInBytes = sizeof(ImDrawIdx) * 1024 * 1024 * 64
        };

        m_indexBuffer = ::SDL_CreateGPUBuffer(device, &indexBufferDesc);
        ::SDL_SetGPUBufferName(device, m_indexBuffer, "ImGui - IndexBuffer");

        auto indexTransferBufferDesc = ::SDL_GPUTransferBufferCreateInfo{
            .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
            .sizeInBytes = indexBufferDesc.sizeInBytes
        };

        m_indexTransferBuffer = ::SDL_CreateGPUTransferBuffer(device, &indexTransferBufferDesc);

        auto samplerDesc = ::SDL_GPUSamplerCreateInfo{
            .minFilter = SDL_GPU_FILTER_NEAREST,
            .magFilter = SDL_GPU_FILTER_NEAREST,
            .mipmapMode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST,
            .addressModeU = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            .addressModeV = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            .addressModeW = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        };

        m_textureSampler = SDL_CreateGPUSampler(device, &samplerDesc);

        unsigned char* pixels;
        int width, height;
        ImGui::GetIO().Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

        auto pixelDataSizeInBytes = (uint32_t)(width * height * sizeof(uint8_t) * 4);

        auto textureDesc = ::SDL_GPUTextureCreateInfo{
            .type = SDL_GPU_TEXTURETYPE_2D,
            .format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
            .usageFlags = SDL_GPU_TEXTUREUSAGE_SAMPLER_BIT,
            .width = (uint32_t)width,
            .height = (uint32_t)height,
            .layerCountOrDepth = 1,
            .levelCount = 1,
            .sampleCount = {},
        };

        m_fontTexture = ::SDL_CreateGPUTexture(device, &textureDesc);
        ::SDL_SetGPUTextureName(device, m_fontTexture, "ImGui - FontAtas");

        auto transferDestDesc = ::SDL_GPUTextureRegion{};
        transferDestDesc.texture = m_fontTexture;
        transferDestDesc.w = (uint32_t)width;
        transferDestDesc.h = (uint32_t)height;
        transferDestDesc.d = 1;

        auto transferBufferDesc = ::SDL_GPUTransferBufferCreateInfo{
            .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
            .sizeInBytes = pixelDataSizeInBytes
        };

        auto* transferBuffer = ::SDL_CreateGPUTransferBuffer(device, &transferBufferDesc);

        void* data = ::SDL_MapGPUTransferBuffer(device, transferBuffer, SDL_FALSE);
        std::memcpy(data, pixels, pixelDataSizeInBytes);
        ::SDL_UnmapGPUTransferBuffer(device, transferBuffer);

        // Upload the transfer data to the vertex buffer
        auto uploadBuffer = ::SDL_AcquireGPUCommandBuffer(device);
        {
            auto* pass = ::SDL_BeginGPUCopyPass(uploadBuffer);
            {
                auto transferSourceDesc = ::SDL_GPUTextureTransferInfo{ transferBuffer };
                ::SDL_UploadToGPUTexture(pass, &transferSourceDesc, &transferDestDesc, SDL_FALSE);
            }
            ::SDL_EndGPUCopyPass(pass);
        }

        ::SDL_SubmitGPU(uploadBuffer);

        ::SDL_ReleaseGPUTransferBuffer(device, transferBuffer);

        auto& io = ImGui::GetIO();
        io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
        io.Fonts->SetTexID(m_fontTexture);
    }

    void UpdateBuffers(::SDL_GPUDevice* device, ::SDL_GPUCommandBuffer* commandBuffer, const ::ImDrawData* drawData)
    {
        if (drawData->TotalVtxCount == 0)
            return;

        //! #TODO: verify if we have enough space in transfer buffer for new payload!
        //! #TODO: support for resizing buffers when we are out of space!

        uint8_t* vertices = (uint8_t*)::SDL_MapGPUTransferBuffer(device, m_vertexTransferBuffer, SDL_TRUE);
        uint8_t* indices = (uint8_t*)::SDL_MapGPUTransferBuffer(device, m_indexTransferBuffer, SDL_TRUE);

        for (int n = 0; n < drawData->CmdListsCount; n++)
        {
            auto* cmdList = drawData->CmdLists[n];

            auto vertexDataSizeInBytes = cmdList->VtxBuffer.Size * sizeof(ImDrawVert);
            std::memcpy(vertices, cmdList->VtxBuffer.Data, vertexDataSizeInBytes);

            vertices += vertexDataSizeInBytes;

            auto indexDataSizeInBytes = cmdList->IdxBuffer.Size * sizeof(ImDrawIdx);
            std::memcpy(indices, cmdList->IdxBuffer.Data, indexDataSizeInBytes);

            indices += indexDataSizeInBytes;
        }

        ::SDL_UnmapGPUTransferBuffer(device, m_vertexTransferBuffer);
        ::SDL_UnmapGPUTransferBuffer(device, m_indexTransferBuffer);

        auto* pass = ::SDL_BeginGPUCopyPass(commandBuffer);
        {
            auto transferSourceDesc = ::SDL_GPUTransferBufferLocation{
                .transferBuffer = m_vertexTransferBuffer
            };

            auto transferDestDesc = ::SDL_GPUBufferRegion{
                .buffer = m_vertexBuffer,
                .offset = 0,
                .size = (uint32_t)(drawData->TotalVtxCount * sizeof(ImDrawVert))
            };

            ::SDL_UploadToGPUBuffer(pass, &transferSourceDesc, &transferDestDesc, SDL_TRUE);
        }
        {
            auto transferSourceDesc = ::SDL_GPUTransferBufferLocation{
                .transferBuffer = m_indexTransferBuffer
            };

            auto transferDestDesc = ::SDL_GPUBufferRegion{
                .buffer = m_indexBuffer,
                .offset = 0,
                .size = (uint32_t)(drawData->TotalIdxCount * sizeof(ImDrawIdx))
            };

            ::SDL_UploadToGPUBuffer(pass, &transferSourceDesc, &transferDestDesc, SDL_FALSE);
        }

        ::SDL_EndGPUCopyPass(pass);
    }

    void Render(::SDL_GPUCommandBuffer* commandBuffer, ::SDL_GPURenderPass* renderPass, const ::ImDrawData* drawData)
    {
        ::SDL_BindGPUGraphicsPipeline(renderPass, m_pipeline);

        auto viewportDesc = ::SDL_GPUViewport{
           .x = 0.0f,
           .y = 0.0f,
           .w = drawData->DisplaySize.x,
           .h = drawData->DisplaySize.y,
           .minDepth = 0.0f,
           .maxDepth = 1.0f,
        };

        ::SDL_SetGPUViewport(renderPass, &viewportDesc);

        float L = drawData->DisplayPos.x;
        float R = drawData->DisplayPos.x + drawData->DisplaySize.x;
        float T = drawData->DisplayPos.y;
        float B = drawData->DisplayPos.y + drawData->DisplaySize.y;

        float mvp[4][4] =
        {
            { 2.0f / (R - L),   0.0f,           0.0f,       0.0f },
            { 0.0f,         2.0f / (T - B),     0.0f,       0.0f },
            { 0.0f,         0.0f,           0.5f,       0.0f },
            { (R + L) / (L - R),  (T + B) / (B - T),    0.5f,       1.0f },
        };

        ::SDL_PushGPUVertexUniformData(commandBuffer, 0, mvp, sizeof(mvp));

        auto vertexBufferBinding = SDL_GPUBufferBinding{
            .buffer = m_vertexBuffer,
            .offset = 0
        };

        ::SDL_BindGPUVertexBuffers(renderPass, 0, &vertexBufferBinding, 1);

        auto indexBufferBinding = SDL_GPUBufferBinding{
            .buffer = m_indexBuffer,
            .offset = 0
        };

        ::SDL_BindGPUIndexBuffer(renderPass, &indexBufferBinding, sizeof(ImDrawIdx) == sizeof(uint16_t) ? SDL_GPU_INDEXELEMENTSIZE_16BIT : SDL_GPU_INDEXELEMENTSIZE_32BIT);

        uint32_t global_idx_offset = 0;
        uint32_t global_vtx_offset = 0;

        ImVec2 clip_off = drawData->DisplayPos;
        for (int n = 0; n < drawData->CmdListsCount; n++)
        {
            const ImDrawList* cmd_list = drawData->CmdLists[n];

            for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; cmd_i++)
            {
                const ImDrawCmd* pcmd = &cmd_list->CmdBuffer[cmd_i];
                if (pcmd->UserCallback != nullptr)
                {
                    if (pcmd->UserCallback == ImDrawCallback_ResetRenderState)
                        ::SDL_BindGPUGraphicsPipeline(renderPass, m_pipeline);
                    else
                        pcmd->UserCallback(cmd_list, pcmd);
                }
                else
                {
                    // Project scissor/clipping rectangles into frame buffer space
                    ImVec2 clip_min(pcmd->ClipRect.x - clip_off.x, pcmd->ClipRect.y - clip_off.y);
                    ImVec2 clip_max(pcmd->ClipRect.z - clip_off.x, pcmd->ClipRect.w - clip_off.y);
                    if (clip_max.x <= clip_min.x || clip_max.y <= clip_min.y)
                        continue;

                    auto scissorDesc = ::SDL_Rect{
                        .x = (Sint32)clip_min.x,
                        .y = (Sint32)clip_min.y,
                        .w = (Sint32)(clip_max.x - clip_min.x),
                        .h = (Sint32)(clip_max.y - clip_min.y)
                    };

                    ::SDL_SetGPUScissor(renderPass, &scissorDesc);

                    auto samplerBinding = ::SDL_GPUTextureSamplerBinding{
                        .texture = (::SDL_GPUTexture*)pcmd->GetTexID(),
                        .sampler = m_textureSampler
                    };

                    ::SDL_BindGPUFragmentSamplers(renderPass, 0, &samplerBinding, 1);
                    ::SDL_DrawGPUIndexedPrimitives(renderPass, pcmd->ElemCount, 1, global_idx_offset + pcmd->IdxOffset, global_vtx_offset + pcmd->VtxOffset, 0);
                }
            }

            global_idx_offset += cmd_list->IdxBuffer.Size;
            global_vtx_offset += cmd_list->VtxBuffer.Size;
        }
    }
protected:
    ::SDL_GPUTransferBuffer* m_vertexTransferBuffer = nullptr;
    ::SDL_GPUTransferBuffer* m_indexTransferBuffer = nullptr;

    ::SDL_GPUSampler* m_textureSampler = nullptr;
    ::SDL_GPUBuffer* m_vertexBuffer = nullptr;
    ::SDL_GPUBuffer* m_indexBuffer = nullptr;
    ::SDL_GPUTexture* m_fontTexture = nullptr;

    ::SDL_GPUGraphicsPipeline* m_pipeline = nullptr;
};

int main()
{
    ::SDL_InitSubSystem(SDL_INIT_VIDEO);

    auto properties = ::SDL_CreateProperties();
    ::SDL_SetStringProperty(properties, SDL_PROP_GPU_DEVICE_CREATE_NAME_STRING, "D3D11");
    ::SDL_SetBooleanProperty(properties, SDL_PROP_GPU_DEVICE_CREATE_SHADERS_DXBC_BOOL, SDL_TRUE);

    auto device = ::SDL_CreateGPUDeviceWithProperties(properties);

    ::SDL_DestroyProperties(properties);

    int windowWidth = 1600;
    int windowHeight = 900;

    auto window = ::SDL_CreateWindow("", windowWidth, windowHeight, SDL_WINDOW_HIGH_PIXEL_DENSITY);
    ::SDL_ClaimGPUWindow(device, window);

    auto depthBufferDesc = ::SDL_GPUTextureCreateInfo{
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = SDL_GPU_TEXTUREFORMAT_D24_UNORM_S8_UINT,
        .usageFlags = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET_BIT,
        .width = (uint32_t)windowWidth,
        .height = (uint32_t)windowHeight,
        .layerCountOrDepth = 1,
        .levelCount = 1,
        .sampleCount = SDL_GPU_SAMPLECOUNT_1,
    };

    //! #TODO: support for resizing depth buffers!
    auto depthBuffer = ::SDL_CreateGPUTexture(device, &depthBufferDesc);

    ImGui::CreateContext();

    ImGuiRenderPass pass;
    pass.Initialize(device, window);

    bool running = true;
    while (running)
    {
        ::SDL_Event evt = {};
        while (::SDL_PollEvent(&evt))
        {
            if (evt.type == ::SDL_EVENT_QUIT)
                return 0;

            ImGui_ImplSDL3_ProcessEvent(&evt);
        }

        if (::SDL_GetWindowSizeInPixels(window, &windowWidth, &windowHeight))
        {
            ImGui::GetIO().DisplaySize = { (float)windowWidth, (float)windowHeight };
        }

        ImGui::NewFrame();

        ImGui::ShowDemoWindow();

        ImGui::EndFrame();
        ImGui::Render();

        auto commandBuffer = ::SDL_AcquireGPUCommandBuffer(device);

        uint32_t w = {}, h = {};
        if (auto windowTexture = SDL_AcquireGPUSwapchainTexture(commandBuffer, window, &w, &h); windowTexture != nullptr)
        {
            auto renderTargetDesc = SDL_GPUColorAttachmentInfo{};
            renderTargetDesc.texture = windowTexture;
            renderTargetDesc.clearColor = SDL_FColor{ 0.3f, 0.4f, 0.5f, 1.0f };
            renderTargetDesc.loadOp = SDL_GPU_LOADOP_CLEAR;
            renderTargetDesc.storeOp = SDL_GPU_STOREOP_STORE;

            auto depthStencilDesc = SDL_GPUDepthStencilAttachmentInfo{};
            depthStencilDesc.texture = depthBuffer;
            depthStencilDesc.cycle = SDL_TRUE;
            depthStencilDesc.depthStencilClearValue.depth = 0;
            depthStencilDesc.depthStencilClearValue.stencil = 0;
            depthStencilDesc.loadOp = SDL_GPU_LOADOP_CLEAR;
            depthStencilDesc.storeOp = SDL_GPU_STOREOP_DONT_CARE;
            depthStencilDesc.stencilLoadOp = SDL_GPU_LOADOP_CLEAR;
            depthStencilDesc.stencilStoreOp = SDL_GPU_STOREOP_DONT_CARE;

            auto drawData = ImGui::GetDrawData();
            pass.UpdateBuffers(device, commandBuffer, drawData);

            auto* renderPass = ::SDL_BeginGPURenderPass(commandBuffer, &renderTargetDesc, 1, &depthStencilDesc);
            pass.Render(commandBuffer, renderPass, drawData);
            ::SDL_EndGPURenderPass(renderPass);
        }

        ::SDL_SubmitGPU(commandBuffer);
    }
}
