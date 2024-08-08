
#include <SDL3/SDL.h>
#include <SDL3/SDL_video.h>
#include <SDL3/SDL_gpu.h>

#include "imgui_sdl3_handler.hpp"

#include <imgui.h>
#include <string>
#include <variant>
#include <vector>

static const std::string s_imguiVertexShader = R"(
cbuffer vertexBuffer : register(b0)
{
	float4x4 ProjectionMatrix;
};

struct VS_INPUT
{
	float2 pos : TEXCOORD0;
	float2 uv  : TEXCOORD1;
	float4 col : TEXCOORD2;
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
	output.col = input.col;
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

struct ImGuiRenderPass
{
public:
    void Initialize(::SDL_GpuDevice* device, ::SDL_Window* window)
    {
        auto vertexShaderDesc = SDL_GpuShaderCreateInfo{};
        vertexShaderDesc.code = (const uint8_t*)s_imguiVertexShader.data();
        vertexShaderDesc.codeSize = s_imguiVertexShader.size();
        vertexShaderDesc.entryPointName = "main";
        vertexShaderDesc.format = SDL_GPU_SHADERFORMAT_HLSL;
        vertexShaderDesc.stage = SDL_GPU_SHADERSTAGE_VERTEX;
        vertexShaderDesc.samplerCount = 0;
        vertexShaderDesc.uniformBufferCount = 1;
        vertexShaderDesc.storageBufferCount = 0;
        vertexShaderDesc.storageTextureCount = 0;

        auto vertexShader = ::SDL_GpuCreateShader(device, &vertexShaderDesc);

        auto fragmentShaderDesc = SDL_GpuShaderCreateInfo{};
        fragmentShaderDesc.code = (const uint8_t*)s_imguiFragmentShader.data();
        fragmentShaderDesc.codeSize = s_imguiFragmentShader.size();
        fragmentShaderDesc.entryPointName = "main";
        fragmentShaderDesc.format = SDL_GPU_SHADERFORMAT_HLSL;
        fragmentShaderDesc.stage = SDL_GPU_SHADERSTAGE_FRAGMENT;
        fragmentShaderDesc.samplerCount = 1;
        fragmentShaderDesc.uniformBufferCount = 0;
        fragmentShaderDesc.storageBufferCount = 0;
        fragmentShaderDesc.storageTextureCount = 0;

        auto fragmentShader = ::SDL_GpuCreateShader(device, &fragmentShaderDesc);

        auto attachmentDesc = SDL_GpuColorAttachmentDescription{};
        attachmentDesc.format = ::SDL_GpuGetSwapchainTextureFormat(device, window);
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
        auto pipelineDesc = SDL_GpuGraphicsPipelineCreateInfo{};
        pipelineDesc.attachmentInfo = {
            .colorAttachmentDescriptions = &attachmentDesc,
            .colorAttachmentCount = 1,
            .hasDepthStencilAttachment = SDL_TRUE,
            .depthStencilFormat = SDL_GPU_TEXTUREFORMAT_D24_UNORM_S8_UINT
        };

        pipelineDesc.depthStencilState.depthTestEnable = SDL_TRUE;
        pipelineDesc.depthStencilState.depthWriteEnable = SDL_FALSE;
        pipelineDesc.depthStencilState.compareOp = SDL_GPU_COMPAREOP_GREATER_OR_EQUAL;

        auto vertexBindingsDesc = ::SDL_GpuVertexBinding{
            .binding = 0,
            .stride = sizeof(ImDrawVert),
            .inputRate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
            .stepRate = 0,
        };

        SDL_GpuVertexAttribute vertexAttributesDesc[] = {
            ::SDL_GpuVertexAttribute{
                .location = 0,
                .binding = 0,
                .format = SDL_GPU_VERTEXELEMENTFORMAT_VECTOR2,
                .offset = 0
            },
            ::SDL_GpuVertexAttribute{
                .location = 1,
                .binding = 0,
                .format = SDL_GPU_VERTEXELEMENTFORMAT_VECTOR2,
                .offset = sizeof(float) * 2,
            },
            ::SDL_GpuVertexAttribute{
                .location = 2,
                .binding = 0,
                .format = SDL_GPU_VERTEXELEMENTFORMAT_COLOR,
                .offset = sizeof(float) * 4,
            },

        };

        pipelineDesc.vertexInputState = {
            .vertexBindings = &vertexBindingsDesc,
            .vertexBindingCount = 1,
            .vertexAttributes = vertexAttributesDesc,
            .vertexAttributeCount = 3,
        };

        pipelineDesc.rasterizerState = ::SDL_GpuRasterizerState{
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

        m_pipeline = ::SDL_GpuCreateGraphicsPipeline(device, &pipelineDesc);

        // Clean up shader resources
        ::SDL_GpuReleaseShader(device, vertexShader);
        ::SDL_GpuReleaseShader(device, fragmentShader);

        m_vertexBuffer = ::SDL_GpuCreateBuffer(device, ::SDL_GPU_BUFFERUSAGE_VERTEX_BIT, sizeof(ImDrawVert) * 1024 * 1024 * 64);
        ::SDL_GpuSetBufferName(device, m_vertexBuffer, "ImGui - VertexBuffer");

        m_vertexTransferBuffer = ::SDL_GpuCreateTransferBuffer(device, SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, sizeof(ImDrawVert) * 1024 * 1024 * 64);

        m_indexBuffer = ::SDL_GpuCreateBuffer(device, ::SDL_GPU_BUFFERUSAGE_INDEX_BIT, sizeof(ImDrawIdx) * 1024 * 1024 * 64);
        ::SDL_GpuSetBufferName(device, m_indexBuffer, "ImGui - IndexBuffer");

        m_indexTransferBuffer = ::SDL_GpuCreateTransferBuffer(device, SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, sizeof(ImDrawIdx) * 1024 * 1024 * 64);

        auto samplerDesc = ::SDL_GpuSamplerCreateInfo{
            .minFilter = SDL_GPU_FILTER_NEAREST,
            .magFilter = SDL_GPU_FILTER_NEAREST,
            .mipmapMode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST,
            .addressModeU = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            .addressModeV = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            .addressModeW = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        };

        m_textureSampler = SDL_GpuCreateSampler(device, &samplerDesc);

        unsigned char* pixels;
        int width, height;
        ImGui::GetIO().Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

        auto textureDesc = ::SDL_GpuTextureCreateInfo{
            .width = (uint32_t)width,
            .height = (uint32_t)height,
            .depth = 1,
            .isCube = SDL_FALSE,
            .layerCount = 1,
            .levelCount = 1,
            .sampleCount = {},
            .format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8,
            .usageFlags = SDL_GPU_TEXTUREUSAGE_SAMPLER_BIT
        };

        m_fontTexture = ::SDL_GpuCreateTexture(device, &textureDesc);
        ::SDL_GpuSetTextureName(device, m_fontTexture, "ImGui - FontAtas");

        auto transferDestDesc = ::SDL_GpuTextureRegion{};
        transferDestDesc.textureSlice.texture = m_fontTexture;
        transferDestDesc.w = (uint32_t)width;
        transferDestDesc.h = (uint32_t)height;
        transferDestDesc.d = 1;

        auto* transferBuffer = ::SDL_GpuCreateTransferBuffer(device, SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, width * height * sizeof(uint8_t) * 4);

        auto transferdDestDesc = ::SDL_GpuTransferBufferRegion{
            .transferBuffer = transferBuffer,
            .offset = 0,
            .size = (uint32_t)(width * height * sizeof(uint8_t) * 4)
        };

        ::SDL_GpuSetTransferData(device, pixels, &transferdDestDesc, SDL_FALSE);

        // Upload the transfer data to the vertex buffer
        auto uploadBuffer = ::SDL_GpuAcquireCommandBuffer(device);
        {
            auto* pass = ::SDL_GpuBeginCopyPass(uploadBuffer);
            {
                auto transferSourceDesc = ::SDL_GpuTextureTransferInfo{ transferBuffer };
                ::SDL_GpuUploadToTexture(pass, &transferSourceDesc, &transferDestDesc, SDL_FALSE);
            }
            ::SDL_GpuEndCopyPass(pass);
        }

        ::SDL_GpuSubmit(uploadBuffer);

        ::SDL_GpuReleaseTransferBuffer(device, transferBuffer);

        auto& io = ImGui::GetIO();
        io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
        io.Fonts->SetTexID(m_fontTexture);
    }

    void UpdateBuffers(::SDL_GpuDevice* device, ::SDL_GpuCommandBuffer* commandBuffer, const ::ImDrawData* drawData)
    {
        if (drawData->TotalVtxCount == 0)
            return;

        //! #TODO: verify if we have enough space in transfer buffer for new payload!
        //! #TODO: support for resizing buffers when we are out of space!

        uint8_t* vertices = nullptr;
        ::SDL_GpuMapTransferBuffer(device, m_vertexTransferBuffer, SDL_TRUE, (void**)&vertices);

        uint8_t* indices = nullptr;
        ::SDL_GpuMapTransferBuffer(device, m_indexTransferBuffer, SDL_TRUE, (void**)&indices);

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

        ::SDL_GpuUnmapTransferBuffer(device, m_vertexTransferBuffer);
        ::SDL_GpuUnmapTransferBuffer(device, m_indexTransferBuffer);

        auto* pass = ::SDL_GpuBeginCopyPass(commandBuffer);
        {
            auto transferSourceDesc = ::SDL_GpuTransferBufferLocation{
                .transferBuffer = m_vertexTransferBuffer
            };

            auto transferDestDesc = ::SDL_GpuBufferRegion{
                .buffer = m_vertexBuffer,
                .offset = 0,
                .size = (uint32_t)(drawData->TotalVtxCount * sizeof(ImDrawVert))
            };

            ::SDL_GpuUploadToBuffer(pass, &transferSourceDesc, &transferDestDesc, SDL_TRUE);
        }
        {
            auto transferSourceDesc = ::SDL_GpuTransferBufferLocation{
                .transferBuffer = m_indexTransferBuffer
            };

            auto transferDestDesc = ::SDL_GpuBufferRegion{
                .buffer = m_indexBuffer,
                .offset = 0,
                .size = (uint32_t)(drawData->TotalIdxCount * sizeof(ImDrawIdx))
            };

            ::SDL_GpuUploadToBuffer(pass, &transferSourceDesc, &transferDestDesc, SDL_FALSE);
        }

        ::SDL_GpuEndCopyPass(pass);
    }

    void Render(::SDL_GpuCommandBuffer* commandBuffer, ::SDL_GpuRenderPass* renderPass, const ::ImDrawData* drawData)
    {
        ::SDL_GpuBindGraphicsPipeline(renderPass, m_pipeline);

        auto viewportDesc = ::SDL_GpuViewport{
           .x = 0.0f,
           .y = 0.0f,
           .w = drawData->DisplaySize.x,
           .h = drawData->DisplaySize.y,
           .minDepth = 0.0f,
           .maxDepth = 1.0f,
        };

        ::SDL_GpuSetViewport(renderPass, &viewportDesc);

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

        ::SDL_GpuPushVertexUniformData(commandBuffer, 0, mvp, sizeof(mvp));

        auto vertexBufferBinding = SDL_GpuBufferBinding{
            .buffer = m_vertexBuffer,
            .offset = 0
        };

        ::SDL_GpuBindVertexBuffers(renderPass, 0, &vertexBufferBinding, 1);

        auto indexBufferBinding = SDL_GpuBufferBinding{
            .buffer = m_indexBuffer,
            .offset = 0
        };

        ::SDL_GpuBindIndexBuffer(renderPass, &indexBufferBinding, sizeof(ImDrawIdx) == sizeof(uint16_t) ? SDL_GPU_INDEXELEMENTSIZE_16BIT : SDL_GPU_INDEXELEMENTSIZE_32BIT);

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
                        ::SDL_GpuBindGraphicsPipeline(renderPass, m_pipeline);
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

                    auto scissorDesc = ::SDL_GpuRect{
                        .x = (Sint32)clip_min.x,
                        .y = (Sint32)clip_min.y,
                        .w = (Sint32)(clip_max.x - clip_min.x),
                        .h = (Sint32)(clip_max.y - clip_min.y)
                    };

                    ::SDL_GpuSetScissor(renderPass, &scissorDesc);

                    auto samplerBinding = ::SDL_GpuTextureSamplerBinding{
                        .texture = (::SDL_GpuTexture*)pcmd->GetTexID(),
                        .sampler = m_textureSampler
                    };

                    ::SDL_GpuBindFragmentSamplers(renderPass, 0, &samplerBinding, 1);
                    ::SDL_GpuDrawIndexedPrimitives(renderPass, global_vtx_offset + pcmd->VtxOffset, global_idx_offset + pcmd->IdxOffset, pcmd->ElemCount / 3, 1);
                }
            }

            global_idx_offset += cmd_list->IdxBuffer.Size;
            global_vtx_offset += cmd_list->VtxBuffer.Size;
        }
    }
protected:
    ::SDL_GpuTransferBuffer* m_vertexTransferBuffer = nullptr;
    ::SDL_GpuTransferBuffer* m_indexTransferBuffer = nullptr;

    ::SDL_GpuSampler* m_textureSampler = nullptr;
    ::SDL_GpuBuffer* m_vertexBuffer = nullptr;
    ::SDL_GpuBuffer* m_indexBuffer = nullptr;
    ::SDL_GpuTexture* m_fontTexture = nullptr;

    ::SDL_GpuGraphicsPipeline* m_pipeline = nullptr;
};

int main()
{
    ::SDL_InitSubSystem(SDL_INIT_VIDEO);

    auto device = ::SDL_GpuCreateDevice(SDL_GPU_BACKEND_D3D11, SDL_TRUE, SDL_FALSE);

    int windowWidth = 1600;
    int windowHeight = 900;

    auto window = ::SDL_CreateWindow("", windowWidth, windowHeight, SDL_WINDOW_HIGH_PIXEL_DENSITY);
    ::SDL_GpuClaimWindow(device, window, SDL_GPU_SWAPCHAINCOMPOSITION_SDR, SDL_GPU_PRESENTMODE_VSYNC);

    auto depthBufferDesc = ::SDL_GpuTextureCreateInfo{
        .width = (uint32_t)windowWidth,
        .height = (uint32_t)windowHeight,
        .depth = 1,
        .layerCount = 1,
        .levelCount = 1,
        .sampleCount = SDL_GPU_SAMPLECOUNT_1,
        .format = SDL_GPU_TEXTUREFORMAT_D24_UNORM_S8_UINT,
        .usageFlags = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET_BIT
    };

    //! #TODO: support for resizing depth buffers!
    auto depthBuffer = ::SDL_GpuCreateTexture(device, &depthBufferDesc);

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

        if (::SDL_GetWindowSizeInPixels(window, &windowWidth, &windowHeight) == 0)
        {
            ImGui::GetIO().DisplaySize = { (float)windowWidth, (float)windowHeight };
        }

        ImGui::NewFrame();

        ImGui::ShowDemoWindow();

        ImGui::EndFrame();
        ImGui::Render();

        auto commandBuffer = ::SDL_GpuAcquireCommandBuffer(device);

        uint32_t w = {}, h = {};
        if (auto windowTexture = SDL_GpuAcquireSwapchainTexture(commandBuffer, window, &w, &h); windowTexture != nullptr)
        {
            auto renderTargetDesc = SDL_GpuColorAttachmentInfo{};
            renderTargetDesc.textureSlice.texture = windowTexture;
            renderTargetDesc.clearColor = SDL_GpuColor{ 0.3f, 0.4f, 0.5f, 1.0f };
            renderTargetDesc.loadOp = SDL_GPU_LOADOP_CLEAR;
            renderTargetDesc.storeOp = SDL_GPU_STOREOP_STORE;

            auto depthStencilDesc = SDL_GpuDepthStencilAttachmentInfo{};
            depthStencilDesc.textureSlice.texture = depthBuffer;
            depthStencilDesc.cycle = SDL_TRUE;
            depthStencilDesc.depthStencilClearValue.depth = 0;
            depthStencilDesc.depthStencilClearValue.stencil = 0;
            depthStencilDesc.loadOp = SDL_GPU_LOADOP_CLEAR;
            depthStencilDesc.storeOp = SDL_GPU_STOREOP_DONT_CARE;
            depthStencilDesc.stencilLoadOp = SDL_GPU_LOADOP_CLEAR;
            depthStencilDesc.stencilStoreOp = SDL_GPU_STOREOP_DONT_CARE;

            auto drawData = ImGui::GetDrawData();
            pass.UpdateBuffers(device, commandBuffer, drawData);

            auto* renderPass = ::SDL_GpuBeginRenderPass(commandBuffer, &renderTargetDesc, 1, &depthStencilDesc);
            pass.Render(commandBuffer, renderPass, drawData);
            ::SDL_GpuEndRenderPass(renderPass);
        }

        ::SDL_GpuSubmit(commandBuffer);
    }
}
