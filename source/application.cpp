#include "application.hpp"
#include "math.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <vector>
#include <tuple>

#include <imgui.h>

namespace application {

namespace {

const auto& context = graphics::internal::context;

using lab_math::Mat4;
using lab_math::Vec3;

const uint32_t pyramid_count = 3;

enum class ProjectionType {
    Perspective = 0,
    Orthographic = 1,
};

struct Vertex {
    float position[3];
};

struct GlobalUniforms {
    float projection[16];
    float heatmapBounds[4];
};


struct ObjectUniforms {
    float model[16];
    float baseColor[4];
};

struct GpuBuffer {
    VkBuffer buffer = nullptr;
    VmaAllocation allocation = nullptr;
    void* mapped = nullptr;
    size_t size = 0;
};

struct Mesh {
    GpuBuffer vertexBuffer;
    GpuBuffer indexBuffer;

    uint32_t triangleIndexCount = 0;
    uint32_t edgeFirstIndex = 0;
    uint32_t edgeIndexCount = 0;
};

struct Pyramid {
    const char* name = "";
    uint32_t baseSides = 0;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    Mesh mesh;

    Vec3 position{};
    Vec3 rotationDegrees{};
    Vec3 scale{1.0f, 1.0f, 1.0f};
    float color[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    float phaseOffset = 0.0f;
    float animationMultiplier = 1.0f;
    bool animate = true;

    Mat4 modelMatrix{};

    VkDescriptorSet descriptorSet = nullptr;
    GpuBuffer uniformBuffer;
};

VkDescriptorSetLayout globalDescriptorSetLayout = nullptr;
VkDescriptorSetLayout objectDescriptorSetLayout = nullptr;
VkDescriptorPool descriptorPool = nullptr;
VkDescriptorSet globalDescriptorSet = nullptr;
GpuBuffer globalUniformBuffer;

VkPipelineLayout pipelineLayout = nullptr;
VkPipeline fillPipeline = nullptr;
VkPipeline edgePipeline = nullptr;

VkShaderModule pyramidVertexShader = nullptr;
VkShaderModule pyramidFragmentShader = nullptr;
VkShaderModule edgeVertexShader = nullptr;
VkShaderModule edgeFragmentShader = nullptr;

Pyramid pyramids[pyramid_count];

ProjectionType projectionType = ProjectionType::Perspective;

float orthographicHeight = 8.5f;
float perspectiveFovDegrees = 55.0f;
float nearPlane = 0.1f;
float farPlane = 60.0f;

const float heatmapLeft = -6.0f;
const float heatmapRight = 6.0f;
const float heatmapTop = -4.5f;
const float heatmapBottom = 4.5f;

double previousWallTime = 0.0;
double animationTime = 0.0;

bool animationPlaying = true;
float animationSpeed = 1.0f;
float trajectoryRadius = 1.0f;
float trajectoryVerticalAmplitude = 0.8f;
float trajectoryDepthAmplitude = 0.7f;
float trajectoryVerticalFrequency = 2.0f;
float trajectoryDepthFrequency = 0.5f;
Vec3 animatedRotationSpeedDegrees{20.0f, 30.0f, 40.0f};

bool resetAnimationRequested = false;

bool createHostBuffer(size_t size, VkBufferUsageFlags usage, GpuBuffer& result) {
    const VkBufferCreateInfo bufferInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    const VmaAllocationCreateInfo allocationInfo = {
        .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };

    if (vmaCreateBuffer(context.allocator, &bufferInfo, &allocationInfo,
                        &result.buffer, &result.allocation,
                        nullptr) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan buffer\n";
        return false;
    }

    if (vmaMapMemory(context.allocator, result.allocation, &result.mapped) != VK_SUCCESS) {
        std::cerr << "Failed to map vertex buffer memory\n";
        return false;
    }
    result.size = size;

    return true;
}

bool loadShaderModule(const char* path, VkShaderModule& shaderModule) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open shader: " << path << '\n';
        return false;
    }

    const std::streamsize fileSize = file.tellg();

    std::vector<uint32_t> code(static_cast<size_t>(fileSize / sizeof(uint32_t)));

    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), fileSize);
    file.close();

    const VkShaderModuleCreateInfo shaderInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = static_cast<size_t>(fileSize),
        .pCode = code.data(),
    };

    if (vkCreateShaderModule(context.device, &shaderInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        std::cerr << "Failed to create shader module: " << path << '\n';
        return false;
    }

    return true;
}

std::pair<std::vector<Vertex>, std::vector<uint32_t>> makePyramidGeometry(uint32_t baseSides) {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    const float radius = 0.8f;
    const float height = std::sqrt(2.0f) * radius;

    vertices.reserve(baseSides + 1);

    for (uint32_t i = 0; i < baseSides; ++i) {
        const float theta =
            2.0f * lab_math::PI * static_cast<float>(i) /
            static_cast<float>(baseSides);

        vertices.push_back({{
            radius * std::cos(theta),
            0.0f,
            radius * std::sin(theta),
        }});
    }

    const uint32_t apexIndex = baseSides;
    vertices.push_back({{0.0f, -height, 0.0f}});

    for (uint32_t i = 0; i < baseSides; ++i) {
        const uint32_t next = (i + 1) % baseSides;
        indices.push_back(i);
        indices.push_back(apexIndex);
        indices.push_back(next);
    }
    
    for (uint32_t i = 1; i + 1 < baseSides; ++i) {
        indices.push_back(0);
        indices.push_back(i);
        indices.push_back(i + 1);
    }

    return {std::move(vertices), std::move(indices)};
}

bool createMesh(Pyramid& pyramid) {
    std::tie(pyramid.vertices, pyramid.indices) = makePyramidGeometry(pyramid.baseSides);
    const uint32_t triangleIndexCount = static_cast<uint32_t>(pyramid.indices.size());

    const uint32_t edgeFirstIndex = static_cast<uint32_t>(pyramid.indices.size());

    for (uint32_t i = 0; i < pyramid.baseSides; ++i) {
        const uint32_t next = (i + 1) % pyramid.baseSides;
        pyramid.indices.push_back(i);
        pyramid.indices.push_back(next);
    }

    const uint32_t apexIndex = pyramid.baseSides;
    for (uint32_t i = 0; i < pyramid.baseSides; ++i) {
        pyramid.indices.push_back(apexIndex);
        pyramid.indices.push_back(i);
    }

    const uint32_t edgeIndexCount =
        static_cast<uint32_t>(pyramid.indices.size()) - edgeFirstIndex;

    pyramid.mesh.triangleIndexCount = triangleIndexCount;
    pyramid.mesh.edgeFirstIndex = edgeFirstIndex;
    pyramid.mesh.edgeIndexCount = edgeIndexCount;

    if (!createHostBuffer(sizeof(Vertex) * pyramid.vertices.size(),
                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                          pyramid.mesh.vertexBuffer)) {
        return false;
    }

    memcpy(pyramid.mesh.vertexBuffer.mapped,
                      pyramid.vertices.data(),
                      sizeof(Vertex) * pyramid.vertices.size());

    if (!createHostBuffer(sizeof(uint32_t) * pyramid.indices.size(),
                          VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                          pyramid.mesh.indexBuffer)) {
        return false;
    }

    memcpy(pyramid.mesh.indexBuffer.mapped,
                      pyramid.indices.data(),
                      sizeof(uint32_t) * pyramid.indices.size());

    return true;
}

bool createDescriptorLayouts() {
    const VkDescriptorSetLayoutBinding globalBinding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
    };

    const VkDescriptorSetLayoutCreateInfo globalLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &globalBinding,
    };

    if (vkCreateDescriptorSetLayout(context.device, &globalLayoutInfo, nullptr,
                                    &globalDescriptorSetLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create global descriptor set layout\n";
        return false;
    }

    const VkDescriptorSetLayoutBinding objectBinding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
    };

    const VkDescriptorSetLayoutCreateInfo objectLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &objectBinding,
    };

    if (vkCreateDescriptorSetLayout(context.device, &objectLayoutInfo, nullptr,
                                    &objectDescriptorSetLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create object descriptor set layout\n";
        return false;
    }

    return true;
}

bool createDescriptorPoolAndSets() {
    const VkDescriptorPoolSize poolSize = {
        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = pyramid_count + 1,
    };

    const VkDescriptorPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = pyramid_count + 1,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize,
    };

    if (vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        std::cerr << "Failed to create application descriptor pool\n";
        return false;
    }

    if (!createHostBuffer(sizeof(GlobalUniforms),
                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                          globalUniformBuffer)) {
        return false;
    }

    const VkDescriptorSetAllocateInfo globalAllocateInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &globalDescriptorSetLayout,
    };

    if (vkAllocateDescriptorSets(context.device, &globalAllocateInfo,
                                 &globalDescriptorSet) != VK_SUCCESS) {
        std::cerr << "Failed to allocate global descriptor set\n";
        return false;
    }

    for (Pyramid& pyramid : pyramids) {
        if (!createHostBuffer(sizeof(ObjectUniforms),
                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                              pyramid.uniformBuffer)) {
            return false;
        }

        const VkDescriptorSetAllocateInfo objectAllocateInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptorPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &objectDescriptorSetLayout,
        };

        if (vkAllocateDescriptorSets(context.device, &objectAllocateInfo,
                                     &pyramid.descriptorSet) != VK_SUCCESS) {
            std::cerr << "Failed to allocate object descriptor set\n";
            return false;
        }
    }

    const VkDescriptorBufferInfo globalBufferInfo = {
        .buffer = globalUniformBuffer.buffer,
        .offset = 0,
        .range = sizeof(GlobalUniforms),
    };

    const VkWriteDescriptorSet globalWrite = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = globalDescriptorSet,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &globalBufferInfo,
    };

    vkUpdateDescriptorSets(context.device, 1, &globalWrite, 0, nullptr);

    for (Pyramid& pyramid : pyramids) {
        const VkDescriptorBufferInfo objectBufferInfo = {
            .buffer = pyramid.uniformBuffer.buffer,
            .offset = 0,
            .range = sizeof(ObjectUniforms),
        };

        const VkWriteDescriptorSet objectWrite = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = pyramid.descriptorSet,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &objectBufferInfo,
        };

        vkUpdateDescriptorSets(context.device, 1, &objectWrite, 0, nullptr);
    }

    return true;
}

VkPipelineColorBlendStateCreateInfo makeBlendState(VkPipelineColorBlendAttachmentState& attachment) {
    attachment = {
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                          VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT |
                          VK_COLOR_COMPONENT_A_BIT,
    };

    return {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &attachment,
    };
}

bool createPipelines() {
    const VkDescriptorSetLayout setLayouts[2] = {
        globalDescriptorSetLayout,
        objectDescriptorSetLayout,
    };

    const VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<uint32_t>(2),
        .pSetLayouts = setLayouts,
    };

    if (vkCreatePipelineLayout(context.device, &pipelineLayoutInfo, nullptr,
                               &pipelineLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create pipeline layout\n";
        return false;
    }

    if (!loadShaderModule("shaders/pyramid.vert.spv", pyramidVertexShader) ||
        !loadShaderModule("shaders/pyramid.frag.spv", pyramidFragmentShader) ||
        !loadShaderModule("shaders/edge.vert.spv", edgeVertexShader) ||
        !loadShaderModule("shaders/edge.frag.spv", edgeFragmentShader)) {
        return false;
    }

    const VkVertexInputBindingDescription binding = {
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };

    const VkVertexInputAttributeDescription attribute = {
        .location = 0,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(Vertex, position),
    };

    const VkPipelineVertexInputStateCreateInfo vertexInput = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = 1,
        .pVertexAttributeDescriptions = &attribute,
    };

    const VkPipelineInputAssemblyStateCreateInfo fillAssembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    const VkPipelineInputAssemblyStateCreateInfo edgeAssembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
    };

    const VkPipelineViewportStateCreateInfo viewportState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };

    const VkPipelineRasterizationStateCreateInfo fillRaster = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f,
    };

    const VkPipelineRasterizationStateCreateInfo edgeRaster = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .lineWidth = 1.0f,
    };

    const VkPipelineMultisampleStateCreateInfo multisampleState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    const VkPipelineDepthStencilStateCreateInfo fillDepth = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
    };

    const VkPipelineDepthStencilStateCreateInfo edgeDepth = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_FALSE,
        .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
    };

    const VkDynamicState dynamicStates[2] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    const VkPipelineDynamicStateCreateInfo dynamicState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = dynamicStates,
    };

    VkPipelineColorBlendAttachmentState fillAttachment{};
    const VkPipelineColorBlendStateCreateInfo fillBlend =
        makeBlendState(fillAttachment);

    VkPipelineColorBlendAttachmentState edgeAttachment{};
    const VkPipelineColorBlendStateCreateInfo edgeBlend =
        makeBlendState(edgeAttachment);

    const VkPipelineShaderStageCreateInfo fillStages[] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = pyramidVertexShader,
            .pName = "main",
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = pyramidFragmentShader,
            .pName = "main",
        },
    };

    const VkPipelineShaderStageCreateInfo edgeStages[] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = edgeVertexShader,
            .pName = "main",
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = edgeFragmentShader,
            .pName = "main",
        },
    };

    const VkGraphicsPipelineCreateInfo fillPipelineInfo = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = fillStages,
        .pVertexInputState = &vertexInput,
        .pInputAssemblyState = &fillAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &fillRaster,
        .pMultisampleState = &multisampleState,
        .pDepthStencilState = &fillDepth,
        .pColorBlendState = &fillBlend,
        .pDynamicState = &dynamicState,
        .layout = pipelineLayout,
        .renderPass = context.render_pass,
    };

    if (vkCreateGraphicsPipelines(context.device, nullptr, 1,
                                  &fillPipelineInfo, nullptr,
                                  &fillPipeline) != VK_SUCCESS) {
        std::cerr << "Failed to create pyramid fill pipeline\n";
        return false;
    }

    const VkGraphicsPipelineCreateInfo edgePipelineInfo = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = edgeStages,
        .pVertexInputState = &vertexInput,
        .pInputAssemblyState = &edgeAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &edgeRaster,
        .pMultisampleState = &multisampleState,
        .pDepthStencilState = &edgeDepth,
        .pColorBlendState = &edgeBlend,
        .pDynamicState = &dynamicState,
        .layout = pipelineLayout,
        .renderPass = context.render_pass,
    };

    if (vkCreateGraphicsPipelines(context.device, nullptr, 1,
                                  &edgePipelineInfo, nullptr,
                                  &edgePipeline) != VK_SUCCESS) {
        std::cerr << "Failed to create pyramid edge pipeline\n";
        return false;
    }

    return true;
}

Mat4 makeModelMatrix(const Pyramid& pyramid) {
    
    const float phase = static_cast<float>(
        animationTime * pyramid.animationMultiplier + pyramid.phaseOffset);

    Vec3 animatedOffset{};
    Vec3 animatedRotationDegrees{};

    if (pyramid.animate) {
        
        animatedOffset = {
            trajectoryRadius * std::cos(phase),
            trajectoryVerticalAmplitude *
                std::sin(trajectoryVerticalFrequency * phase + pyramid.phaseOffset * 0.5f),
            trajectoryDepthAmplitude *
                std::sin(trajectoryDepthFrequency * phase + pyramid.phaseOffset),
        };

        animatedRotationDegrees = {
            animatedRotationSpeedDegrees.x * static_cast<float>(animationTime),
            animatedRotationSpeedDegrees.y * static_cast<float>(animationTime),
            animatedRotationSpeedDegrees.z * static_cast<float>(animationTime),
        };
    }

    const Vec3 finalPosition = lab_math::add(pyramid.position, animatedOffset);
    const Vec3 finalRotationDegrees = lab_math::add(
        pyramid.rotationDegrees,
        animatedRotationDegrees);

    const Vec3 finalRotationRadians = {
        lab_math::radians(finalRotationDegrees.x),
        lab_math::radians(finalRotationDegrees.y),
        lab_math::radians(finalRotationDegrees.z),
    };

    const Mat4 S = lab_math::scaleMatrix(pyramid.scale);
    const Mat4 Rx = lab_math::rotateX(finalRotationRadians.x);
    const Mat4 Ry = lab_math::rotateY(finalRotationRadians.y);
    const Mat4 Rz = lab_math::rotateZ(finalRotationRadians.z);
    const Mat4 T = lab_math::translate(finalPosition);

    const Mat4 RxS = lab_math::multiply(Rx, S);
    const Mat4 RyRxS = lab_math::multiply(Ry, RxS);
    const Mat4 RzRyRxS = lab_math::multiply(Rz, RyRxS);

    return lab_math::multiply(T, RzRyRxS);
}

Mat4 makeProjectionMatrix() {

    const float width = static_cast<float>(context.swapchain_extent.width);
    const float height = static_cast<float>(context.swapchain_extent.height);
    const float aspect = width / height;

    if (projectionType == ProjectionType::Orthographic) {
        
        const float halfHeight = orthographicHeight * 0.5f;
        const float halfWidth = halfHeight * aspect;

        return lab_math::orthographic(
            -halfWidth, halfWidth,
            -halfHeight, halfHeight,
            nearPlane, farPlane);
    }

    return lab_math::perspective(
        lab_math::radians(perspectiveFovDegrees),
        aspect,
        nearPlane,
        farPlane);
}

void drawProjectionUI() {
    if (!ImGui::CollapsingHeader("Projection", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    int projection = static_cast<int>(projectionType);

    const char* projectionItems[] = {
        "Perspective",
        "Orthographic",
    };

    if (ImGui::Combo("Projection type", &projection, projectionItems, 2)) {
        projectionType = static_cast<ProjectionType>(projection);
    }

    if (projectionType == ProjectionType::Perspective) {
        ImGui::SliderFloat("FOV Y", &perspectiveFovDegrees, 25.0f, 100.0f, "%.1f deg");
    } else {
        ImGui::SliderFloat("Ortho area height", &orthographicHeight, 4.0f, 15.0f, "%.2f");
    }

    if (ImGui::SliderFloat("Near", &nearPlane, 0.01f, 19.9f, "%.2f")) {
        nearPlane = std::min(nearPlane, farPlane - 0.01f);
        nearPlane = std::max(nearPlane, 0.01f);
    }

    if (ImGui::SliderFloat("Far", &farPlane, 0.05f, 20.0f, "%.2f")) {
        farPlane = std::max(farPlane, nearPlane + 0.01f);
        farPlane = std::min(farPlane, 20.0f);
    }
}

void drawAnimationUI() {
    if (!ImGui::CollapsingHeader("Animation & Trajectory", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    if (ImGui::Button(animationPlaying ? "Pause" : "Play")) {
        animationPlaying = !animationPlaying;
    }

    ImGui::SameLine();
    if (ImGui::Button("Reset time")) {
        resetAnimationRequested = true;
    }

    ImGui::SliderFloat("Speed", &animationSpeed, 0.0f, 3.0f, "%.2fx");
    ImGui::SliderFloat("Trajectory radius", &trajectoryRadius, 0.0f, 2.5f, "%.2f");
    ImGui::SliderFloat("Y amplitude", &trajectoryVerticalAmplitude, 0.0f, 2.5f, "%.2f");
    ImGui::SliderFloat("Z amplitude", &trajectoryDepthAmplitude, 0.0f, 2.0f, "%.2f");
    ImGui::SliderFloat("Y frequency", &trajectoryVerticalFrequency, 0.1f, 5.0f, "%.2f");
    ImGui::SliderFloat("Z frequency", &trajectoryDepthFrequency, 0.1f, 3.0f, "%.2f");
    ImGui::DragFloat3("Rotation speed", &animatedRotationSpeedDegrees.x,
                      1.0f, -180.0f, 180.0f, "%.1f deg/s");
}

void drawPyramidUI(Pyramid& pyramid) {
    if (!ImGui::TreeNode(pyramid.name)) {
        return;
    }

    ImGui::Checkbox("Animate", &pyramid.animate);
    ImGui::DragFloat("Speed multiplier", &pyramid.animationMultiplier,
                     0.01f, 0.0f, 3.0f, "%.2f");
    ImGui::DragFloat("Phase offset", &pyramid.phaseOffset,
                     0.01f, -lab_math::PI, lab_math::PI, "%.2f rad");

    ImGui::Separator();

    ImGui::DragFloat3("Position", &pyramid.position.x, 0.01f, -10.0f, 10.0f, "%.2f");
    ImGui::DragFloat3("Rotation", &pyramid.rotationDegrees.x,
                      1.0f, -360.0f, 360.0f, "%.1f deg");
    ImGui::DragFloat3("Scale", &pyramid.scale.x,
                      0.01f, 0.1f, 4.0f, "%.2f");
    ImGui::ColorEdit3("Pyramid color", pyramid.color);

    ImGui::Text("Base sides: %u", pyramid.baseSides);
    ImGui::Text("Edges: %u", pyramid.mesh.edgeIndexCount / 2);

    ImGui::TreePop();
}

void drawUI() {
    ImGui::Begin("Vulkan Lab 1 - Pyramids");

    drawProjectionUI();
    drawAnimationUI();

    if (ImGui::CollapsingHeader("Pyramids", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (Pyramid& pyramid : pyramids) {
            drawPyramidUI(pyramid);
        }
    }

    ImGui::End();
}

void updateGlobalUniform() {
    GlobalUniforms global{};

    const Mat4 projection = makeProjectionMatrix();
    lab_math::toColumnMajor(projection, global.projection);

    global.heatmapBounds[0] = heatmapLeft;
    global.heatmapBounds[1] = heatmapRight;
    global.heatmapBounds[2] = heatmapTop;
    global.heatmapBounds[3] = heatmapBottom;

    memcpy(globalUniformBuffer.mapped, &global, sizeof(global));
}

void updateObjectUniform(Pyramid& pyramid) {
    ObjectUniforms object{};

    pyramid.modelMatrix = makeModelMatrix(pyramid);
    lab_math::toColumnMajor(pyramid.modelMatrix, object.model);

    object.baseColor[0] = pyramid.color[0];
    object.baseColor[1] = pyramid.color[1];
    object.baseColor[2] = pyramid.color[2];
    object.baseColor[3] = pyramid.color[3];

    memcpy(pyramid.uniformBuffer.mapped, &object, sizeof(object));
}

void drawPyramid(const graphics::internal::FrameData& fd, Pyramid& pyramid) {
    const size_t vertexBufferOffset = 0;

    vkCmdBindVertexBuffers(fd.command_buffer,
                           0, 1,
                           &pyramid.mesh.vertexBuffer.buffer,
                           &vertexBufferOffset);

    vkCmdBindIndexBuffer(fd.command_buffer,
                         pyramid.mesh.indexBuffer.buffer,
                         0,
                         VK_INDEX_TYPE_UINT32);

    vkCmdBindDescriptorSets(fd.command_buffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout,
                            0, 1,
                            &globalDescriptorSet,
                            0, nullptr);

    vkCmdBindDescriptorSets(fd.command_buffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout,
                            1, 1,
                            &pyramid.descriptorSet,
                            0, nullptr);

    vkCmdBindPipeline(fd.command_buffer,
                      VK_PIPELINE_BIND_POINT_GRAPHICS,
                      fillPipeline);

    vkCmdDrawIndexed(fd.command_buffer,
                     pyramid.mesh.triangleIndexCount,
                     1,
                     0,
                     0,
                     0);

    vkCmdBindPipeline(fd.command_buffer,
                      VK_PIPELINE_BIND_POINT_GRAPHICS,
                      edgePipeline);

    vkCmdDrawIndexed(fd.command_buffer,
                     pyramid.mesh.edgeIndexCount,
                     1,
                     pyramid.mesh.edgeFirstIndex,
                     0,
                     0);
}

} // namespace

bool initialize() {
    
    pyramids[0].name = "Tetrahedron (3 base sides)";
    pyramids[0].baseSides = 3;
    pyramids[0].position = {-3.0f, 0.0f, 6.0f};
    pyramids[0].color[0] = 1.0f;
    pyramids[0].color[1] = 1.0f;
    pyramids[0].color[2] = 1.0f;
    pyramids[0].phaseOffset = 0.0f;

    pyramids[1].name = "Square pyramid";
    pyramids[1].baseSides = 4;
    pyramids[1].position = {0.0f, 0.0f, 6.5f};
    pyramids[1].color[0] = 1.0f;
    pyramids[1].color[1] = 0.80f;
    pyramids[1].color[2] = 0.70f;
    pyramids[1].phaseOffset = 2.0f * lab_math::PI / 3.0f;

    pyramids[2].name = "Pentagonal pyramid";
    pyramids[2].baseSides = 5;
    pyramids[2].position = {3.0f, 0.0f, 7.0f};
    pyramids[2].color[0] = 0.70f;
    pyramids[2].color[1] = 0.85f;
    pyramids[2].color[2] = 1.0f;
    pyramids[2].phaseOffset = 4.0f * lab_math::PI / 3.0f;

    for (Pyramid& pyramid : pyramids) {
        if (!createMesh(pyramid)) {
            std::cerr << "Failed to create mesh for " << pyramid.name << '\n';
            return false;
        }
    }

    if (!createDescriptorLayouts()) {
        return false;
    }

    if (!createDescriptorPoolAndSets()) {
        return false;
    }

    if (!createPipelines()) {
        return false;
    }

    previousWallTime = 0.0;
    animationTime = 0.0;
    animationPlaying = true;

    return true;
}

void shutdown() {
    
    vkQueueWaitIdle(context.graphics_queue);
    
    for (Pyramid& pyramid : pyramids) {
        vmaDestroyBuffer(context.allocator, pyramid.uniformBuffer.buffer,
                         pyramid.uniformBuffer.allocation);
        vmaDestroyBuffer(context.allocator, pyramid.mesh.vertexBuffer.buffer,
                         pyramid.mesh.vertexBuffer.allocation);
        vmaDestroyBuffer(context.allocator, pyramid.mesh.indexBuffer.buffer,
                         pyramid.mesh.indexBuffer.allocation);
    }

    vmaDestroyBuffer(context.allocator, globalUniformBuffer.buffer,
                     globalUniformBuffer.allocation);

    vkDestroyPipeline(context.device, fillPipeline, nullptr);
    vkDestroyPipeline(context.device, edgePipeline, nullptr);

    vkDestroyShaderModule(context.device, pyramidVertexShader, nullptr);
    vkDestroyShaderModule(context.device, pyramidFragmentShader, nullptr);
    vkDestroyShaderModule(context.device, edgeVertexShader, nullptr);
    vkDestroyShaderModule(context.device, edgeFragmentShader, nullptr);

    vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);

    vkDestroyDescriptorPool(context.device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(context.device, globalDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(context.device, objectDescriptorSetLayout, nullptr);
}

void update(double time) {
    double delta = time - previousWallTime;
    previousWallTime = time;

    drawUI();

    if (resetAnimationRequested) {
        animationTime = 0.0;
        resetAnimationRequested = false;
    } else if (animationPlaying) {
        animationTime += delta * animationSpeed;
    }
}

void render(const graphics::internal::FrameData& fd) {
    updateGlobalUniform();
    for (Pyramid& pyramid : pyramids) {
        updateObjectUniform(pyramid);
    }

    vkResetCommandBuffer(fd.command_buffer, 0);

    const VkCommandBufferBeginInfo commandBufferBegin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    if (vkBeginCommandBuffer(fd.command_buffer, &commandBufferBegin) != VK_SUCCESS) {
        std::cerr << "Failed to begin application command buffer\n";
        return;
    }

    const VkClearValue clearValues[] = {
        {
            .color = {
                .float32 = {0.035f, 0.04f, 0.055f, 1.0f},
            },
        },
        {
            .depthStencil = {1.0f, 0},
        },
    };

    const VkRenderPassBeginInfo renderPassBegin = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = context.render_pass,
        .framebuffer = fd.framebuffer,
        .renderArea = { .extent = context.swapchain_extent, },
        .clearValueCount = 2,
        .pClearValues = clearValues,
    };

    vkCmdBeginRenderPass(fd.command_buffer,
                         &renderPassBegin,
                         VK_SUBPASS_CONTENTS_INLINE);

    const VkViewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(context.swapchain_extent.width),
        .height = static_cast<float>(context.swapchain_extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    const VkRect2D scissor = { .extent = context.swapchain_extent, };

    vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
    vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

    for (Pyramid& pyramid : pyramids) {
        drawPyramid(fd, pyramid);
    }

    vkCmdEndRenderPass(fd.command_buffer);

    if (vkEndCommandBuffer(fd.command_buffer) != VK_SUCCESS) {
        std::cerr << "Failed to end application command buffer\n";
    }
}

} // namespace application
