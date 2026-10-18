// ============================================================================
//  application.cpp  —  ЛАБОРАТОРНАЯ №2: «Камера и освещение»
//
//  КРАТКО: ЧТО ИЗМЕНИЛОСЬ ПО СРАВНЕНИЮ С ЛАБОРАТОРНОЙ №1
//  ---------------------------------------------------------------------------
//  1. ВЕРШИНА: к позиции добавилась НОРМАЛЬ (struct Vertex). Нормали граней
//     генерируются процедурно (векторное произведение, makePyramidGeometry).
//     Чтобы грани были "гранёными" (у каждой грани своя нормаль), вершины
//     каждого треугольника теперь СВОИ (не общие между гранями).
//  2. ДВА НАБОРА ДЕСКРИПТОРОВ (доп. задание 1):
//        set = 0 — СЦЕНА:  binding 0 = uniform SceneUniforms,
//                          binding 1 = storage-буфер точечных источников,
//                          binding 2 = storage-буфер прожекторных источников;
//        set = 1 — МОДЕЛЬ: binding 0 = uniform ObjectUniforms
//                          (матрица модели + материал).
//     GlobalUniforms из лабы №1 стал SceneUniforms (переименован и расширен).
//  3. МАТЕРИАЛ: вместо цвета baseColor — альбедо Kd, цвет блика Ks, блеск alpha.
//  4. КАМЕРА: матрица вида V = R^T * T(-c), окно "Vulkan Lab 2 - Camera"
//     (6 степеней свободы: позиция X/Y/Z + yaw/pitch/roll). Блок "Projection"
//     переехал из окна пирамид в окно камеры.
//  5. СВЕТ: окно "Vulkan Lab 2 - Lighting": ambient, направленный источник,
//     списки точечных и прожекторных источников (добавить/удалить/править).
//  6. ВИЗУАЛИЗАЦИЯ: сферы-маркеры источников и проволочные конусы прожекторов
//     (два новых пайплайна, меши сферы и конуса).
//  7. createPipelines() переписана через вспомогательную функцию
//     createGraphicsPipeline(), чтобы не копировать одни и те же структуры
//     4 раза (параметры состояний — те же, что были в лабе №1).
//  8. В render(): набор 0 привязывается ОДИН РАЗ за кадр, а не для каждой
//     пирамиды; добавлена отрисовка маркеров и конусов.
//
//  Что НЕ менялось: анимация пирамид, makeModelMatrix, makeProjectionMatrix,
//  loadShaderModule, createHostBuffer, рёбра пирамид (чёрные, без освещения),
//  тепловая карта (теперь её можно ослабить/выключить слайдером).
//
//  СИСТЕМА КООРДИНАТ: +X вправо, +Y ВНИЗ, +Z вглубь экрана (см. math.hpp).
//  Поэтому "над сценой" — это ОТРИЦАТЕЛЬНЫЙ Y, а свет "вниз" — это +Y.
// ============================================================================

#include "application.hpp"
#include "math.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>
#include <tuple>

#include <imgui.h>

namespace application {

namespace {

const auto& context = graphics::internal::context;

using lab_math::Mat4;
using lab_math::Vec3;

const uint32_t pyramid_count = 3;

const uint32_t max_point_lights = 64;
const uint32_t max_spot_lights = 64;

enum class ProjectionType {
    Perspective = 0,
    Orthographic = 1,
};

struct Vertex {
    float position[3];
    float normal[3];
};

struct SceneUniforms {
    float viewProjection[16];                  
    alignas(16) float cameraPosition[3];       
    alignas(16) float directionalDirection[3]; 
    alignas(16) float directionalColor[3];     
    alignas(16) float ambientLight[3];         
    uint32_t pointLightCount;                  
    uint32_t spotLightCount;                   
    float heatmapStrength;                     
    float markerRadius;                        
    float coneLength;                          
    alignas(16) float heatmapBounds[4];        
};

struct ObjectUniforms {
    float model[16];
    alignas(16) float albedo[3];
    alignas(16) float specular[3];
    float shininess;
};

struct GpuPointLight {
    float position[3];
    float intensity;
    float color[3];
    float _padding;
};

struct GpuSpotLight {
    float position[3];
    float intensity;
    float color[3];
    float cosInner;
    float direction[3];
    float cosOuter;
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
    float specularColor[3] = {0.5f, 0.5f, 0.5f};
    float shininess = 32.0f;

    float phaseOffset = 0.0f;
    float animationMultiplier = 1.0f;
    bool animate = true;

    Mat4 modelMatrix{};

    VkDescriptorSet descriptorSet = nullptr;
    GpuBuffer uniformBuffer;
};

struct PointLight {
    Vec3 position{};
    float color[3] = {1.0f, 1.0f, 1.0f};
    float intensity = 10.0f;
};

struct SpotLight {
    Vec3 position{};
    float yawDegrees = 0.0f;
    float pitchDegrees = -90.0f;
    float color[3] = {1.0f, 1.0f, 1.0f};
    float intensity = 20.0f;
    float innerAngleDegrees = 20.0f;
    float outerAngleDegrees = 35.0f;
};

VkDescriptorSetLayout sceneDescriptorSetLayout = nullptr;
VkDescriptorSetLayout objectDescriptorSetLayout = nullptr;
VkDescriptorPool descriptorPool = nullptr;
VkDescriptorSet sceneDescriptorSet = nullptr;

GpuBuffer sceneUniformBuffer;
GpuBuffer pointLightBuffer;
GpuBuffer spotLightBuffer;

VkPipelineLayout pipelineLayout = nullptr;
VkPipeline fillPipeline = nullptr;
VkPipeline edgePipeline = nullptr;
VkPipeline markerPipeline = nullptr;
VkPipeline conePipeline = nullptr;

VkShaderModule pyramidVertexShader = nullptr;
VkShaderModule pyramidFragmentShader = nullptr;
VkShaderModule edgeVertexShader = nullptr;
VkShaderModule edgeFragmentShader = nullptr;
VkShaderModule markerVertexShader = nullptr;
VkShaderModule markerFragmentShader = nullptr;
VkShaderModule coneVertexShader = nullptr;

Pyramid pyramids[pyramid_count];

Mesh sphereMesh;
Mesh coneMesh;

ProjectionType projectionType = ProjectionType::Perspective;

float orthographicHeight = 8.5f;
float perspectiveFovDegrees = 55.0f;
float nearPlane = 0.1f;
float farPlane = 60.0f;

const Vec3 defaultCameraPosition{0.0f, 0.0f, -3.0f};

Vec3 cameraPosition = defaultCameraPosition;
float cameraYawDegrees = 0.0f;
float cameraPitchDegrees = 0.0f;
float cameraRollDegrees = 0.0f;
float cameraMoveStep = 0.25f;

float ambientColor[3] = {1.0f, 1.0f, 1.0f};
float ambientIntensity = 0.08f;

bool directionalEnabled = false;
float directionalColor[3] = {1.0f, 1.0f, 1.0f};
float directionalIntensity = 1.0f;
float directionalYawDegrees = 0.0f;
float directionalPitchDegrees = -45.0f;

std::vector<PointLight> pointLights;
std::vector<SpotLight> spotLights;

const Vec3 sceneCenter{0.0f, 0.0f, 6.5f};

bool showLightMarkers = true;
bool showSpotCones = true;
float markerRadius = 0.15f;
float coneLength = 3.5f;

const float heatmapLeft = -6.0f;
const float heatmapRight = 6.0f;
const float heatmapTop = -4.5f;
const float heatmapBottom = 4.5f;
float heatmapStrength = 1.0f;

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

void destroyBuffer(GpuBuffer& buffer) {
    if (buffer.buffer != nullptr) {
        vmaDestroyBuffer(context.allocator, buffer.buffer, buffer.allocation);
        buffer.buffer = nullptr;
        buffer.allocation = nullptr;
        buffer.mapped = nullptr;
    }
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

    std::vector<Vec3> corners;
    corners.reserve(baseSides + 1);

    for (uint32_t i = 0; i < baseSides; ++i) {
        const float theta =
            2.0f * lab_math::PI * static_cast<float>(i) /
            static_cast<float>(baseSides);

        corners.push_back({
            radius * std::cos(theta),
            0.0f,
            radius * std::sin(theta),
        });
    }

    corners.push_back({0.0f, -height, 0.0f});

    for (const Vec3& corner : corners) {
        vertices.push_back({{corner.x, corner.y, corner.z}, {0.0f, 0.0f, 0.0f}});
    }

    const Vec3 bodyCenter{0.0f, -height * 0.25f, 0.0f};

    auto addTriangle = [&](const Vec3& a, const Vec3& b, const Vec3& c) {
        Vec3 normal = lab_math::normalize(
            lab_math::cross(lab_math::sub(b, a), lab_math::sub(c, a)));

        const Vec3 faceCenter = lab_math::scale(
            lab_math::add(lab_math::add(a, b), c), 1.0f / 3.0f);

        if (lab_math::dot(normal, lab_math::sub(faceCenter, bodyCenter)) < 0.0f) {
            normal = lab_math::scale(normal, -1.0f);
        }

        const uint32_t first = static_cast<uint32_t>(vertices.size());
        const Vec3 points[3] = {a, b, c};

        for (const Vec3& p : points) {
            vertices.push_back({{p.x, p.y, p.z}, {normal.x, normal.y, normal.z}});
        }

        indices.push_back(first);
        indices.push_back(first + 1);
        indices.push_back(first + 2);
    };

    const Vec3& apex = corners[baseSides];

    for (uint32_t i = 0; i < baseSides; ++i) {
        const uint32_t next = (i + 1) % baseSides;
        addTriangle(corners[i], apex, corners[next]);
    }

    for (uint32_t i = 1; i + 1 < baseSides; ++i) {
        addTriangle(corners[0], corners[i], corners[i + 1]);
    }

    return {std::move(vertices), std::move(indices)};
}

std::pair<std::vector<Vertex>, std::vector<uint32_t>> makeSphereGeometry(uint32_t stacks,
                                                                         uint32_t slices) {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    for (uint32_t i = 0; i <= stacks; ++i) {
        const float phi = lab_math::PI * static_cast<float>(i) / static_cast<float>(stacks);

        for (uint32_t j = 0; j <= slices; ++j) {
            const float theta = 2.0f * lab_math::PI * static_cast<float>(j) /
                                static_cast<float>(slices);

            const float x = std::sin(phi) * std::cos(theta);
            const float y = std::cos(phi);
            const float z = std::sin(phi) * std::sin(theta);

            vertices.push_back({{x, y, z}, {x, y, z}});
        }
    }

    for (uint32_t i = 0; i < stacks; ++i) {
        for (uint32_t j = 0; j < slices; ++j) {
            const uint32_t a = i * (slices + 1) + j;
            const uint32_t b = a + slices + 1;

            indices.push_back(a);
            indices.push_back(b);
            indices.push_back(a + 1);

            indices.push_back(a + 1);
            indices.push_back(b);
            indices.push_back(b + 1);
        }
    }

    return {std::move(vertices), std::move(indices)};
}

std::pair<std::vector<Vertex>, std::vector<uint32_t>> makeConeGeometry(uint32_t segments) {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    vertices.push_back({{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}});

    for (uint32_t i = 0; i < segments; ++i) {
        const float a = 2.0f * lab_math::PI * static_cast<float>(i) /
                        static_cast<float>(segments);
        vertices.push_back({{std::cos(a), std::sin(a), 1.0f}, {0.0f, 0.0f, 0.0f}});
    }

    for (uint32_t i = 0; i < segments; ++i) {
        indices.push_back(1 + i);
        indices.push_back(1 + (i + 1) % segments);
    }

    for (uint32_t i = 0; i < segments; i += 4) {
        indices.push_back(0);
        indices.push_back(1 + i);
    }

    return {std::move(vertices), std::move(indices)};
}

bool uploadMesh(const std::vector<Vertex>& vertices,
                const std::vector<uint32_t>& indices,
                Mesh& mesh) {
    if (!createHostBuffer(sizeof(Vertex) * vertices.size(),
                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                          mesh.vertexBuffer)) {
        return false;
    }

    std::memcpy(mesh.vertexBuffer.mapped, vertices.data(), sizeof(Vertex) * vertices.size());

    if (!createHostBuffer(sizeof(uint32_t) * indices.size(),
                          VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                          mesh.indexBuffer)) {
        return false;
    }

    std::memcpy(mesh.indexBuffer.mapped, indices.data(), sizeof(uint32_t) * indices.size());

    return true;
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

    return uploadMesh(pyramid.vertices, pyramid.indices, pyramid.mesh);
}

bool createMarkerMeshes() {
    {
        auto [vertices, indices] = makeSphereGeometry(16, 32);
        sphereMesh.triangleIndexCount = static_cast<uint32_t>(indices.size());
        if (!uploadMesh(vertices, indices, sphereMesh)) {
            return false;
        }
    }

    {
        auto [vertices, indices] = makeConeGeometry(32);
        coneMesh.edgeFirstIndex = 0;
        coneMesh.edgeIndexCount = static_cast<uint32_t>(indices.size());
        if (!uploadMesh(vertices, indices, coneMesh)) {
            return false;
        }
    }

    return true;
}

bool createDescriptorLayouts() {
    const VkShaderStageFlags bothStages =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    const VkDescriptorSetLayoutBinding sceneBindings[3] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = bothStages,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = bothStages,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = bothStages,
        },
    };

    const VkDescriptorSetLayoutCreateInfo sceneLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = sceneBindings,
    };

    if (vkCreateDescriptorSetLayout(context.device, &sceneLayoutInfo, nullptr,
                                    &sceneDescriptorSetLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create scene descriptor set layout\n";
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
    const VkDescriptorPoolSize poolSizes[2] = {
        {
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = pyramid_count + 1,
        },
        {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 2,
        },
    };

    const VkDescriptorPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = pyramid_count + 1,
        .poolSizeCount = 2,
        .pPoolSizes = poolSizes,
    };

    if (vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        std::cerr << "Failed to create application descriptor pool\n";
        return false;
    }

    if (!createHostBuffer(sizeof(SceneUniforms),
                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                          sceneUniformBuffer)) {
        return false;
    }

    if (!createHostBuffer(sizeof(GpuPointLight) * max_point_lights,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          pointLightBuffer)) {
        return false;
    }

    if (!createHostBuffer(sizeof(GpuSpotLight) * max_spot_lights,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          spotLightBuffer)) {
        return false;
    }

    const VkDescriptorSetAllocateInfo sceneAllocateInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &sceneDescriptorSetLayout,
    };

    if (vkAllocateDescriptorSets(context.device, &sceneAllocateInfo,
                                 &sceneDescriptorSet) != VK_SUCCESS) {
        std::cerr << "Failed to allocate scene descriptor set\n";
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

    const VkDescriptorBufferInfo sceneBufferInfo = {
        .buffer = sceneUniformBuffer.buffer,
        .offset = 0,
        .range = sizeof(SceneUniforms),
    };

    const VkDescriptorBufferInfo pointBufferInfo = {
        .buffer = pointLightBuffer.buffer,
        .offset = 0,
        .range = pointLightBuffer.size,
    };

    const VkDescriptorBufferInfo spotBufferInfo = {
        .buffer = spotLightBuffer.buffer,
        .offset = 0,
        .range = spotLightBuffer.size,
    };

    const VkWriteDescriptorSet sceneWrites[3] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = sceneDescriptorSet,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &sceneBufferInfo,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = sceneDescriptorSet,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &pointBufferInfo,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = sceneDescriptorSet,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &spotBufferInfo,
        },
    };

    vkUpdateDescriptorSets(context.device, 3, sceneWrites, 0, nullptr);

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

bool createGraphicsPipeline(VkShaderModule vertexShader,
                            VkShaderModule fragmentShader,
                            VkPrimitiveTopology topology,
                            VkBool32 depthWrite,
                            VkCompareOp depthCompare,
                            VkPipeline& pipeline) {
    const VkPipelineShaderStageCreateInfo stages[] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertexShader,
            .pName = "main",
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragmentShader,
            .pName = "main",
        },
    };

    const VkVertexInputBindingDescription binding = {
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };

    const VkVertexInputAttributeDescription attributes[2] = {
        {
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, position),
        },
        {
            .location = 1,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, normal),
        },
    };

    const VkPipelineVertexInputStateCreateInfo vertexInput = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = 2,
        .pVertexAttributeDescriptions = attributes,
    };

    const VkPipelineInputAssemblyStateCreateInfo inputAssembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = topology,
    };

    const VkPipelineViewportStateCreateInfo viewportState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };

    const VkPipelineRasterizationStateCreateInfo raster = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f,
    };

    const VkPipelineMultisampleStateCreateInfo multisampleState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    const VkPipelineDepthStencilStateCreateInfo depth = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = depthWrite,
        .depthCompareOp = depthCompare,
    };

    VkPipelineColorBlendAttachmentState blendAttachment{};
    const VkPipelineColorBlendStateCreateInfo blend = makeBlendState(blendAttachment);

    const VkDynamicState dynamicStates[2] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    const VkPipelineDynamicStateCreateInfo dynamicState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = dynamicStates,
    };

    const VkGraphicsPipelineCreateInfo pipelineInfo = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = stages,
        .pVertexInputState = &vertexInput,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &raster,
        .pMultisampleState = &multisampleState,
        .pDepthStencilState = &depth,
        .pColorBlendState = &blend,
        .pDynamicState = &dynamicState,
        .layout = pipelineLayout,
        .renderPass = context.render_pass,
    };

    if (vkCreateGraphicsPipelines(context.device, nullptr, 1,
                                  &pipelineInfo, nullptr,
                                  &pipeline) != VK_SUCCESS) {
        std::cerr << "Failed to create graphics pipeline\n";
        return false;
    }

    return true;
}

bool createPipelines() {
    const VkDescriptorSetLayout setLayouts[2] = {
        sceneDescriptorSetLayout,
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
        !loadShaderModule("shaders/edge.frag.spv", edgeFragmentShader) ||
        !loadShaderModule("shaders/marker.vert.spv", markerVertexShader) ||
        !loadShaderModule("shaders/marker.frag.spv", markerFragmentShader) ||
        !loadShaderModule("shaders/cone.vert.spv", coneVertexShader)) {
        return false;
    }

    if (!createGraphicsPipeline(pyramidVertexShader, pyramidFragmentShader,
                                VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                                VK_TRUE, VK_COMPARE_OP_LESS, fillPipeline)) {
        return false;
    }

    if (!createGraphicsPipeline(edgeVertexShader, edgeFragmentShader,
                                VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
                                VK_FALSE, VK_COMPARE_OP_LESS_OR_EQUAL, edgePipeline)) {
        return false;
    }

    if (!createGraphicsPipeline(markerVertexShader, markerFragmentShader,
                                VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                                VK_TRUE, VK_COMPARE_OP_LESS, markerPipeline)) {
        return false;
    }

    if (!createGraphicsPipeline(coneVertexShader, markerFragmentShader,
                                VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
                                VK_FALSE, VK_COMPARE_OP_LESS_OR_EQUAL, conePipeline)) {
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

Mat4 makeViewMatrix() {
    return lab_math::viewMatrix(
        cameraPosition,
        lab_math::radians(cameraYawDegrees),
        lab_math::radians(cameraPitchDegrees),
        lab_math::radians(cameraRollDegrees));
}

void moveCamera(const Vec3& localAxis, float distance) {
    const Mat4 rotation = lab_math::cameraRotation(
        lab_math::radians(cameraYawDegrees),
        lab_math::radians(cameraPitchDegrees),
        lab_math::radians(cameraRollDegrees));

    const Vec3 worldAxis = lab_math::transformDirection(rotation, localAxis);
    cameraPosition = lab_math::add(cameraPosition, lab_math::scale(worldAxis, distance));
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

    if (ImGui::SliderFloat("Far", &farPlane, 0.05f, 200.0f, "%.2f")) {
        farPlane = std::max(farPlane, nearPlane + 0.01f);
        farPlane = std::min(farPlane, 200.0f);
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

    // ---- НОВОЕ: параметры материала Блинна–Фонга ----
    ImGui::Separator();
    ImGui::Text("Material (Blinn-Phong)");

    // Это тот же цвет, что был "Pyramid color" в лабе №1; теперь он — альбедо Kd.
    ImGui::ColorEdit3("Albedo (Kd)", pyramid.color);
    ImGui::ColorEdit3("Specular color (Ks)", pyramid.specularColor);
    ImGui::SliderFloat("Shininess (alpha)", &pyramid.shininess,
                       1.0f, 256.0f, "%.1f", ImGuiSliderFlags_Logarithmic);

    ImGui::Separator();

    ImGui::Text("Base sides: %u", pyramid.baseSides);
    ImGui::Text("Edges: %u", pyramid.mesh.edgeIndexCount / 2);

    ImGui::TreePop();
}

void drawPyramidsWindow() {
    ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(340.0f, 430.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("Vulkan Lab 1 - Pyramids");

    drawAnimationUI();

    if (ImGui::CollapsingHeader("Heatmap tint (Lab 1)")) {
        // 0 — тепловая карта выключена (видно только освещение и материал),
        // 1 — полностью как в лабе №1.
        ImGui::SliderFloat("Heatmap strength", &heatmapStrength, 0.0f, 1.0f, "%.2f");
    }

    if (ImGui::CollapsingHeader("Pyramids", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (Pyramid& pyramid : pyramids) {
            drawPyramidUI(pyramid);
        }
    }

    ImGui::End();
}

void drawCameraWindow() {
    ImGui::SetNextWindowPos(ImVec2(10.0f, 450.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(340.0f, 260.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("Vulkan Lab 2 - Camera");

    if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragFloat3("Position", &cameraPosition.x, 0.05f, -100.0f, 100.0f, "%.2f");

        ImGui::DragFloat("Yaw (around Y)", &cameraYawDegrees, 0.5f, -360.0f, 360.0f, "%.1f deg");
        ImGui::DragFloat("Pitch (around X)", &cameraPitchDegrees, 0.5f, -360.0f, 360.0f, "%.1f deg");
        ImGui::DragFloat("Roll (around Z)", &cameraRollDegrees, 0.5f, -360.0f, 360.0f, "%.1f deg");

        if (ImGui::Button("Reset camera")) {
            cameraPosition = defaultCameraPosition;
            cameraYawDegrees = 0.0f;
            cameraPitchDegrees = 0.0f;
            cameraRollDegrees = 0.0f;
        }
    }

    if (ImGui::CollapsingHeader("Move relative to camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Step", &cameraMoveStep, 0.01f, 2.0f, "%.2f");

        if (ImGui::Button("Forward")) { moveCamera({0.0f, 0.0f, 1.0f}, cameraMoveStep); }
        ImGui::SameLine();
        if (ImGui::Button("Back"))    { moveCamera({0.0f, 0.0f, 1.0f}, -cameraMoveStep); }
        ImGui::SameLine();
        if (ImGui::Button("Left"))    { moveCamera({1.0f, 0.0f, 0.0f}, -cameraMoveStep); }
        ImGui::SameLine();
        if (ImGui::Button("Right"))   { moveCamera({1.0f, 0.0f, 0.0f}, cameraMoveStep); }

        if (ImGui::Button("Up"))      { moveCamera({0.0f, 1.0f, 0.0f}, -cameraMoveStep); }
        ImGui::SameLine();
        if (ImGui::Button("Down"))    { moveCamera({0.0f, 1.0f, 0.0f}, cameraMoveStep); }
    }

    drawProjectionUI();

    ImGui::End();
}

void drawLightColorAndIntensity(float color[3], float& intensity) {
    ImGui::ColorEdit3("Color", color);
    ImGui::DragFloat("Intensity", &intensity, 0.1f, 0.0f, 1000.0f, "%.1f");
}

void drawPointLightsUI() {
    if (!ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    ImGui::Text("Count: %d / %u", static_cast<int>(pointLights.size()), max_point_lights);

    if (pointLights.size() >= max_point_lights) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button("Add point light")) {
        PointLight light;
        light.position = sceneCenter;
        pointLights.push_back(light);
    }
    if (pointLights.size() >= max_point_lights) {
        ImGui::EndDisabled();
    }

    int removeIndex = -1;

    for (int i = 0; i < static_cast<int>(pointLights.size()); ++i) {
        PointLight& light = pointLights[i];

        ImGui::PushID(i);

        char label[32];
        std::snprintf(label, sizeof(label), "Point light %d", i + 1);

        if (ImGui::TreeNode(label)) {
            ImGui::DragFloat3("Position", &light.position.x, 0.05f, -50.0f, 50.0f, "%.2f");
            drawLightColorAndIntensity(light.color, light.intensity);

            if (ImGui::Button("Remove")) {
                removeIndex = i;
            }

            ImGui::TreePop();
        }

        ImGui::PopID();
    }

    if (removeIndex >= 0) {
        pointLights.erase(pointLights.begin() + removeIndex);
    }
}

void drawSpotLightsUI() {
    if (!ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }

    ImGui::Text("Count: %d / %u", static_cast<int>(spotLights.size()), max_spot_lights);

    if (spotLights.size() >= max_spot_lights) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button("Add spot light")) {
        SpotLight light;
        light.position = sceneCenter;
        spotLights.push_back(light);
    }
    if (spotLights.size() >= max_spot_lights) {
        ImGui::EndDisabled();
    }

    int removeIndex = -1;

    for (int i = 0; i < static_cast<int>(spotLights.size()); ++i) {
        SpotLight& light = spotLights[i];

        ImGui::PushID(i);

        char label[32];
        std::snprintf(label, sizeof(label), "Spot light %d", i + 1);

        if (ImGui::TreeNode(label)) {
            ImGui::DragFloat3("Position", &light.position.x, 0.05f, -50.0f, 50.0f, "%.2f");

            ImGui::DragFloat("Yaw (around Y)", &light.yawDegrees, 0.5f, -360.0f, 360.0f, "%.1f deg");
            ImGui::DragFloat("Pitch (-90 = down)", &light.pitchDegrees, 0.5f, -90.0f, 90.0f, "%.1f deg");

            drawLightColorAndIntensity(light.color, light.intensity);

            if (ImGui::SliderFloat("Inner angle", &light.innerAngleDegrees, 0.0f, 89.0f, "%.1f deg")) {
                light.outerAngleDegrees = std::max(light.outerAngleDegrees, light.innerAngleDegrees);
            }
            if (ImGui::SliderFloat("Outer angle", &light.outerAngleDegrees, 1.0f, 89.0f, "%.1f deg")) {
                light.innerAngleDegrees = std::min(light.innerAngleDegrees, light.outerAngleDegrees);
            }

            if (ImGui::Button("Remove")) {
                removeIndex = i;
            }

            ImGui::TreePop();
        }

        ImGui::PopID();
    }

    if (removeIndex >= 0) {
        spotLights.erase(spotLights.begin() + removeIndex);
    }
}

void drawLightingWindow() {
    ImGui::SetNextWindowPos(ImVec2(930.0f, 10.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(340.0f, 700.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("Vulkan Lab 2 - Lighting");

    ImGui::TextWrapped("World axes: +X right, +Y DOWN, +Z forward (into the screen).");

    if (ImGui::CollapsingHeader("Ambient light", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Ambient color", ambientColor);
        ImGui::SliderFloat("Ambient intensity", &ambientIntensity, 0.0f, 1.0f, "%.3f");
    }

    if (ImGui::CollapsingHeader("Directional light")) {
        ImGui::Checkbox("Enabled", &directionalEnabled);
        ImGui::ColorEdit3("Directional color", directionalColor);
        ImGui::DragFloat("Directional intensity", &directionalIntensity, 0.01f, 0.0f, 10.0f, "%.2f");
        ImGui::DragFloat("Yaw (around Y)", &directionalYawDegrees, 0.5f, -360.0f, 360.0f, "%.1f deg");
        ImGui::DragFloat("Pitch (-90 = down)", &directionalPitchDegrees, 0.5f, -90.0f, 90.0f, "%.1f deg");

        const Vec3 d = lab_math::directionFromYawPitch(
            lab_math::radians(directionalYawDegrees),
            lab_math::radians(directionalPitchDegrees));
        ImGui::Text("Light travels along: (%.2f, %.2f, %.2f)", d.x, d.y, d.z);
    }

    drawPointLightsUI();
    drawSpotLightsUI();

    if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Show light markers (spheres)", &showLightMarkers);
        ImGui::Checkbox("Show spot light cones", &showSpotCones);
        ImGui::SliderFloat("Marker radius", &markerRadius, 0.02f, 1.0f, "%.2f");
        ImGui::SliderFloat("Cone length", &coneLength, 0.5f, 15.0f, "%.2f");
    }

    ImGui::End();
}

void drawUI() {
    drawPyramidsWindow();
    drawLightingWindow();
    drawCameraWindow();
}

void updateSceneUniform() {
    SceneUniforms scene{};

    const Mat4 viewProjection = lab_math::multiply(makeProjectionMatrix(), makeViewMatrix());
    lab_math::toColumnMajor(viewProjection, scene.viewProjection);

    scene.cameraPosition[0] = cameraPosition.x;
    scene.cameraPosition[1] = cameraPosition.y;
    scene.cameraPosition[2] = cameraPosition.z;

    const Vec3 dir = lab_math::directionFromYawPitch(
        lab_math::radians(directionalYawDegrees),
        lab_math::radians(directionalPitchDegrees));
    scene.directionalDirection[0] = dir.x;
    scene.directionalDirection[1] = dir.y;
    scene.directionalDirection[2] = dir.z;

    const float directionalScale = directionalEnabled ? directionalIntensity : 0.0f;
    for (int i = 0; i < 3; ++i) {
        scene.directionalColor[i] = directionalColor[i] * directionalScale;
        scene.ambientLight[i] = ambientColor[i] * ambientIntensity;
    }

    scene.pointLightCount = static_cast<uint32_t>(pointLights.size());
    scene.spotLightCount = static_cast<uint32_t>(spotLights.size());

    scene.heatmapStrength = heatmapStrength;
    scene.markerRadius = markerRadius;
    scene.coneLength = coneLength;

    scene.heatmapBounds[0] = heatmapLeft;
    scene.heatmapBounds[1] = heatmapRight;
    scene.heatmapBounds[2] = heatmapTop;
    scene.heatmapBounds[3] = heatmapBottom;

    std::memcpy(sceneUniformBuffer.mapped, &scene, sizeof(scene));
}

void updateLightBuffers() {
    GpuPointLight* gpuPoints = static_cast<GpuPointLight*>(pointLightBuffer.mapped);

    for (size_t i = 0; i < pointLights.size(); ++i) {
        const PointLight& light = pointLights[i];

        GpuPointLight gpu{};
        gpu.position[0] = light.position.x;
        gpu.position[1] = light.position.y;
        gpu.position[2] = light.position.z;
        gpu.intensity = light.intensity;
        gpu.color[0] = light.color[0];
        gpu.color[1] = light.color[1];
        gpu.color[2] = light.color[2];

        gpuPoints[i] = gpu;
    }

    GpuSpotLight* gpuSpots = static_cast<GpuSpotLight*>(spotLightBuffer.mapped);

    for (size_t i = 0; i < spotLights.size(); ++i) {
        const SpotLight& light = spotLights[i];

        const Vec3 dir = lab_math::directionFromYawPitch(
            lab_math::radians(light.yawDegrees),
            lab_math::radians(light.pitchDegrees));

        GpuSpotLight gpu{};
        gpu.position[0] = light.position.x;
        gpu.position[1] = light.position.y;
        gpu.position[2] = light.position.z;
        gpu.intensity = light.intensity;
        gpu.color[0] = light.color[0];
        gpu.color[1] = light.color[1];
        gpu.color[2] = light.color[2];
        gpu.direction[0] = dir.x;
        gpu.direction[1] = dir.y;
        gpu.direction[2] = dir.z;

        // В шейдер передаём КОСИНУСЫ углов — так не нужно вызывать cos() для
        // каждого пикселя (в шейдере сравниваем напрямую с dot(...)).
        gpu.cosInner = std::cos(lab_math::radians(light.innerAngleDegrees));
        gpu.cosOuter = std::cos(lab_math::radians(light.outerAngleDegrees));

        gpuSpots[i] = gpu;
    }
}

void updateObjectUniform(Pyramid& pyramid) {
    ObjectUniforms object{};

    pyramid.modelMatrix = makeModelMatrix(pyramid);
    lab_math::toColumnMajor(pyramid.modelMatrix, object.model);

    object.albedo[0] = pyramid.color[0];
    object.albedo[1] = pyramid.color[1];
    object.albedo[2] = pyramid.color[2];

    object.specular[0] = pyramid.specularColor[0];
    object.specular[1] = pyramid.specularColor[1];
    object.specular[2] = pyramid.specularColor[2];

    object.shininess = pyramid.shininess;

    std::memcpy(pyramid.uniformBuffer.mapped, &object, sizeof(object));
}

void drawPyramid(const graphics::internal::FrameData& fd, Pyramid& pyramid) {
    const VkDeviceSize vertexBufferOffset = 0;

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

void drawLightVisualization(const graphics::internal::FrameData& fd) {
    const VkDeviceSize vertexBufferOffset = 0;

    const uint32_t pointCount = static_cast<uint32_t>(pointLights.size());
    const uint32_t spotCount = static_cast<uint32_t>(spotLights.size());

    if (showLightMarkers && pointCount + spotCount > 0) {
        vkCmdBindPipeline(fd.command_buffer,
                          VK_PIPELINE_BIND_POINT_GRAPHICS,
                          markerPipeline);

        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1,
                               &sphereMesh.vertexBuffer.buffer,
                               &vertexBufferOffset);

        vkCmdBindIndexBuffer(fd.command_buffer,
                             sphereMesh.indexBuffer.buffer,
                             0,
                             VK_INDEX_TYPE_UINT32);

        vkCmdDrawIndexed(fd.command_buffer,
                         sphereMesh.triangleIndexCount,
                         pointCount + spotCount,
                         0, 0, 0);
    }

    if (showSpotCones && spotCount > 0) {
        vkCmdBindPipeline(fd.command_buffer,
                          VK_PIPELINE_BIND_POINT_GRAPHICS,
                          conePipeline);

        vkCmdBindVertexBuffers(fd.command_buffer, 0, 1,
                               &coneMesh.vertexBuffer.buffer,
                               &vertexBufferOffset);

        vkCmdBindIndexBuffer(fd.command_buffer,
                             coneMesh.indexBuffer.buffer,
                             0,
                             VK_INDEX_TYPE_UINT32);

        vkCmdDrawIndexed(fd.command_buffer,
                         coneMesh.edgeIndexCount,
                         spotCount,
                         coneMesh.edgeFirstIndex,
                         0, 0);
    }
}

void addPointLight(float x, float y, float z,
                   float r, float g, float b,
                   float intensity) {
    PointLight light;
    light.position = {x, y, z};
    light.color[0] = r;
    light.color[1] = g;
    light.color[2] = b;
    light.intensity = intensity;
    pointLights.push_back(light);
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

    addPointLight(-5.0f, 0.0f, 4.5f, 1.0f, 0.55f, 0.40f, 8.0f);
    addPointLight( 5.0f, 0.0f, 4.5f, 0.45f, 0.70f, 1.0f, 8.0f);
    addPointLight(-5.0f, 0.0f, 9.0f, 0.60f, 1.0f, 0.60f, 8.0f);
    addPointLight( 5.0f, 0.0f, 9.0f, 1.0f, 0.90f, 0.50f, 8.0f);

    SpotLight lamp;
    lamp.position = {0.0f, -3.5f, 6.5f};
    lamp.yawDegrees = 0.0f;
    lamp.pitchDegrees = -90.0f;
    lamp.color[0] = 1.0f;
    lamp.color[1] = 1.0f;
    lamp.color[2] = 1.0f;
    lamp.intensity = 15.0f;
    lamp.innerAngleDegrees = 25.0f;
    lamp.outerAngleDegrees = 40.0f;
    spotLights.push_back(lamp);

    for (Pyramid& pyramid : pyramids) {
        if (!createMesh(pyramid)) {
            std::cerr << "Failed to create mesh for " << pyramid.name << '\n';
            return false;
        }
    }

    if (!createMarkerMeshes()) {
        std::cerr << "Failed to create light marker meshes\n";
        return false;
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
        destroyBuffer(pyramid.uniformBuffer);
        destroyBuffer(pyramid.mesh.vertexBuffer);
        destroyBuffer(pyramid.mesh.indexBuffer);
    }

    destroyBuffer(sphereMesh.vertexBuffer);
    destroyBuffer(sphereMesh.indexBuffer);
    destroyBuffer(coneMesh.vertexBuffer);
    destroyBuffer(coneMesh.indexBuffer);

    destroyBuffer(sceneUniformBuffer);
    destroyBuffer(pointLightBuffer);
    destroyBuffer(spotLightBuffer);

    vkDestroyPipeline(context.device, fillPipeline, nullptr);
    vkDestroyPipeline(context.device, edgePipeline, nullptr);
    vkDestroyPipeline(context.device, markerPipeline, nullptr);
    vkDestroyPipeline(context.device, conePipeline, nullptr);

    vkDestroyShaderModule(context.device, pyramidVertexShader, nullptr);
    vkDestroyShaderModule(context.device, pyramidFragmentShader, nullptr);
    vkDestroyShaderModule(context.device, edgeVertexShader, nullptr);
    vkDestroyShaderModule(context.device, edgeFragmentShader, nullptr);
    vkDestroyShaderModule(context.device, markerVertexShader, nullptr);
    vkDestroyShaderModule(context.device, markerFragmentShader, nullptr);
    vkDestroyShaderModule(context.device, coneVertexShader, nullptr);

    vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);

    vkDestroyDescriptorPool(context.device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(context.device, sceneDescriptorSetLayout, nullptr);
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
    updateSceneUniform();
    updateLightBuffers();
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

    vkCmdBindDescriptorSets(fd.command_buffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout,
                            0, 1,
                            &sceneDescriptorSet,
                            0, nullptr);

    for (Pyramid& pyramid : pyramids) {
        drawPyramid(fd, pyramid);
    }

    drawLightVisualization(fd);

    vkCmdEndRenderPass(fd.command_buffer);

    if (vkEndCommandBuffer(fd.command_buffer) != VK_SUCCESS) {
        std::cerr << "Failed to end application command buffer\n";
    }
}

} // namespace application
