#version 450 core

layout(std140, set = 0, binding = 0) uniform GlobalUniforms {
    mat4 projection;
    vec4 heatmapBounds; // xMin, xMax, yMin, yMax
} globalUniforms;

layout(std140, set = 1, binding = 0) uniform ObjectUniforms {
    mat4 model;
    vec4 baseColor;
} objectUniforms;

layout(location = 0) in vec3 inPosition;
layout(location = 0) out vec3 outHeatColor;

void main() {
    vec4 worldPosition = objectUniforms.model * vec4(inPosition, 1.0);

    float u = clamp(
        (worldPosition.x - globalUniforms.heatmapBounds.x) /
        (globalUniforms.heatmapBounds.y - globalUniforms.heatmapBounds.x),
        0.0, 1.0);

    float v = clamp(
        (worldPosition.y - globalUniforms.heatmapBounds.z) /
        (globalUniforms.heatmapBounds.w - globalUniforms.heatmapBounds.z),
        0.0, 1.0);

    vec3 topColor = mix(
        vec3(0.0, 0.0, 1.0),
        vec3(1.0, 1.0, 1.0),
        u);

    vec3 bottomColor = mix(
        vec3(0.0, 1.0, 0.0),
        vec3(1.0, 0.0, 0.0),
        u);

    outHeatColor = mix(topColor, bottomColor, v);

    gl_Position = globalUniforms.projection * worldPosition;
}
