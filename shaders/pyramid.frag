#version 450 core

layout(std140, set = 1, binding = 0) uniform ObjectUniforms {
    mat4 model;
    vec4 baseColor;
} objectUniforms;

layout(location = 0) in vec3 inHeatColor;
layout(location = 0) out vec4 outColor;

void main() {
    vec3 finalColor = objectUniforms.baseColor.rgb * inHeatColor;

    outColor = vec4(finalColor, objectUniforms.baseColor.a);
}
