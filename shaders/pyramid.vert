#version 450 core

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4  view_projection;
    vec3  camera_position;
    vec3  directional_light_direction;
    vec3  directional_light_color;
    vec3  ambient_light;
    uint  point_light_count;
    uint  spot_light_count;
    float heatmap_strength;
    float marker_radius;
    float cone_length;
    vec4  heatmap_bounds;
} scene_uniforms;

layout(std140, set = 1, binding = 0) uniform ObjectUniforms {
    mat4  model;
    vec3  albedo;
    vec3  specular;
    float shininess;
} objectUniforms;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 outWorldPosition;
layout(location = 1) out vec3 outWorldNormal;
layout(location = 2) out vec3 outHeatColor;

void main() {
    vec4 worldPosition = objectUniforms.model * vec4(inPosition, 1.0);
    outWorldPosition = worldPosition.xyz;

    outWorldNormal = transpose(inverse(mat3(objectUniforms.model))) * inNormal;

    float u = clamp(
        (worldPosition.x - scene_uniforms.heatmap_bounds.x) /
        (scene_uniforms.heatmap_bounds.y - scene_uniforms.heatmap_bounds.x),
        0.0, 1.0);

    float v = clamp(
        (worldPosition.y - scene_uniforms.heatmap_bounds.z) /
        (scene_uniforms.heatmap_bounds.w - scene_uniforms.heatmap_bounds.z),
        0.0, 1.0);

    vec3 topColor = mix(vec3(0.0, 0.0, 1.0), vec3(1.0, 1.0, 1.0), u);
    vec3 bottomColor = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), u);

    outHeatColor = mix(topColor, bottomColor, v);

    gl_Position = scene_uniforms.view_projection * worldPosition;
}
