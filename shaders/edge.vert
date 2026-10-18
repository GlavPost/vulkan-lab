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

void main() {
    gl_Position = scene_uniforms.view_projection *
                  objectUniforms.model *
                  vec4(inPosition, 1.0);
}
