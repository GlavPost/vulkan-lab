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

struct PointLight {
    vec3  position;
    float intensity;
    vec3  color;
    float padding;
};

struct SpotLight {
    vec3  position;
    float intensity;
    vec3  color;
    float cos_inner;
    vec3  direction;
    float cos_outer;
};

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight spot_lights[];
};

layout(location = 0) in vec3 inPosition;

layout(location = 0) out vec3 outColor;

void main() {
    uint index = uint(gl_InstanceIndex);

    vec3 center;
    vec3 color;

    if (index < scene_uniforms.point_light_count) {
        center = point_lights[index].position;
        color  = point_lights[index].color;
    } else {
        uint spot_index = index - scene_uniforms.point_light_count;
        center = spot_lights[spot_index].position;
        color  = spot_lights[spot_index].color;
    }

    vec3 world_position = center + inPosition * scene_uniforms.marker_radius;

    gl_Position = scene_uniforms.view_projection * vec4(world_position, 1.0);

    outColor = color;
}
