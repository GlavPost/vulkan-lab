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

struct SpotLight {
    vec3  position;
    float intensity;
    vec3  color;
    float cos_inner;
    vec3  direction;
    float cos_outer;
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight spot_lights[];
};

layout(location = 0) in vec3 inPosition;

layout(location = 0) out vec3 outColor;

void main() {
    SpotLight spot = spot_lights[gl_InstanceIndex];

    vec3 d = normalize(spot.direction);
    vec3 helper = (abs(d.y) < 0.99) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 right = normalize(cross(helper, d));
    vec3 up    = cross(d, right);

    float sin_outer = sqrt(max(1.0 - spot.cos_outer * spot.cos_outer, 0.0));

    vec3 local = vec3(inPosition.x * sin_outer,
                      inPosition.y * sin_outer,
                      inPosition.z * spot.cos_outer) * scene_uniforms.cone_length;

    vec3 world_position = spot.position + right * local.x + up * local.y + d * local.z;

    gl_Position = scene_uniforms.view_projection * vec4(world_position, 1.0);

    outColor = vec3(1.0);
}
