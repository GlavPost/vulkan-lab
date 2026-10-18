#version 450 core

layout(location = 0) in vec3 inWorldPosition;
layout(location = 1) in vec3 inWorldNormal;
layout(location = 2) in vec3 inHeatColor;

layout(location = 0) out vec4 outColor;

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


struct PointLight {
    vec3  position;
    float intensity;
    vec3  color;
    float _padding;
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

vec3 blinn_phong(vec3 n, vec3 v, vec3 l, vec3 E, vec3 Kd) {
    float cos_theta = max(dot(n, l), 0.0);
    if (cos_theta == 0.0) {
        return vec3(0.0);
    }

    vec3  h       = normalize(l + v);
    float cos_phi = max(dot(n, h), 0.0);
    float spec    = pow(cos_phi, objectUniforms.shininess);

    return E * (cos_theta * Kd + spec * objectUniforms.specular);
}

void main() {
    vec3 n = normalize(inWorldNormal);

    vec3 v = normalize(scene_uniforms.camera_position - inWorldPosition);

    vec3 Kd = objectUniforms.albedo *
              mix(vec3(1.0), inHeatColor, scene_uniforms.heatmap_strength);

    vec3 color = scene_uniforms.ambient_light * Kd;

    vec3 l_dir = -normalize(scene_uniforms.directional_light_direction);
    color += blinn_phong(n, v, l_dir, scene_uniforms.directional_light_color, Kd);

    for (uint i = 0u; i < scene_uniforms.point_light_count; ++i) {
        vec3  to_light = point_lights[i].position - inWorldPosition;
        float dist2 = max(dot(to_light, to_light), 1e-4);
        vec3  l     = to_light * inversesqrt(dist2);

        vec3 E = point_lights[i].color * point_lights[i].intensity / dist2;

        color += blinn_phong(n, v, l, E, Kd);
    }

    for (uint i = 0u; i < scene_uniforms.spot_light_count; ++i) {
        vec3  to_light = spot_lights[i].position - inWorldPosition;
        float dist2    = max(dot(to_light, to_light), 1e-4);
        vec3  l        = to_light * inversesqrt(dist2);

        float cos_gamma = dot(-l, spot_lights[i].direction);

        float cone;
        if (spot_lights[i].cos_inner - spot_lights[i].cos_outer > 1e-5) {
            cone = smoothstep(spot_lights[i].cos_outer,
                              spot_lights[i].cos_inner,
                              cos_gamma);
        } else {
            cone = step(spot_lights[i].cos_outer, cos_gamma);
        }

        if (cone > 0.0) {
            vec3 E = spot_lights[i].color * spot_lights[i].intensity * cone / dist2;
            color += blinn_phong(n, v, l, E, Kd);
        }
    }

    outColor = vec4(color, 1.0);
}
