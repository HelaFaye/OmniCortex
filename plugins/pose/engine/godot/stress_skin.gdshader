// OmniCortex joint stress on a skinned mesh (Godot 4): each vertex blends its bones' stress by its skin weights,
// painting stable tracking green, shear yellow and a flagged joint red. pose_receiver.gd fills `stress` per bone.
shader_type spatial;
render_mode blend_mix, cull_back;

uniform float stress[128];
uniform vec4 base_color : source_color = vec4(0.8, 0.8, 0.8, 1.0);
uniform float tint : hint_range(0.0, 1.0) = 0.85;
uniform float glow : hint_range(0.0, 4.0) = 1.2;
varying float v_stress;

vec3 ramp(float s) {
	s = clamp(s, 0.0, 1.0);
	return vec3(min(1.0, 2.0 * s), min(1.0, 2.0 * (1.0 - s)), 0.1);
}

void vertex() {
	v_stress = stress[BONE_INDICES.x] * BONE_WEIGHTS.x + stress[BONE_INDICES.y] * BONE_WEIGHTS.y
	         + stress[BONE_INDICES.z] * BONE_WEIGHTS.z + stress[BONE_INDICES.w] * BONE_WEIGHTS.w;
}

void fragment() {
	vec3 c = ramp(v_stress);
	ALBEDO = mix(base_color.rgb, c, tint);
	EMISSION = c * glow * v_stress;
}
