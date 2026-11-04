// X-ray pass for a character (Godot 4): set as the material's next_pass. Drawn without the depth test, so the
// character's flagged joints (by stress, as in stress_skin.gdshader) show through whatever stands in front.
shader_type spatial;
render_mode unshaded, depth_test_disabled, cull_front, blend_add;

uniform float stress[128];
uniform float strength : hint_range(0.0, 1.0) = 0.5;
varying float v_stress;

void vertex() {
	v_stress = stress[BONE_INDICES.x] * BONE_WEIGHTS.x + stress[BONE_INDICES.y] * BONE_WEIGHTS.y
	         + stress[BONE_INDICES.z] * BONE_WEIGHTS.z + stress[BONE_INDICES.w] * BONE_WEIGHTS.w;
}

void fragment() {
	float s = clamp(v_stress, 0.0, 1.0);
	ALBEDO = vec3(min(1.0, 2.0 * s), min(1.0, 2.0 * (1.0 - s)), 0.1);
	ALPHA = strength * (0.2 + 0.8 * s);
}
