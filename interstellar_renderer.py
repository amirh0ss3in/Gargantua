import taichi as ti
import numpy as np
import math
import time
import os

@ti.data_oriented
class InterstellarRenderer:
    def __init__(self, width=1200, show_gui=True, save_video_path=None, video_fps=24):
        ti.init(arch=ti.cuda, default_fp=ti.f32)

        self.show_gui = show_gui
        self.save_video = save_video_path is not None
        
        self.ASPECT_RATIO = 2.35 / 1.0
        self.WIDTH = width
        self.HEIGHT = int(self.WIDTH / self.ASPECT_RATIO)
        self.RESOLUTION = (self.WIDTH, self.HEIGHT)
        self.GM = 0.5
        self.SCHWARZSCHILD_RADIUS = 2.0 * self.GM
        self.HORIZON_RADIUS = self.SCHWARZSCHILD_RADIUS
        self.DISK_INNER_RADIUS = self.SCHWARZSCHILD_RADIUS * 2.5
        self.DISK_OUTER_RADIUS = self.SCHWARZSCHILD_RADIUS * 12.0
        self.DISK_HALF_THICKNESS = (self.DISK_OUTER_RADIUS - self.DISK_INNER_RADIUS) * 0.6
        self.SUN_RADIUS = 3.0
        self.SUN1_POS = ti.Vector([60.0, 10.0, -25.0])
        self.SUN1_COLOR = ti.Vector([1.0, 0.6, 0.3])
        self.SUN2_POS = ti.Vector([-50.0, -15.0, 35.0])
        self.SUN2_COLOR = ti.Vector([0.6, 0.7, 1.0])
        self.SUN_NOISE_SCALE_1 = 2.0
        self.SUN_NOISE_SCALE_2 = 8.0
        self.SUN_NOISE_CONTRAST = 10.0
        self.SUN_BRIGHTNESS_BOOST = 2.0
        self.SUN_CORE_BRIGHTNESS = 10.0
        self.SUN_GLOW_STRENGTH = 0.1
        self.SUN_GLOW_FALLOFF = 0.2
        self.SUN_RAY_NOISE_SCALE = 0.9
        self.SUN_RAY_STRETCH = 500.0
        self.SUN_RAY_CONTRAST = 9.0
        self.MAX_STEPS = 1000
        self.TOLERANCE, self.DT_INITIAL, self.DT_MIN, self.DT_MAX = 1.0e-5, 0.5, 1.0e-4, 1.0
        self.SAFETY_FACTOR = 0.9
        self.FAR_FIELD_RADIUS = 75.0
        self.BLOOM_THRESHOLD, self.BLOOM_STRENGTH = 0.4, 0.05
        self.ANAMORPHIC_FLARE_STRENGTH = 0.1
        self.VIGNETTE_STRENGTH, self.GRAIN_INTENSITY = 0.2, 0.015
        self.CUBE_SIZE = self.DISK_OUTER_RADIUS * 2.5
        self.CUBE_CENTER = ti.Vector([0.0, 0.0, 0.0])
        self.GRID_RES = 200
        self.GRID_DX = 1.0 / self.GRID_RES
        self.GRID_INV_DX = float(self.GRID_RES)
        self.VOLUME_SUBSTEPS = 2
        self.EMISSION_STRENGTH = 200.0
        self.ABSORPTION_COEFFICIENT = 25.0
        self.DENSITY_MULTIPLIER = 1.5
        self.DENSITY_POW = 5.0
        self.SCATTERING_STRENGTH = 60.0
        self.HG_ASYMMETRY_FACTOR = 0.4
        self.SELF_SHADOW_STRENGTH = 0.8
        self.DISK_COLOR_HOT = ti.Vector([1.0, 0.9, 0.7])
        self.DISK_COLOR_COLD = ti.Vector([1.0, 0.5, 0.2])
        self.DOPPLER_STRENGTH = 4.0
        self.BASE_STRUCTURE_SCALE = 0.9
        self.WARP_FIELD_SCALE = 0.9
        self.WARP_STRENGTH = 1.8
        self.FILAMENT_NOISE_SCALE = 1.2
        self.FILAMENT_CONTRAST = 3.0
        self.DETAIL_NOISE_SCALE = 60.0
        self.DETAIL_STRENGTH = 0.5
        self.DISK_NOISE_STRENGTH = 1.5
        self.VELOCITY_TURBULENCE_STRENGTH = 0.2
        self.DT_SIM = 0.02
        self.JACOBI_ITERATIONS = 10
        self.DISSIPATION = 0.03
        self.FORCE_RADIUS = self.GRID_RES / 8.0
        self.FORCE_STRENGTH = 80.0
        self.ORBITAL_ASSIST_STRENGTH = 0.03
        self.ORBITAL_VELOCITY_SCALE = 10
        self.GRAVITY_STRENGTH = 1.0
        self.CONFINEMENT_FALLOFF = self.DISK_HALF_THICKNESS * 2.5
        self.ADVECTION_STRENGTH = 0.0000001
        self.MAX_VELOCITY = 2.0 * self.GRID_RES
        
        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=self.RESOLUTION)
        skybox_img_np = ti.tools.image.imread('starmap.jpg').astype(np.float32) / 255.0
        self.skybox_texture = ti.Vector.field(3, dtype=ti.f32, shape=skybox_img_np.shape[:2])
        self.skybox_texture.from_numpy(skybox_img_np)
        self.density_field = ti.field(dtype=ti.f32, shape=(self.GRID_RES,) * 3)
        self.new_density_field = ti.field(dtype=ti.f32, shape=(self.GRID_RES,) * 3)
        self.velocity_field = ti.Vector.field(3, dtype=ti.f32, shape=(self.GRID_RES,) * 3, layout=ti.Layout.SOA)
        self.new_velocity_field = ti.Vector.field(3, dtype=ti.f32, shape=(self.GRID_RES,) * 3, layout=ti.Layout.SOA)
        self.pressure_field = ti.field(dtype=ti.f32, shape=(self.GRID_RES,) * 3)
        self.divergence_field = ti.field(dtype=ti.f32, shape=(self.GRID_RES,) * 3)
        self.cam_to_world_matrix = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())

        self.gui = None
        if self.show_gui:
            self.gui = ti.GUI("Cinematic Black Hole Flight", res=self.RESOLUTION, fast_gui=True)

        self.video_manager = None
        if self.save_video:
            self.temp_video_dir = "temp_video_frames"
            os.makedirs(self.temp_video_dir, exist_ok=True)
            
            self.video_manager = ti.tools.VideoManager(
                output_dir=self.temp_video_dir, # Use the stored path
                framerate=video_fps,
                automatic_build=False
            )
            self.final_video_path = save_video_path


        self.reset_scene()
        print("Renderer Initialized. Gas simulation ready.")

    def reset_scene(self):
        print("Resetting gas simulation to initial state...")
        self.init_scene()
        self.init_velocity()

    def step(self, cam_pos_vec, cam_fwd_vec, cam_up_vec, fov):
        self.simulation_step(cam_fwd_vec, 0)
        fwd = cam_fwd_vec.normalized()
        up = cam_up_vec.normalized()
        right = fwd.cross(up).normalized()
        up = right.cross(fwd)
        self.cam_to_world_matrix[None] = ti.Matrix.cols([right, up, fwd])
        self.render(cam_pos_vec, self.cam_to_world_matrix[None], fov)
        
        if self.show_gui and self.gui:
            self.gui.set_image(self.pixels)
            self.gui.show()
        
        if self.save_video and self.video_manager:
            frame_img = self.pixels.to_numpy()
            self.video_manager.write_frame(frame_img)

    def close(self):
        """Finalizes and closes resources, with a robust retry mechanism for saving the video."""
        if self.video_manager:
            print("Finalizing video...")
            self.video_manager.make_video(gif=False, mp4=True)

            default_output = os.path.join(self.temp_video_dir, "video.mp4")
            
            max_retries = 10  # Try for up to 5 seconds
            retry_delay = 0.5 # Wait half a second between tries

            for attempt in range(max_retries):
                try:
                    # Attempt to move the file
                    if os.path.exists(default_output):
                        # Use os.replace for better cross-platform compatibility
                        # It will overwrite the destination if it already exists.
                        os.replace(default_output, self.final_video_path)
                        print(f"Video saved to {self.final_video_path}")
                        # If successful, break out of the loop
                        break
                    else:
                        # If the file doesn't even exist, ffmpeg likely failed.
                        print(f"Error: ffmpeg did not generate a video file. Is ffmpeg installed and in your PATH?")
                        break

                except PermissionError:
                    if attempt < max_retries - 1:
                        print(f"Video file is locked by another process (ffmpeg), retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        # If all retries fail, print an informative error.
                        print("="*50)
                        print("FATAL ERROR: Could not move the video file after multiple retries.")
                        print(f"The video may be saved at: {default_output}")
                        print("The file is likely still locked by a hanging ffmpeg process.")
                        print("="*50)
                        # Re-raise the exception to halt the program.
                        raise

        if self.gui:
            self.gui.close()
        print("Renderer closed.")
        
    @ti.func
    def sample_texture(self, texture: ti.template(), u: ti.f32, v: ti.f32): #type: ignore
        shape = ti.Vector([texture.shape[1], texture.shape[0]])
        p = ti.Vector([u, v]) * shape; i = ti.cast(ti.floor(p), ti.i32); f = ti.math.fract(p)
        i0 = ti.max(0, ti.min(shape - 1, i)); i1 = ti.max(0, ti.min(shape - 1, i + 1))
        c00 = texture[i0.y, i0.x]; c10 = texture[i0.y, i1.x]; c01 = texture[i1.y, i0.x]; c11 = texture[i1.y, i1.x]
        return ti.math.mix(ti.math.mix(c00, c10, f.x), ti.math.mix(c01, c11, f.x), f.y)

    @ti.func
    def get_background_color_from_skybox(self, ray_dir):
        phi = ti.acos(ray_dir.y); theta = ti.atan2(ray_dir.z, ray_dir.x)
        u = (theta + math.pi) / (2.0 * math.pi); v = 1.0 - (phi / math.pi)
        return self.sample_texture(self.skybox_texture, u, v)

    @ti.func
    def hash31(self, p):
        p3 = ti.math.fract(p * ti.Vector([437.585453, 223.13306, 353.72935])); p3 += p3.dot(p3 + 19.19)
        return ti.math.fract((p3.x + p3.y) * p3.z)

    @ti.func
    def value_noise_3d(self, p):
        i = ti.floor(p); f = ti.math.fract(p); f = f * f * (3.0 - 2.0 * f)
        return ti.math.mix(ti.math.mix(ti.math.mix(self.hash31(i + ti.Vector([0, 0, 0])), self.hash31(i + ti.Vector([1, 0, 0])), f.x), ti.math.mix(self.hash31(i + ti.Vector([0, 1, 0])), self.hash31(i + ti.Vector([1, 1, 0])), f.x), f.y), ti.math.mix(ti.math.mix(self.hash31(i + ti.Vector([0, 0, 1])), self.hash31(i + ti.Vector([1, 0, 1])), f.x), ti.math.mix(self.hash31(i + ti.Vector([0, 1, 1])), self.hash31(i + ti.Vector([1, 1, 1])), f.x), f.y), f.z)

    @ti.func
    def fbm_ridged_3d(self, p):
        val, amp, freq = 0.0, 0.5, 1.0
        for _ in range(6):
            noise = self.value_noise_3d(p * freq); ridged_noise = 1.0 - abs(noise - 0.5) * 2.0
            val += amp * ridged_noise; amp *= 0.5; freq *= 2.0
        return val

    @ti.func
    def get_sun_surface_color(self, surface_pos, base_color):
        noise1 = self.fbm_ridged_3d(surface_pos * self.SUN_NOISE_SCALE_1)
        noise2 = self.fbm_ridged_3d(surface_pos * self.SUN_NOISE_SCALE_2)
        combined_noise = noise1 * 0.7 + noise2 * 0.3
        final_noise = ti.pow(combined_noise, self.SUN_NOISE_CONTRAST)
        hot_color = ti.Vector([1.0, 1.0, 0.9])
        modulated_color = ti.math.mix(base_color, hot_color, final_noise)
        return modulated_color * (1.0 + final_noise * self.SUN_BRIGHTNESS_BOOST)

    @ti.func
    def sample_grid(self, field, pos_grid):
        p_floor = ti.floor(pos_grid); p_frac = pos_grid - p_floor; i, j, k = ti.cast(p_floor, ti.i32)
        i, j, k = ti.math.clamp(i, 0, self.GRID_RES - 2), ti.math.clamp(j, 0, self.GRID_RES - 2), ti.math.clamp(k, 0, self.GRID_RES - 2)
        v000=field[i,j,k];     v100=field[i+1,j,k];   v010=field[i,j+1,k];   v110=field[i+1,j+1,k]
        v001=field[i,j,k+1];   v101=field[i+1,j,k+1]; v011=field[i,j+1,k+1]; v111=field[i+1,j+1,k+1]
        v00=ti.math.mix(v000,v100,p_frac.x); v10=ti.math.mix(v010,v110,p_frac.x)
        v01=ti.math.mix(v001,v101,p_frac.x); v11=ti.math.mix(v011,v111,p_frac.x)
        v0=ti.math.mix(v00,v10,p_frac.y);   v1=ti.math.mix(v01,v11,p_frac.y)
        return ti.math.mix(v0, v1, p_frac.z)

    @ti.kernel
    def advect_density(self, field_in: ti.template(), field_out: ti.template()): #type:ignore
        for i, j, k in field_in:
            p = ti.Vector([float(i), float(j), float(k)]); vel_grid = self.velocity_field[i, j, k]
            p_prev = p - vel_grid * self.DT_SIM * self.ADVECTION_STRENGTH
            field_out[i, j, k] = self.sample_grid(field_in, p_prev)

    @ti.kernel
    def advect_velocity(self, field_in: ti.template(), field_out: ti.template()): #type:ignore
        for i, j, k in field_in:
            p = ti.Vector([float(i), float(j), float(k)]); vel_grid = field_in[i, j, k]
            p_prev = p - vel_grid * self.DT_SIM * self.ADVECTION_STRENGTH
            sampled_vel = self.sample_grid(field_in, p_prev)
            field_out[i, j, k] = ti.math.mix(sampled_vel, ti.Vector([0.0,0.0,0.0]), self.DISSIPATION)

    @ti.kernel
    def confine_density_to_disk(self):
        for i, j, k in self.density_field:
            pos_world = ((ti.Vector([i,j,k]) * self.GRID_DX) - 0.5) * self.CUBE_SIZE
            radius_xz = pos_world.xz.norm(); height_y = abs(pos_world.y)
            confinement_factor = ti.math.smoothstep(self.CONFINEMENT_FALLOFF, 0.0, height_y)
            if not (self.DISK_INNER_RADIUS < radius_xz < self.DISK_OUTER_RADIUS): confinement_factor = 0.0
            self.density_field[i, j, k] *= confinement_factor

    @ti.kernel
    def apply_boundary_conditions(self):
        for i, j, k in self.velocity_field:
            pos_world = ((ti.Vector([i,j,k]) * self.GRID_DX) - 0.5) * self.CUBE_SIZE
            if pos_world.xz.norm() < self.DISK_INNER_RADIUS:
                self.velocity_field[i, j, k] *= 0.0; self.density_field[i, j, k] *= 0.0

    @ti.kernel
    def apply_forces(self, cam_fwd: ti.types.vector(3, ti.f32), apply_user_force: ti.i32): #type: ignore
        for i, j, k in self.velocity_field:
            user_force = ti.Vector([0.0, 0.0, 0.0])
            if apply_user_force:
                dist = (ti.Vector([i,j,k]) - ti.Vector([self.GRID_RES/2, self.GRID_RES/2, self.GRID_RES/2])).norm()
                if dist < self.FORCE_RADIUS: user_force = cam_fwd * self.FORCE_STRENGTH * (1.0 - dist / self.FORCE_RADIUS)
            pos_world = (ti.Vector([i,j,k]) * self.GRID_DX - 0.5) * self.CUBE_SIZE
            orbital_assist_force = ti.Vector([0.0, 0.0, 0.0]); radius_xz = pos_world.xz.norm()
            if radius_xz > 0.1:
                ideal_speed = ti.sqrt(self.GM / (radius_xz + 0.1)); tangential_dir = ti.Vector([-pos_world.z, 0.0, pos_world.x]).normalized()
                ideal_velocity_world = ideal_speed * tangential_dir * self.ORBITAL_VELOCITY_SCALE
                ideal_velocity_grid = ideal_velocity_world * (self.GRID_INV_DX / self.CUBE_SIZE)
                correction_vec = ideal_velocity_grid - self.velocity_field[i, j, k]
                orbital_assist_force = correction_vec * self.ORBITAL_ASSIST_STRENGTH
            gravity_force = ti.Vector([0.0, 0.0, 0.0]); r_sqr = pos_world.norm_sqr()
            if r_sqr > 0.1:
                acceleration_magnitude = self.GRAVITY_STRENGTH * self.GM / r_sqr
                gravity_force = -pos_world.normalized() * acceleration_magnitude
            total_force = user_force + orbital_assist_force + gravity_force
            self.velocity_field[i, j, k] += total_force * self.DT_SIM

    @ti.kernel
    def clamp_velocity(self):
        for i, j, k in self.velocity_field:
            vel_sqr = self.velocity_field[i, j, k].norm_sqr()
            if vel_sqr > self.MAX_VELOCITY * self.MAX_VELOCITY:
                self.velocity_field[i, j, k] = self.velocity_field[i, j, k].normalized() * self.MAX_VELOCITY

    @ti.kernel
    def compute_divergence(self):
        divergence_scale = 0.5 * self.GRID_INV_DX
        for i, j, k in self.velocity_field:
            vl=self.velocity_field[i-1,j,k].x if i>0 else 0; vr=self.velocity_field[i+1,j,k].x if i<self.GRID_RES-1 else 0
            vb=self.velocity_field[i,j-1,k].y if j>0 else 0; vt=self.velocity_field[i,j+1,k].y if j<self.GRID_RES-1 else 0
            vz_b=self.velocity_field[i,j,k-1].z if k>0 else 0; vz_f=self.velocity_field[i,j,k+1].z if k<self.GRID_RES-1 else 0
            self.divergence_field[i, j, k] = divergence_scale * (vr - vl + vt - vb + vz_f - vz_b); self.pressure_field[i, j, k] = 0.0

    @ti.kernel
    def solve_pressure_red_black(self, is_red_pass: ti.i32): #type:ignore
        inv_h_sqr = 1.0 / (self.GRID_DX * self.GRID_DX)
        for i, j, k in self.pressure_field:
            if (i + j + k) % 2 == is_red_pass:
                pl=self.pressure_field[i-1,j,k] if i>0 else 0; pr=self.pressure_field[i+1,j,k] if i<self.GRID_RES-1 else 0
                pb=self.pressure_field[i,j-1,k] if j>0 else 0; pt=self.pressure_field[i,j+1,k] if j<self.GRID_RES-1 else 0
                p_zb=self.pressure_field[i,j,k-1] if k>0 else 0; p_zf=self.pressure_field[i,j,k+1] if k<self.GRID_RES-1 else 0
                self.pressure_field[i,j,k] = (pl + pr + pb + pt + p_zb + p_zf - self.divergence_field[i,j,k] * inv_h_sqr) / 6.0

    @ti.kernel
    def project(self):
        gradient_scale = 0.5 * self.GRID_INV_DX
        for i, j, k in self.velocity_field:
            pl=self.pressure_field[i-1,j,k] if i>0 else 0; pr=self.pressure_field[i+1,j,k] if i<self.GRID_RES-1 else 0
            pb=self.pressure_field[i,j-1,k] if j>0 else 0; pt=self.pressure_field[i,j+1,k] if j<self.GRID_RES-1 else 0
            p_zb=self.pressure_field[i,j,k-1] if k>0 else 0; p_zf=self.pressure_field[i,j,k+1] if k<self.GRID_RES-1 else 0
            grad = gradient_scale * ti.Vector([pr - pl, pt - pb, p_zf - p_zb]); self.velocity_field[i, j, k] -= grad

    @ti.kernel
    def copy_field(self, field_in: ti.template(), field_out: ti.template()): #type:ignore
        for I in ti.grouped(field_in): field_out[I] = field_in[I]

    def simulation_step(self, cam_fwd, apply_user_force):
        self.advect_velocity(self.velocity_field, self.new_velocity_field); self.advect_density(self.density_field, self.new_density_field)
        self.copy_field(self.new_velocity_field, self.velocity_field); self.copy_field(self.new_density_field, self.density_field)
        self.apply_forces(cam_fwd, apply_user_force); self.compute_divergence()
        for _ in range(self.JACOBI_ITERATIONS):
            self.solve_pressure_red_black(1); self.solve_pressure_red_black(0)
        self.project(); self.confine_density_to_disk(); self.apply_boundary_conditions(); self.clamp_velocity()

    @ti.kernel
    def init_scene(self):
        for i, j, k in self.density_field:
            pos_grid_centered = (ti.Vector([i,j,k]) * self.GRID_DX) - 0.5; pos_world = pos_grid_centered * self.CUBE_SIZE
            radius_xz = pos_world.xz.norm(); height_y = pos_world.y; base_density_canvas = 0.0
            if self.DISK_INNER_RADIUS < radius_xz < self.DISK_OUTER_RADIUS and abs(height_y) < self.DISK_HALF_THICKNESS:
                falloff_y = ti.math.smoothstep(self.DISK_HALF_THICKNESS, self.DISK_HALF_THICKNESS*0.8, abs(height_y))
                falloff_in = ti.math.smoothstep(self.DISK_INNER_RADIUS, self.DISK_INNER_RADIUS*1.2, radius_xz)
                falloff_out = ti.math.smoothstep(self.DISK_OUTER_RADIUS, self.DISK_OUTER_RADIUS*0.8, radius_xz)
                base_density_canvas = falloff_y * falloff_in * falloff_out
            warp_coords = pos_world * self.WARP_FIELD_SCALE
            warp_vector = ti.Vector([self.fbm_ridged_3d(warp_coords + ti.Vector([13.7, 0.0, 11.3])), self.fbm_ridged_3d(warp_coords + ti.Vector([0.0, 24.2, 7.5])), self.fbm_ridged_3d(warp_coords + ti.Vector([19.1, 9.8, 0.0]))])
            warp_vector = (warp_vector - 0.5) * 2.0 * self.WARP_STRENGTH; warped_pos = pos_world + warp_vector
            base_structure_noise = self.fbm_ridged_3d(pos_world * self.BASE_STRUCTURE_SCALE); base_structure_value = ti.pow(base_structure_noise, 1.5)
            filament_noise = self.fbm_ridged_3d(warped_pos * self.FILAMENT_NOISE_SCALE); filament_value = ti.pow(filament_noise, self.FILAMENT_CONTRAST)
            detail_noise_val = self.value_noise_3d(pos_world * self.DETAIL_NOISE_SCALE); detail_factor = ti.math.mix(1.0 - self.DETAIL_STRENGTH, 1.0 + self.DETAIL_STRENGTH, detail_noise_val)
            combined_noise = base_structure_value * filament_value * detail_factor; final_density = base_density_canvas * combined_noise * self.DISK_NOISE_STRENGTH
            self.density_field[i, j, k] = ti.max(0.0, final_density)

    @ti.kernel
    def init_velocity(self):
        world_to_grid_scale = self.GRID_INV_DX / self.CUBE_SIZE
        for i, j, k in self.velocity_field:
            pos_world = ((ti.Vector([i,j,k]) * self.GRID_DX) - 0.5) * self.CUBE_SIZE; radius_xz = pos_world.xz.norm()
            ideal_velocity_world = ti.Vector([0.0, 0.0, 0.0])
            if radius_xz > 0.1:
                tangential_dir = ti.Vector([-pos_world.z, 0, pos_world.x]).normalized()
                speed = ti.sqrt(self.GM / (radius_xz + 0.1)) * self.ORBITAL_VELOCITY_SCALE; ideal_velocity_world = speed * tangential_dir
            noise_vec = ti.Vector([self.value_noise_3d(pos_world*5.0+12.3), self.value_noise_3d(pos_world*5.0+45.6), self.value_noise_3d(pos_world*5.0+78.9)])
            turbulence_world = (noise_vec * 2.0 - 1.0) * self.VELOCITY_TURBULENCE_STRENGTH
            final_velocity_world = ideal_velocity_world + turbulence_world; self.velocity_field[i, j, k] = final_velocity_world * world_to_grid_scale

    @ti.func
    def get_acceleration_gr(self, pos, vel):
        r = pos.norm() + 1e-9; L_vec = pos.cross(vel); L2 = L_vec.dot(L_vec)
        gr_term = (3.0 * self.SCHWARZSCHILD_RADIUS * L2) / (2.0 * r**5); return -gr_term * pos

    @ti.func
    def dopri5_step(self, pos, vel, dt):
        c2, c3, c4, c5, c6, c7 = 1/5, 3/10, 4/5, 8/9, 1, 1; a21 = 1/5; a31, a32 = 3/40, 9/40; a41, a42, a43 = 44/45, -56/15, 32/9; a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729; a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656; a71, a73, a74, a75, a76 = 35/384, 500/1113, 125/192, -2187/6784, 11/84; b5_1, b5_3, b5_4, b5_5, b5_6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84; b4_1, b4_3, b4_4, b4_5, b4_6, b4_7 = 5179/57600, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40
        k1_pos = vel; k1_vel = self.get_acceleration_gr(pos, vel)
        k2_pos = vel + dt * a21 * k1_vel; k2_vel = self.get_acceleration_gr(pos + dt * a21 * k1_pos, k2_pos)
        k3_pos = vel + dt * (a31*k1_vel + a32*k2_vel); k3_vel = self.get_acceleration_gr(pos + dt * (a31*k1_pos + a32*k2_pos), k3_pos)
        k4_pos = vel + dt * (a41*k1_vel + a42*k2_vel + a43*k3_vel); k4_vel = self.get_acceleration_gr(pos + dt * (a41*k1_pos + a42*k2_pos + a43*k3_pos), k4_pos)
        k5_pos = vel + dt * (a51*k1_vel + a52*k2_vel + a53*k3_vel + a54*k4_vel); k5_vel = self.get_acceleration_gr(pos + dt * (a51*k1_pos + a52*k2_pos + a53*k3_pos + a54*k4_pos), k5_pos)
        k6_pos = vel + dt * (a61*k1_vel + a62*k2_vel + a63*k3_vel + a64*k4_vel + a65*k5_vel); k6_vel = self.get_acceleration_gr(pos + dt * (a61*k1_pos + a62*k2_pos + a63*k3_pos + a64*k4_pos + a65*k5_pos), k6_pos)
        k7_pos = vel + dt * (a71*k1_vel + a73*k3_vel + a74*k4_vel + a75*k5_vel + a76*k6_vel); k7_vel = self.get_acceleration_gr(pos + dt * (a71*k1_pos + a73*k3_pos + a74*k4_pos + a75*k5_pos + a76*k6_pos), k7_pos)
        pos_5 = pos + dt * (b5_1*k1_pos + b5_3*k3_pos + b5_4*k4_pos + b5_5*k5_pos + b5_6*k6_pos); vel_5 = vel + dt * (b5_1*k1_vel + b5_3*k3_vel + b5_4*k4_vel + b5_5*k5_vel + b5_6*k6_vel)
        pos_4 = pos + dt * (b4_1*k1_pos + b4_3*k3_pos + b4_4*k4_pos + b4_5*k5_pos + b4_6*k6_pos + b4_7*k7_pos)
        error = (pos_5 - pos_4).norm(); return pos_5, vel_5, error

    @ti.func
    def sample_world_density(self, pos_world):
        is_inside = True; cube_min = self.CUBE_CENTER - self.CUBE_SIZE / 2.0
        if not (cube_min.x < pos_world.x < cube_min.x + self.CUBE_SIZE and cube_min.y < pos_world.y < cube_min.y + self.CUBE_SIZE and cube_min.z < pos_world.z < cube_min.z + self.CUBE_SIZE): is_inside = False
        density = 0.0
        if is_inside: pos_grid = (pos_world - cube_min) / self.CUBE_SIZE * self.GRID_RES; density = self.sample_grid(self.density_field, pos_grid)
        return density

    @ti.func
    def get_disk_emission_properties(self, pos_world, ray_dir):
        radius_xz = pos_world.xz.norm(); temp_factor = ti.pow(self.DISK_INNER_RADIUS / (radius_xz + 1e-6), 2.5)
        base_color = ti.math.mix(self.DISK_COLOR_COLD, self.DISK_COLOR_HOT, ti.math.clamp(temp_factor * 0.5, 0.0, 1.0))
        speed = ti.sqrt(self.GM / (radius_xz + 0.1)) * self.ORBITAL_VELOCITY_SCALE * (self.CUBE_SIZE / self.GRID_RES)
        tangential_dir = ti.Vector([-pos_world.z, 0.0, pos_world.x]).normalized(); velocity_world = speed * tangential_dir
        beta = velocity_world.dot(-ray_dir); gamma = 1.0 / ti.sqrt(1.0 - velocity_world.norm_sqr()); delta = 1.0 / (gamma * (1.0 - beta))
        brightness = temp_factor * ti.pow(delta, self.DOPPLER_STRENGTH); color_shift = ti.Vector([1.0/delta, 1.0, delta])
        final_color = base_color * color_shift; return final_color * brightness

    @ti.func
    def henyey_greenstein_phase_func(self, cos_theta, g):
        g2 = g * g; return (1.0 - g2) / ti.pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5)

    @ti.func
    def get_volume_scattering(self, pos_world, ray_dir):
        total_scattered_light = ti.Vector([0.0, 0.0, 0.0])
        light_dir_1 = (self.SUN1_POS - pos_world).normalized(); dist_sq_1 = (self.SUN1_POS - pos_world).norm_sqr()
        cos_theta_1 = light_dir_1.dot(-ray_dir); phase_1 = self.henyey_greenstein_phase_func(cos_theta_1, self.HG_ASYMMETRY_FACTOR)
        disk_normal = ti.Vector([0.0, 1.0, 0.0]) if pos_world.y > 0 else ti.Vector([0.0, -1.0, 0.0])
        shadow_factor_1 = ti.math.mix(1.0 - self.SELF_SHADOW_STRENGTH, 1.0, ti.math.clamp(light_dir_1.dot(disk_normal), 0.0, 1.0))
        light_from_sun1 = self.SUN1_COLOR * (1.0 / (dist_sq_1 + 1.0)) * phase_1 * shadow_factor_1
        total_scattered_light += light_from_sun1
        light_dir_2 = (self.SUN2_POS - pos_world).normalized(); dist_sq_2 = (self.SUN2_POS - pos_world).norm_sqr()
        cos_theta_2 = light_dir_2.dot(-ray_dir); phase_2 = self.henyey_greenstein_phase_func(cos_theta_2, self.HG_ASYMMETRY_FACTOR)
        shadow_factor_2 = ti.math.mix(1.0 - self.SELF_SHADOW_STRENGTH, 1.0, ti.math.clamp(light_dir_2.dot(disk_normal), 0.0, 1.0))
        light_from_sun2 = self.SUN2_COLOR * (1.0 / (dist_sq_2 + 1.0)) * phase_2 * shadow_factor_2
        total_scattered_light += light_from_sun2; return total_scattered_light * self.SCATTERING_STRENGTH

    @ti.func
    def march_volume_segment(self, start_pos, end_pos, transmittance_in, ray_dir):
        color_out = ti.Vector([0.0, 0.0, 0.0]); transmittance_out = transmittance_in
        segment_vec = end_pos - start_pos; segment_len = segment_vec.norm()
        step_size = segment_len / self.VOLUME_SUBSTEPS
        if segment_len > 1e-4:
            segment_ray_dir = segment_vec / segment_len
            for i in range(self.VOLUME_SUBSTEPS):
                if transmittance_out < 1e-3: break
                p = start_pos + segment_ray_dir * (i + 0.5) * step_size
                density = self.sample_world_density(p) * self.DENSITY_MULTIPLIER
                if density > 1e-3:
                    density = ti.pow(density, self.DENSITY_POW)
                    emitted_light = self.get_disk_emission_properties(p, ray_dir) * self.EMISSION_STRENGTH
                    scattered_light = self.get_volume_scattering(p, ray_dir); total_light = emitted_light + scattered_light
                    step_transmittance = ti.exp(-density * step_size * self.ABSORPTION_COEFFICIENT)
                    color_out += total_light * density * transmittance_out * step_size
                    transmittance_out *= step_transmittance
        return color_out, transmittance_out
    
    @ti.func
    def get_glow_rays(self, ray_pos, sun_pos):
        vec_from_sun = ray_pos - sun_pos; radius = vec_from_sun.norm()
        inclination = ti.acos(ti.math.clamp(vec_from_sun.y / (radius + 1e-6), -1.0, 1.0))
        azimuth = ti.atan2(vec_from_sun.x, vec_from_sun.z)
        noise_coords = ti.Vector([radius * 0.1, inclination * self.SUN_RAY_STRETCH, azimuth * self.SUN_RAY_STRETCH])
        noise_val = self.fbm_ridged_3d(noise_coords * self.SUN_RAY_NOISE_SCALE)
        return ti.pow(noise_val, self.SUN_RAY_CONTRAST)

    @ti.func
    def add_sun_glow(self, ray_pos, ray_dir, sun_pos, sun_color, transmittance):
        glow_color = ti.Vector([0.0, 0.0, 0.0]); oc = ray_pos - sun_pos
        t_closest = -oc.dot(ray_dir)
        if t_closest > -self.SUN_RADIUS * 5:
            dist_sq = (oc + t_closest * ray_dir).norm_sqr()
            smooth_glow_factor = ti.exp(-dist_sq * self.SUN_GLOW_FALLOFF)
            ray_multiplier = self.get_glow_rays(ray_pos, sun_pos)
            glow_factor = smooth_glow_factor * ray_multiplier
            cool_glow_color = sun_color * ti.Vector([0.5, 0.6, 1.0])
            color_mix_factor = ti.math.smoothstep(0.0, 0.8, smooth_glow_factor)
            final_glow_color = ti.math.mix(cool_glow_color, sun_color, color_mix_factor)
            glow_color = final_glow_color * glow_factor * self.SUN_GLOW_STRENGTH * transmittance
        return glow_color

    @ti.func
    def trace_ray(self, ray_origin, ray_dir):
        pos = ray_origin; vel = ray_dir.normalized()
        color = ti.Vector([0.0, 0.0, 0.0]); transmittance = 1.0; hit_object = 0; dt = self.DT_INITIAL; step = 0
        while step < self.MAX_STEPS and hit_object == 0 and transmittance > 1e-3:
            r = pos.norm()
            if r <= self.HORIZON_RADIUS + 0.001: hit_object = 1
            elif (pos - self.SUN1_POS).norm_sqr() < self.SUN_RADIUS**2:
                hit_object = 1; color += self.get_sun_surface_color(pos, self.SUN1_COLOR) * transmittance * self.SUN_CORE_BRIGHTNESS
            elif (pos - self.SUN2_POS).norm_sqr() < self.SUN_RADIUS**2:
                hit_object = 1; color += self.get_sun_surface_color(pos, self.SUN2_COLOR) * transmittance * self.SUN_CORE_BRIGHTNESS
            elif r > self.FAR_FIELD_RADIUS: hit_object = 1; color += self.get_background_color_from_skybox(vel) * transmittance
            else:
                color += self.add_sun_glow(pos, vel, self.SUN1_POS, self.SUN1_COLOR, transmittance)
                color += self.add_sun_glow(pos, vel, self.SUN2_POS, self.SUN2_COLOR, transmittance)
                pos_new, vel_new, error = self.dopri5_step(pos, vel, dt)
                color_from_gas, new_transmittance = self.march_volume_segment(pos, pos_new, transmittance, ray_dir)
                color += color_from_gas; transmittance = new_transmittance
                if transmittance > 1e-3:
                    if error <= self.TOLERANCE: pos = pos_new; vel = vel_new.normalized(); step += 1
                    dt_new = 0.0
                    if error > 1e-12: dt_new = self.SAFETY_FACTOR * dt * ti.pow(self.TOLERANCE / error, 1.0 / 6.0)
                    else: dt_new = self.DT_MAX
                    dt = ti.max(self.DT_MIN, ti.min(self.DT_MAX, dt_new))
        if hit_object == 0: color += self.get_background_color_from_skybox(vel) * transmittance
        return color

    @ti.func
    def tone_map_aces(self, color):
        A, B, C, D, E = 2.51, 0.03, 2.43, 0.59, 0.14
        color = (color * (A * color + B)) / (color * (C * color + D) + E)
        return ti.max(0.0, ti.min(color, 1.0))

    @ti.func
    def apply_cinematic_post_processing(self, color, u, v):
        brightness = color.dot(ti.Vector([0.299, 0.587, 0.114]))
        bloom_factor = ti.math.smoothstep(self.BLOOM_THRESHOLD, self.BLOOM_THRESHOLD + 0.5, brightness)
        bloom_color = color * self.BLOOM_STRENGTH * bloom_factor; processed_color = color + bloom_color
        flare_factor = ti.exp(-abs(v) * 15.0) * bloom_factor * self.ANAMORPHIC_FLARE_STRENGTH
        flare_color = ti.Vector([0.3, 0.5, 1.0]) * flare_factor; processed_color += flare_color
        dist_from_center_sq = u*u + v*v * (self.ASPECT_RATIO**2)
        vignette_factor = 1.0 - dist_from_center_sq * self.VIGNETTE_STRENGTH; processed_color *= vignette_factor
        grain = (ti.random() - 0.5) * self.GRAIN_INTENSITY; processed_color += grain
        return processed_color

    @ti.kernel
    def render(self, cam_pos: ti.types.vector(3, ti.f32), cam_to_world: ti.types.matrix(3, 3, ti.f32), fov_local: ti.f32): #type: ignore
        exposure = 0.5
        for i, j in self.pixels:
            u = (i - self.WIDTH * 0.5) / self.HEIGHT; v = (j - self.HEIGHT * 0.5) / self.HEIGHT
            local_dir = ti.Vector([u, v, fov_local]).normalized()
            ray_dir = cam_to_world @ local_dir
            color = self.trace_ray(cam_pos, ray_dir)
            color *= exposure
            processed_color = self.apply_cinematic_post_processing(color, u, v)
            mapped_color = self.tone_map_aces(processed_color)
            self.pixels[i, j] = ti.pow(mapped_color, 1.0 / 2.2)