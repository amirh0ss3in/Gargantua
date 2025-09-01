import taichi as ti
import numpy as np
import math
import time
import os
import tempfile
import hashlib
import json

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    _rich_available = True
    console = Console()
except ImportError:
    _rich_available = False
    class _DummyConsole:
        def print(self, text): print(text)
    console = _DummyConsole()
    print("Warning: `rich` not found. Output will be plain. To improve, run: `pip install rich`")

try:
    from tqdm import tqdm
    _tqdm_available = True
except ImportError:
    _tqdm_available = False
    print("Warning: `tqdm` not found. Progress bar will be disabled. To improve, run: `pip install tqdm`")


try:
    import psutil
    _psutil_available = True
except ImportError:
    _psutil_available = False
    if _rich_available:
        console.print("[bold yellow]⚠️ Warning:[/bold yellow] [dim]`psutil` not found. System RAM/CPU monitoring will be disabled. `pip install psutil`[/dim]")
    else:
        print("Warning: `psutil` not found. System RAM/CPU monitoring will be disabled.")

try:
    from pynvml import *
    nvmlInit()
    _pynvml_available = True
except (ImportError, NVMLError):
    _pynvml_available = False
    if _rich_available:
        console.print("[bold yellow]⚠️ Warning:[/bold yellow] [dim]`pynvml` not found or failed to initialize. NVIDIA GPU monitoring will be disabled. `pip install pynvml`[/dim]")
    else:
        print("Warning: `pynvml` not found. NVIDIA GPU monitoring will be disabled.")


@ti.data_oriented
class InterstellarRenderer:
    def __init__(self, width=1200, show_gui=True, save_video_path=None, video_fps=24, use_caching=False):
        
        ti.init(arch=ti.cuda, default_fp=ti.f32, log_level=ti.WARN)

        self.show_gui = show_gui
        self.save_video = save_video_path is not None
        self.use_caching = use_caching
        self.frame_count = 0
        self.ASPECT_RATIO = 16.0 / 9.0
        self.WIDTH = width
        self.HEIGHT = int(self.WIDTH / self.ASPECT_RATIO)
        self.RESOLUTION = (self.WIDTH, self.HEIGHT)
        self.GM = 0.5
        self.SCHWARZSCHILD_RADIUS = 2.0 * self.GM
        self.HORIZON_RADIUS = self.SCHWARZSCHILD_RADIUS
        self.DISK_INNER_RADIUS = self.SCHWARZSCHILD_RADIUS * 2.5
        self.DISK_OUTER_RADIUS = self.SCHWARZSCHILD_RADIUS * 12.0
        self.DISK_HALF_THICKNESS = (self.DISK_OUTER_RADIUS - self.DISK_INNER_RADIUS) * 0.005
        self.SUN_RADIUS = 3.0
        self.SUN1_POS = ti.Vector([60.0, 10.0, -25.0])
        self.SUN1_COLOR = ti.Vector([1.0, 0.6, 0.3])
        self.SUN2_POS = ti.Vector([-50.0, -15.0, 35.0])
        self.SUN2_COLOR = ti.Vector([0.6, 0.7, 1.0])
        self.SUN_NOISE_SCALE_1, self.SUN_NOISE_SCALE_2, self.SUN_NOISE_CONTRAST = 2.0, 8.0, 10.0
        self.SUN_BRIGHTNESS_BOOST, self.SUN_CORE_BRIGHTNESS = 2.0, 10.0
        self.SUN_GLOW_STRENGTH, self.SUN_GLOW_FALLOFF = 0.1, 0.2
        self.SUN_RAY_NOISE_SCALE, self.SUN_RAY_STRETCH, self.SUN_RAY_CONTRAST = 0.9, 500.0, 9.0
        self.MAX_STEPS = 1000
        self.TOLERANCE, self.DT_INITIAL, self.DT_MIN, self.DT_MAX = 1.0e-5, 0.5, 1.0e-4, 1.0
        self.SAFETY_FACTOR = 0.9
        self.FAR_FIELD_RADIUS = 75.0
        self.BLOOM_THRESHOLD, self.BLOOM_STRENGTH = 0.3, 0.15
        self.ANAMORPHIC_FLARE_STRENGTH = 0.8
        self.VIGNETTE_STRENGTH, self.GRAIN_INTENSITY = 0.4, 0.02
        self.GRID_R, self.GRID_THETA, self.GRID_Y = 512, 1024, 128
        self.CYLINDER_MIN_RADIUS = self.DISK_INNER_RADIUS
        self.CYLINDER_MAX_RADIUS = self.DISK_OUTER_RADIUS
        self.CYLINDER_HALF_HEIGHT = self.DISK_HALF_THICKNESS * 5.0
        self.DR = (self.CYLINDER_MAX_RADIUS - self.CYLINDER_MIN_RADIUS) / self.GRID_R
        self.DTHETA = (2 * math.pi) / self.GRID_THETA
        self.DY = (2 * self.CYLINDER_HALF_HEIGHT) / self.GRID_Y
        self.VOLUME_SUBSTEPS = 256
        self.EMISSION_STRENGTH = 1200.0
        self.ABSORPTION_COEFFICIENT = 50.0
        self.DENSITY_MULTIPLIER = 1.5
        self.DENSITY_POW = 3.0
        self.SCATTERING_STRENGTH = 40.0
        self.HG_ASYMMETRY_FACTOR = 0.4
        self.SELF_SHADOW_STRENGTH = 0.8
        self.DISK_COLOR_HOT = ti.Vector([1.0, 1.0, 0.95])
        self.DISK_COLOR_COLD = ti.Vector([1.0, 0.4, 0.1])
        self.DOPPLER_STRENGTH = 7.0
        self.WARP_FIELD_SCALE, self.WARP_STRENGTH = 1.5, 1.2
        self.FILAMENT_NOISE_SCALE = 1.8
        self.FILAMENT_CONTRAST = 4.0
        self.TANGENTIAL_STRETCH = 25.0
        self.CLUMP_NOISE_SCALE = 0.5
        self.CLUMP_STRENGTH = 0.6
        self.VERTICAL_NOISE_SCALE = 2.0
        self.VERTICAL_STRENGTH = 0.6
        self.DISK_NOISE_STRENGTH = 3.0
        self.EQUATORIAL_SHADOW_WIDTH = 0.1
        self.EQUATORIAL_SHADOW_STRENGTH = 0.9
        self.DT_SIM = 0.005
        self.ADVECTION_STRENGTH = 1e-5
        self.MAX_VELOCITY_PHYSICAL = 2.0 * self.DR / self.DT_SIM
        self.JACOBI_ITERATIONS = 100
        self.DISSIPATION = 0.03
        self.ORBITAL_ASSIST_STRENGTH = 0.03
        self.ORBITAL_VELOCITY_SCALE = 1.0
        self.GRAVITY_STRENGTH = 1.0

        self.pixels = ti.Vector.field(3, dtype=ti.f32, shape=self.RESOLUTION)
        skybox_img_np = ti.tools.image.imread('starmap.jpg').astype(np.float32) / 255.0
        self.skybox_texture = ti.Vector.field(3, dtype=ti.f32, shape=skybox_img_np.shape[:2])
        self.skybox_texture.from_numpy(skybox_img_np)
        self.grid_shape = (self.GRID_R, self.GRID_THETA, self.GRID_Y)
        self.density_field = ti.field(dtype=ti.f32, shape=self.grid_shape)
        self.new_density_field = ti.field(dtype=ti.f32, shape=self.grid_shape)
        self.velocity_field = ti.Vector.field(3, dtype=ti.f32, shape=self.grid_shape)
        self.new_velocity_field = ti.Vector.field(3, dtype=ti.f32, shape=self.grid_shape)
        self.pressure_field = ti.field(dtype=ti.f32, shape=self.grid_shape)
        self.divergence_field = ti.field(dtype=ti.f32, shape=self.grid_shape)
        self.cam_to_world_matrix = ti.Matrix.field(3, 3, dtype=ti.f32, shape=())

        self.gui = None
        if self.show_gui:
            self.gui = ti.GUI("Cinematic Black Hole Flight", res=self.RESOLUTION, fast_gui=True)
            
        self._temp_dir_obj = None
        self.video_manager = None
        if self.save_video:
            self._temp_dir_obj = tempfile.TemporaryDirectory()
            temp_video_dir = self._temp_dir_obj.name
            self.video_manager = ti.tools.VideoManager(output_dir=temp_video_dir, framerate=video_fps, automatic_build=False)
            self.final_video_path = save_video_path
        
        self.cache_dir = "frame_cache"
        self.config_hash = self._get_config_hash()
        self.active_cache_dir = os.path.join(self.cache_dir, self.config_hash)
        if self.use_caching:
            os.makedirs(self.active_cache_dir, exist_ok=True)
        
        self._print_init_summary(save_video_path)
        self.reset_scene()

    def _print_init_summary(self, save_video_path):
        if not _rich_available:
            print("--- Renderer Initialized ---")
            print(f"  Mode: {'GUI Display' if self.show_gui else 'Offline Render'}")
            print(f"  Resolution: {self.WIDTH}x{self.HEIGHT}")
            print(f"  Video Output: {save_video_path if self.save_video else 'Disabled'}")
            print(f"  Caching: {'Enabled' if self.use_caching else 'Disabled'}")
            return

        settings_text = Text.from_markup(f"""
[bold]Resolution:[/bold]   {self.WIDTH}x{self.HEIGHT}
[bold]Mode:[/bold]         {'GUI Display' if self.show_gui else 'Offline Render'}
[bold]Video Output:[/bold] {f'[cyan]"{save_video_path}"[/cyan]' if self.save_video else '[dim]Disabled[/dim]'}
[bold]Caching:[/bold]      {f'[green]ENABLED[/green] ([dim]path: {self.active_cache_dir}[/dim])' if self.use_caching else '[yellow]DISABLED[/yellow]'}
""")
        console.print(Panel(settings_text, title="[bold blue]Interstellar Renderer Initialized[/bold blue]", subtitle="[dim]Ready to render[/dim]", border_style="blue"))


    def _get_config_hash(self):
        params = {
            'WIDTH': self.WIDTH, 'HEIGHT': self.HEIGHT, 'GM': self.GM,
            'DISK_INNER_RADIUS': self.DISK_INNER_RADIUS, 'DISK_OUTER_RADIUS': self.DISK_OUTER_RADIUS,
            'DISK_HALF_THICKNESS': self.DISK_HALF_THICKNESS, 'SUN_RADIUS': self.SUN_RADIUS,
            'SUN1_POS': list(self.SUN1_POS), 'SUN1_COLOR': list(self.SUN1_COLOR),
            'SUN2_POS': list(self.SUN2_POS), 'SUN2_COLOR': list(self.SUN2_COLOR),
            'SUN_NOISE_SCALE_1': self.SUN_NOISE_SCALE_1, 'SUN_NOISE_SCALE_2': self.SUN_NOISE_SCALE_2,
            'SUN_NOISE_CONTRAST': self.SUN_NOISE_CONTRAST, 'SUN_BRIGHTNESS_BOOST': self.SUN_BRIGHTNESS_BOOST,
            'SUN_CORE_BRIGHTNESS': self.SUN_CORE_BRIGHTNESS, 'SUN_GLOW_STRENGTH': self.SUN_GLOW_STRENGTH,
            'SUN_GLOW_FALLOFF': self.SUN_GLOW_FALLOFF, 'SUN_RAY_NOISE_SCALE': self.SUN_RAY_NOISE_SCALE,
            'SUN_RAY_STRETCH': self.SUN_RAY_STRETCH, 'SUN_RAY_CONTRAST': self.SUN_RAY_CONTRAST,
            'MAX_STEPS': self.MAX_STEPS, 'TOLERANCE': self.TOLERANCE, 'DT_INITIAL': self.DT_INITIAL,
            'DT_MIN': self.DT_MIN, 'DT_MAX': self.DT_MAX, 'SAFETY_FACTOR': self.SAFETY_FACTOR,
            'FAR_FIELD_RADIUS': self.FAR_FIELD_RADIUS, 'BLOOM_THRESHOLD': self.BLOOM_THRESHOLD,
            'BLOOM_STRENGTH': self.BLOOM_STRENGTH, 'ANAMORPHIC_FLARE_STRENGTH': self.ANAMORPHIC_FLARE_STRENGTH,
            'VIGNETTE_STRENGTH': self.VIGNETTE_STRENGTH, 'GRAIN_INTENSITY': self.GRAIN_INTENSITY,
            'GRID_R': self.GRID_R, 'GRID_THETA': self.GRID_THETA, 'GRID_Y': self.GRID_Y,
            'VOLUME_SUBSTEPS': self.VOLUME_SUBSTEPS, 'EMISSION_STRENGTH': self.EMISSION_STRENGTH,
            'ABSORPTION_COEFFICIENT': self.ABSORPTION_COEFFICIENT, 'DENSITY_MULTIPLIER': self.DENSITY_MULTIPLIER,
            'DENSITY_POW': self.DENSITY_POW, 'SCATTERING_STRENGTH': self.SCATTERING_STRENGTH,
            'HG_ASYMMETRY_FACTOR': self.HG_ASYMMETRY_FACTOR, 'SELF_SHADOW_STRENGTH': self.SELF_SHADOW_STRENGTH,
            'DISK_COLOR_HOT': list(self.DISK_COLOR_HOT), 'DISK_COLOR_COLD': list(self.DISK_COLOR_COLD),
            'DOPPLER_STRENGTH': self.DOPPLER_STRENGTH, 'WARP_FIELD_SCALE': self.WARP_FIELD_SCALE,
            'WARP_STRENGTH': self.WARP_STRENGTH, 'FILAMENT_NOISE_SCALE': self.FILAMENT_NOISE_SCALE,
            'FILAMENT_CONTRAST': self.FILAMENT_CONTRAST, 'TANGENTIAL_STRETCH': self.TANGENTIAL_STRETCH,
            'CLUMP_NOISE_SCALE': self.CLUMP_NOISE_SCALE, 'CLUMP_STRENGTH': self.CLUMP_STRENGTH,
            'VERTICAL_NOISE_SCALE': self.VERTICAL_NOISE_SCALE, 'VERTICAL_STRENGTH': self.VERTICAL_STRENGTH,
            'DISK_NOISE_STRENGTH': self.DISK_NOISE_STRENGTH, 'EQUATORIAL_SHADOW_WIDTH': self.EQUATORIAL_SHADOW_WIDTH,
            'EQUATORIAL_SHADOW_STRENGTH': self.EQUATORIAL_SHADOW_STRENGTH, 'DT_SIM': self.DT_SIM,
            'ADVECTION_STRENGTH': self.ADVECTION_STRENGTH, 'MAX_VELOCITY_PHYSICAL': self.MAX_VELOCITY_PHYSICAL,
            'JACOBI_ITERATIONS': self.JACOBI_ITERATIONS, 'DISSIPATION': self.DISSIPATION,
            'ORBITAL_ASSIST_STRENGTH': self.ORBITAL_ASSIST_STRENGTH, 'ORBITAL_VELOCITY_SCALE': self.ORBITAL_VELOCITY_SCALE,
            'GRAVITY_STRENGTH': self.GRAVITY_STRENGTH
        }
        param_string = json.dumps(params, sort_keys=True).encode('utf-8')
        return hashlib.sha256(param_string).hexdigest()

    def get_system_stats_str(self):
        """Returns a compact, color-coded string of system stats for TQDM."""
        if not _rich_available: return ""
        log_parts = []
        
        def get_color(percent):
            if percent < 50: return "green"
            if percent < 80: return "yellow"
            return "red"

        if _psutil_available:
            ram = psutil.virtual_memory()
            ram_color = get_color(ram.percent)
            cpu_percent = psutil.cpu_percent()
            cpu_color = get_color(cpu_percent)
            log_parts.append(f"💻 CPU: [bold {cpu_color}]{cpu_percent: >4.1f}%[/bold {cpu_color}]")
            log_parts.append(f"🧠 RAM: [bold {ram_color}]{ram.percent: >4.1f}%[/bold {ram_color}]")
        
        if _pynvml_available:
            try:
                handle = nvmlDeviceGetHandleByIndex(0)
                mem_info = nvmlDeviceGetMemoryInfo(handle)
                vram_percent = mem_info.used / mem_info.total * 100
                vram_color = get_color(vram_percent)
                log_parts.append(f"🎮 VRAM: [bold {vram_color}]{vram_percent: >4.1f}%[/bold {vram_color}]")
            except NVMLError:
                log_parts.append("VRAM: [red]N/A[/red]")

        return " | ".join(log_parts)

    def _get_frame_hash(self, cam_pos_vec, cam_fwd_vec, cam_up_vec, fov):
        """Returns a hash of the current simulation state."""
        state_string = (
            f"{self.frame_count}-"
            f"{cam_pos_vec.x:.4f}-{cam_pos_vec.y:.4f}-{cam_pos_vec.z:.4f}-"
            f"{cam_fwd_vec.x:.4f}-{cam_fwd_vec.y:.4f}-{cam_fwd_vec.z:.4f}-"
            f"{cam_up_vec.x:.4f}-{cam_up_vec.y:.4f}-{cam_up_vec.z:.4f}-"
            f"{fov:.4f}"
        )
        return hashlib.sha256(state_string.encode()).hexdigest()

    def reset_scene(self):
        if _rich_available:
            console.print("[dim]Resetting gas simulation to initial state...[/dim]")
        else:
            print("Resetting gas simulation to initial state...")
        self.init_scene()
        self.init_velocity()

    def step(self, cam_pos_vec, cam_fwd_vec, cam_up_vec, fov):
        """Renders one frame and returns cache status."""
        self.frame_count += 1
        self.simulation_step() 
        
        cache_hit = False
        if self.use_caching:
            frame_hash = self._get_frame_hash(cam_pos_vec, cam_fwd_vec, cam_up_vec, fov)
            cache_path = os.path.join(self.active_cache_dir, f"{frame_hash}.png")
            
            if os.path.exists(cache_path):
                cached_frame_np = ti.tools.image.imread(cache_path).astype(np.float32) / 255.0
                if self.save_video and self.video_manager:
                    self.video_manager.write_frame(cached_frame_np)
                self.pixels.from_numpy(cached_frame_np)
                cache_hit = True
        
        if not cache_hit:
            fwd, up = cam_fwd_vec.normalized(), cam_up_vec.normalized()
            right = fwd.cross(up).normalized(); up = right.cross(fwd)
            self.cam_to_world_matrix[None] = ti.Matrix.cols([right, up, fwd])
            self.render(cam_pos_vec, self.cam_to_world_matrix[None], fov)
            
            if self.save_video or self.use_caching:
                frame_numpy = self.pixels.to_numpy()
                if self.save_video and self.video_manager:
                    self.video_manager.write_frame(frame_numpy)
                if self.use_caching:
                    ti.tools.image.imwrite(frame_numpy, cache_path)

        if self.show_gui and self.gui: self.gui.set_image(self.pixels); self.gui.show()
        
        return "cache_hit" if cache_hit else "cache_miss"

    def close(self):
        if self.video_manager:
            spinner_text = Text.from_markup("[bold green]Finalizing video... This may take a moment.[/bold green]")
            status_context = console.status(spinner_text, spinner="dots") if _rich_available else open(os.devnull, 'w')
            
            with status_context:
                try:
                    self.video_manager.make_video(gif=False, mp4=True)
                    default_output = os.path.join(self._temp_dir_obj.name, "video.mp4")
                    for attempt in range(10):
                        try:
                            if os.path.exists(default_output):
                                os.replace(default_output, self.final_video_path)
                                break
                        except PermissionError:
                            if attempt < 9: time.sleep(0.5)
                            else: raise
                finally:
                    if self._temp_dir_obj:
                        self._temp_dir_obj.cleanup()
            
            if _rich_available:
                final_path_abs = os.path.abspath(self.final_video_path)
                final_text = Text.from_markup(f"""\
[bold green]✓ Video saved successfully![/bold green]

[dim]Output file:[/dim] [link=file://{final_path_abs}][cyan]{self.final_video_path}[/cyan][/link]
""")
                console.print(Panel(final_text, title="[magenta]Render Complete[/magenta]", border_style="magenta", expand=False))
            else:
                print(f"Video saved to '{self.final_video_path}'")


        if self.gui: self.gui.close()
        
        if _pynvml_available:
            nvmlShutdown()

        console.print("[bold]Renderer closed.[/bold]")
    def get_init_panel(self):
        """Returns a rich Panel summarizing the renderer's configuration."""
        if not _rich_available:
            return "" # Return empty string if rich is not installed

        settings_text = Text.from_markup(f"""
    [bold]Resolution:[/bold]   {self.WIDTH}x{self.HEIGHT}
    [bold]Mode:[/bold]         {'GUI Display' if self.show_gui else 'Offline Render'}
    [bold]Caching:[/bold]      {f'[green]ENABLED[/green] ([dim]path: {self.active_cache_dir}[/dim])' if self.use_caching else '[yellow]DISABLED[/yellow]'}
    """)
        return Panel(settings_text, title="[bold blue]Interstellar Renderer Initialized[/bold blue]", border_style="blue", expand=False)

    @ti.func
    def sample_texture(self, texture: ti.template(), u: ti.f32, v: ti.f32):  #type: ignore
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
        for _ in range(6): val += amp * (1.0 - abs(self.value_noise_3d(p * freq) - 0.5) * 2.0); amp *= 0.5; freq *= 2.0
        return val

    @ti.func
    def get_sun_surface_color(self, surface_pos, base_color):
        noise1 = self.fbm_ridged_3d(surface_pos * self.SUN_NOISE_SCALE_1)
        noise2 = self.fbm_ridged_3d(surface_pos * self.SUN_NOISE_SCALE_2)
        final_noise = ti.pow(noise1 * 0.7 + noise2 * 0.3, self.SUN_NOISE_CONTRAST)
        return ti.math.mix(base_color, ti.Vector([1.0, 1.0, 0.9]), final_noise) * (1.0 + final_noise * self.SUN_BRIGHTNESS_BOOST)

    @ti.func
    def grid_to_world_cylindrical(self, i, j, k):
        r = self.CYLINDER_MIN_RADIUS + (i + 0.5) * self.DR
        theta = (j + 0.5) * self.DTHETA - math.pi
        y = -self.CYLINDER_HALF_HEIGHT + (k + 0.5) * self.DY
        return ti.Vector([r * ti.cos(theta), y, r * ti.sin(theta)])

    @ti.func
    def world_to_grid_cylindrical(self, pos_world):
        r = pos_world.xz.norm()
        theta = ti.atan2(pos_world.z, pos_world.x)
        y = pos_world.y
        r_idx = (r - self.CYLINDER_MIN_RADIUS) / self.DR - 0.5
        theta_idx = (theta + math.pi) / self.DTHETA - 0.5
        y_idx = (y + self.CYLINDER_HALF_HEIGHT) / self.DY - 0.5
        return ti.Vector([r_idx, theta_idx, y_idx])

    @ti.func
    def sample_cylindrical_grid(self, field, pos_grid):
        p_floor = ti.floor(pos_grid); p_frac = pos_grid - p_floor
        i, j, k = ti.cast(p_floor, ti.i32)
        i = ti.max(0, ti.min(i, self.GRID_R - 2)); k = ti.max(0, ti.min(k, self.GRID_Y - 2))
        j0, j1 = j % self.GRID_THETA, (j + 1) % self.GRID_THETA
        v000=field[i,j0,k];   v100=field[i+1,j0,k]; v010=field[i,j1,k];   v110=field[i+1,j1,k]
        v001=field[i,j0,k+1]; v101=field[i+1,j0,k+1]; v011=field[i,j1,k+1]; v111=field[i+1,j1,k+1]
        v00=ti.math.mix(v000,v100,p_frac.x); v10=ti.math.mix(v010,v110,p_frac.x)
        v01=ti.math.mix(v001,v101,p_frac.x); v11=ti.math.mix(v011,v111,p_frac.x)
        v0=ti.math.mix(v00,v10,p_frac.y);   v1=ti.math.mix(v01,v11,p_frac.y)
        return ti.math.mix(v0, v1, p_frac.z)
    
    @ti.func
    def backtrace_cylindrical(self, i, j, k):
        vel = self.velocity_field[i, j, k]
        r = self.CYLINDER_MIN_RADIUS + (i + 0.5) * self.DR
        dt_adv = self.DT_SIM * self.ADVECTION_STRENGTH
        dr = vel.x * dt_adv
        d_theta_rad = (vel.y / (r + 1e-6)) * dt_adv
        dy = vel.z * dt_adv
        dr_grid = dr / self.DR
        d_theta_grid = d_theta_rad / self.DTHETA
        dy_grid = dy / self.DY
        return ti.Vector([i - dr_grid, j - d_theta_grid, k - dy_grid])

    @ti.kernel
    def advect_density(self, field_in: ti.template(), field_out: ti.template()):  #type: ignore
        for i, j, k in field_in:
            p_prev = self.backtrace_cylindrical(i, j, k)
            field_out[i,j,k] = self.sample_cylindrical_grid(field_in, p_prev)
    
    @ti.kernel
    def advect_velocity(self, field_in: ti.template(), field_out: ti.template()):  #type: ignore
        for i, j, k in field_in:
            p_prev = self.backtrace_cylindrical(i, j, k)
            sampled_val = self.sample_cylindrical_grid(field_in, p_prev)
            field_out[i,j,k] = ti.math.mix(sampled_val, ti.Vector([0.0,0.0,0.0]), self.DISSIPATION)

    @ti.kernel
    def apply_forces(self):
        for i, j, k in self.velocity_field:
            pos_world = self.grid_to_world_cylindrical(i, j, k)
            r = pos_world.xz.norm()
            ideal_speed = ti.sqrt(self.GM / (r + 0.1)); 
            tangential_dir = ti.Vector([-pos_world.z, 0.0, pos_world.x]).normalized()
            ideal_velocity_world = ideal_speed * tangential_dir * self.ORBITAL_VELOCITY_SCALE
            theta = ti.atan2(pos_world.z, pos_world.x)
            sin_t, cos_t = ti.sin(theta), ti.cos(theta)
            v_r, v_theta, v_y = self.velocity_field[i, j, k]
            current_velocity_world = ti.Vector([v_r*cos_t - v_theta*sin_t, v_y, v_r*sin_t + v_theta*cos_t])
            correction_vec_world = ideal_velocity_world - current_velocity_world
            gravity_force_world = ti.Vector([0.0, 0.0, 0.0]); r_sqr = pos_world.norm_sqr()
            if r_sqr > 0.1: gravity_force_world = -pos_world.normalized() * (self.GRAVITY_STRENGTH * self.GM / r_sqr)
            total_force_world = correction_vec_world * self.ORBITAL_ASSIST_STRENGTH + gravity_force_world
            force_r = total_force_world.dot(ti.Vector([cos_t, 0.0, sin_t]))
            force_theta = total_force_world.dot(ti.Vector([-sin_t, 0.0, cos_t]))
            force_y = total_force_world.y
            self.velocity_field[i, j, k] += ti.Vector([force_r, force_theta, force_y]) * self.DT_SIM

    @ti.kernel
    def clamp_velocity(self):
        for i, j, k in self.velocity_field:
            vel = self.velocity_field[i, j, k]
            vel_sqr = vel.norm_sqr()
            if vel_sqr > self.MAX_VELOCITY_PHYSICAL**2:
                self.velocity_field[i, j, k] = vel.normalized() * self.MAX_VELOCITY_PHYSICAL

    @ti.kernel
    def compute_divergence(self):
        inv_dr, inv_dtheta, inv_dy = 1.0/self.DR, 1.0/self.DTHETA, 1.0/self.DY
        for i, j, k in self.velocity_field:
            if i > 0 and i < self.GRID_R -1:
                r_i = self.CYLINDER_MIN_RADIUS + (i + 0.5) * self.DR
                r_p = self.CYLINDER_MIN_RADIUS + (i + 1.5) * self.DR
                r_m = self.CYLINDER_MIN_RADIUS + (i - 0.5) * self.DR
                vr_p = self.velocity_field[i+1, j, k].x; vr_m = self.velocity_field[i-1, j, k].x
                vtheta_p = self.velocity_field[i, (j+1)%self.GRID_THETA, k].y; vtheta_m = self.velocity_field[i, (j-1+self.GRID_THETA)%self.GRID_THETA, k].y
                vy_p = self.velocity_field[i, j, k+1].z; vy_m = self.velocity_field[i, j, k-1].z
                div_r = (r_p * vr_p - r_m * vr_m) * 0.5 * inv_dr / r_i
                div_theta = (vtheta_p - vtheta_m) * 0.5 * inv_dtheta / r_i
                div_y = (vy_p - vy_m) * 0.5 * inv_dy
                self.divergence_field[i,j,k] = div_r + div_theta + div_y
                self.pressure_field[i,j,k] = 0.0
            else: self.divergence_field[i,j,k] = 0.0; self.pressure_field[i,j,k] = 0.0
                
    @ti.kernel
    def solve_pressure_red_black(self, is_red_pass: ti.i32):  #type: ignore
        for i, j, k in self.pressure_field:
            if i > 0 and i < self.GRID_R-1 and k > 0 and k < self.GRID_Y-1 and (i + j + k) % 2 == is_red_pass:
                pr = self.pressure_field[i+1, j, k]; pl = self.pressure_field[i-1, j, k]
                pt = self.pressure_field[i, (j+1)%self.GRID_THETA, k]; pb = self.pressure_field[i, (j-1+self.GRID_THETA)%self.GRID_THETA, k]
                pf = self.pressure_field[i, j, k+1]; p_bk = self.pressure_field[i, j, k-1]
                self.pressure_field[i,j,k] = (pl + pr + pb + pt + p_bk + pf - self.divergence_field[i,j,k]) / 6.0

    @ti.kernel
    def project(self):
        inv_dr, inv_dtheta, inv_dy = 1.0/self.DR, 1.0/self.DTHETA, 1.0/self.DY
        for i, j, k in self.velocity_field:
            if i > 0 and i < self.GRID_R-1 and k > 0 and k < self.GRID_Y-1:
                r = self.CYLINDER_MIN_RADIUS + (i + 0.5) * self.DR
                grad_p_r = (self.pressure_field[i+1, j, k] - self.pressure_field[i-1, j, k]) * 0.5 * inv_dr
                grad_p_theta = (self.pressure_field[i, (j+1)%self.GRID_THETA, k] - self.pressure_field[i, (j-1+self.GRID_THETA)%self.GRID_THETA, k]) * 0.5 * inv_dtheta / r
                grad_p_y = (self.pressure_field[i, j, k+1] - self.pressure_field[i, j, k-1]) * 0.5 * inv_dy
                self.velocity_field[i, j, k] -= ti.Vector([grad_p_r, grad_p_theta, grad_p_y])

    @ti.kernel
    def copy_field(self, field_in: ti.template(), field_out: ti.template()): #type: ignore
        for I in ti.grouped(field_in): field_out[I] = field_in[I]

    def simulation_step(self):
        self.advect_velocity(self.velocity_field, self.new_velocity_field)
        self.advect_density(self.density_field, self.new_density_field)
        self.copy_field(self.new_velocity_field, self.velocity_field)
        self.copy_field(self.new_density_field, self.density_field)
        self.apply_forces()
        self.clamp_velocity() 
        self.compute_divergence()
        for _ in range(self.JACOBI_ITERATIONS): 
            self.solve_pressure_red_black(1)
            self.solve_pressure_red_black(0)
        self.project()

    @ti.kernel
    def init_scene(self):
        for i, j, k in self.density_field:
            pos_world = self.grid_to_world_cylindrical(i, j, k)
            radius_xz = pos_world.xz.norm()           
            vertical_noise_pos = pos_world * self.VERTICAL_NOISE_SCALE
            vertical_mod = 1.0 + (self.fbm_ridged_3d(vertical_noise_pos) - 0.5) * 2.0 * self.VERTICAL_STRENGTH
            modulated_half_thickness = self.DISK_HALF_THICKNESS * vertical_mod
            falloff_y = ti.math.smoothstep(modulated_half_thickness, modulated_half_thickness * 0.7, abs(pos_world.y))
            falloff_in = ti.math.smoothstep(self.DISK_INNER_RADIUS, self.DISK_INNER_RADIUS * 1.2, radius_xz)
            falloff_out = ti.math.smoothstep(self.DISK_OUTER_RADIUS, self.DISK_OUTER_RADIUS * 0.8, radius_xz)
            base_density_canvas = falloff_y * falloff_in * falloff_out          
            warp_coords = pos_world * self.WARP_FIELD_SCALE
            warp_vec = ti.Vector([
                self.fbm_ridged_3d(warp_coords + 13.7),
                self.fbm_ridged_3d(warp_coords + 24.2),
                self.fbm_ridged_3d(warp_coords + 19.1)
            ])
            warped_pos = pos_world + (warp_vec - 0.5) * 2.0 * self.WARP_STRENGTH
            theta = ti.atan2(warped_pos.z, warped_pos.x)
            cos_t, sin_t = ti.cos(theta), ti.sin(theta)
            radial_dir = ti.Vector([cos_t, 0.0, sin_t])
            tangential_dir = ti.Vector([-sin_t, 0.0, cos_t])
            vertical_dir = ti.Vector([0.0, 1.0, 0.0])
            local_coords = ti.Vector([
                warped_pos.dot(radial_dir),
                warped_pos.dot(tangential_dir),
                warped_pos.dot(vertical_dir)
            ])
            filament_coords = local_coords * ti.Vector([1.0, self.TANGENTIAL_STRETCH, 1.0]) * self.FILAMENT_NOISE_SCALE
            filament_noise = ti.pow(self.fbm_ridged_3d(filament_coords), self.FILAMENT_CONTRAST)
            clump_coords = local_coords * self.CLUMP_NOISE_SCALE
            clump_noise = self.fbm_ridged_3d(clump_coords)
            combined_noise = ti.math.mix(filament_noise, clump_noise, self.CLUMP_STRENGTH)
            final_density = base_density_canvas * combined_noise * self.DISK_NOISE_STRENGTH
            self.density_field[i, j, k] = ti.max(0.0, final_density)
    @ti.kernel
    def init_velocity(self):
        for i, j, k in self.velocity_field:
            pos_world = self.grid_to_world_cylindrical(i, j, k)
            r = pos_world.xz.norm()
            ideal_velocity_world = ti.Vector([0.0, 0.0, 0.0])
            if r > 0.1:
                tangential_dir = ti.Vector([-pos_world.z, 0, pos_world.x]).normalized()
                speed = ti.sqrt(self.GM / (r + 0.1)) * self.ORBITAL_VELOCITY_SCALE
                ideal_velocity_world = speed * tangential_dir
            theta = ti.atan2(pos_world.z, pos_world.x)
            cos_t, sin_t = ti.cos(theta), ti.sin(theta)
            v_r = ideal_velocity_world.x * cos_t + ideal_velocity_world.z * sin_t
            v_theta = -ideal_velocity_world.x * sin_t + ideal_velocity_world.z * cos_t
            self.velocity_field[i, j, k] = ti.Vector([v_r, v_theta, 0.0])

    @ti.func
    def get_acceleration_gr(self, pos, vel):
        r = pos.norm() + 1e-9; L_vec = pos.cross(vel); L2 = L_vec.dot(L_vec)
        gr_term = (3.0 * self.SCHWARZSCHILD_RADIUS * L2) / (2.0 * r**5); return -gr_term * pos

    @ti.func
    def dopri5_step(self, pos, vel, dt):
        a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a62, a63, a64, a65, a71, a73, a74, a75, a76 = 1/5, 3/40, 9/40, 44/45, -56/15, 32/9, 19372/6561, -25360/2187, 64448/6561, -212/729, 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 35/384, 500/1113, 125/192, -2187/6784, 11/84
        b5_1, b5_3, b5_4, b5_5, b5_6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84
        b4_1, b4_3, b4_4, b4_5, b4_6, b4_7 = 5179/57600, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40
        k1_pos = vel; k1_vel = self.get_acceleration_gr(pos, vel)
        k2_pos = vel + dt*a21*k1_vel; k2_vel = self.get_acceleration_gr(pos+dt*a21*k1_pos, k2_pos)
        k3_pos = vel + dt*(a31*k1_vel+a32*k2_vel); k3_vel = self.get_acceleration_gr(pos+dt*(a31*k1_pos+a32*k2_pos), k3_pos)
        k4_pos = vel + dt*(a41*k1_vel+a42*k2_vel+a43*k3_vel); k4_vel = self.get_acceleration_gr(pos+dt*(a41*k1_pos+a42*k2_pos+a43*k3_pos), k4_pos)
        k5_pos = vel + dt*(a51*k1_vel+a52*k2_vel+a53*k3_vel+a54*k4_vel); k5_vel = self.get_acceleration_gr(pos+dt*(a51*k1_pos+a52*k2_pos+a53*k3_pos+a54*k4_pos), k5_pos)
        k6_pos = vel + dt*(a61*k1_vel+a62*k2_vel+a63*k3_vel+a64*k4_vel+a65*k5_vel); k6_vel = self.get_acceleration_gr(pos+dt*(a61*k1_pos+a62*k2_pos+a63*k3_pos+a64*k4_pos+a65*k5_pos), k6_pos)
        k7_pos = vel + dt * (a71*k1_vel + a73*k3_vel + a74*k4_vel + a75*k5_vel + a76*k6_vel)
        pos_5 = pos + dt*(b5_1*k1_pos + b5_3*k3_pos + b5_4*k4_pos + b5_5*k5_pos + b5_6*k6_pos)
        vel_5 = vel + dt*(b5_1*k1_vel + b5_3*k3_vel + b5_4*k4_vel + b5_5*k5_vel + b5_6*k6_vel)
        pos_4 = pos + dt*(b4_1*k1_pos + b4_3*k3_pos + b4_4*k4_pos + b4_5*k5_pos + b4_6*k6_pos + b4_7*k7_pos)
        return pos_5, vel_5, (pos_5 - pos_4).norm()

    @ti.func
    def sample_world_density(self, pos_world):
        r = pos_world.xz.norm()
        density = 0.0
        if self.CYLINDER_MIN_RADIUS < r < self.CYLINDER_MAX_RADIUS and abs(pos_world.y) < self.CYLINDER_HALF_HEIGHT:
            vertical_falloff = ti.exp(- (pos_world.y * pos_world.y) / (2.0 * self.DISK_HALF_THICKNESS**2))
            pos_grid_2d_slice = self.world_to_grid_cylindrical(ti.Vector([pos_world.x, 0.0, pos_world.z]))
            density_from_grid = self.sample_cylindrical_grid(self.density_field, pos_grid_2d_slice)
            density = density_from_grid * vertical_falloff            
        return density

    @ti.func
    def get_disk_emission_properties(self, pos_world, ray_dir):
        radius_xz = pos_world.xz.norm()
        temp_factor = ti.pow(self.DISK_INNER_RADIUS / (radius_xz + 1e-6), 2.5)
        temp_mapped = ti.pow(ti.math.clamp(temp_factor, 0.0, 1.0), 0.8)
        base_color = ti.math.mix(self.DISK_COLOR_COLD, self.DISK_COLOR_HOT, temp_mapped)
        speed = ti.sqrt(self.GM / (radius_xz + 0.1)) * self.ORBITAL_VELOCITY_SCALE
        tangential_dir = ti.Vector([-pos_world.z, 0.0, pos_world.x]).normalized()
        velocity_world = speed * tangential_dir
        beta = velocity_world.dot(-ray_dir)
        gamma = 1.0 / ti.sqrt(1.0 - velocity_world.norm_sqr())
        delta = 1.0 / (gamma * (1.0 - beta))
        brightness = temp_factor * ti.pow(delta, self.DOPPLER_STRENGTH)
        color_shift = ti.Vector([1.0/delta, 1.0, delta])
        shadow_falloff = ti.math.smoothstep(self.EQUATORIAL_SHADOW_WIDTH, 0.0, abs(pos_world.y))
        shadow_factor = 1.0 - self.EQUATORIAL_SHADOW_STRENGTH * shadow_falloff
        return base_color * color_shift * brightness * shadow_factor
        
    @ti.func
    def henyey_greenstein_phase_func(self, cos_theta, g):
        g2 = g * g; return (1.0 - g2) / ti.pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5)

    @ti.func
    def get_volume_scattering(self, pos_world, ray_dir):
        total_scattered_light = ti.Vector([0.0, 0.0, 0.0])
        disk_normal = ti.Vector([0.0, 1.0, 0.0]) if pos_world.y > 0 else ti.Vector([0.0, -1.0, 0.0])
        light_dir_1 = (self.SUN1_POS - pos_world).normalized(); dist_sq_1 = (self.SUN1_POS - pos_world).norm_sqr()
        cos_theta_1 = light_dir_1.dot(-ray_dir); phase_1 = self.henyey_greenstein_phase_func(cos_theta_1, self.HG_ASYMMETRY_FACTOR)
        shadow_factor_1 = ti.math.mix(1.0 - self.SELF_SHADOW_STRENGTH, 1.0, ti.math.clamp(light_dir_1.dot(disk_normal), 0.0, 1.0))
        total_scattered_light += self.SUN1_COLOR * (1.0 / (dist_sq_1 + 1.0)) * phase_1 * shadow_factor_1
        light_dir_2 = (self.SUN2_POS - pos_world).normalized(); dist_sq_2 = (self.SUN2_POS - pos_world).norm_sqr()
        cos_theta_2 = light_dir_2.dot(-ray_dir); phase_2 = self.henyey_greenstein_phase_func(cos_theta_2, self.HG_ASYMMETRY_FACTOR)
        shadow_factor_2 = ti.math.mix(1.0 - self.SELF_SHADOW_STRENGTH, 1.0, ti.math.clamp(light_dir_2.dot(disk_normal), 0.0, 1.0))
        total_scattered_light += self.SUN2_COLOR * (1.0 / (dist_sq_2 + 1.0)) * phase_2 * shadow_factor_2
        return total_scattered_light * self.SCATTERING_STRENGTH

    @ti.func
    def march_volume_segment(self, start_pos, end_pos, transmittance_in, ray_dir):
        color_out = ti.Vector([0.0, 0.0, 0.0]); transmittance_out = transmittance_in
        segment_vec = end_pos - start_pos; segment_len = segment_vec.norm()
        if segment_len > 1e-4:
            step_size = segment_len / self.VOLUME_SUBSTEPS
            segment_ray_dir = segment_vec / segment_len
            for i in range(self.VOLUME_SUBSTEPS):
                if transmittance_out < 1e-3: break
                p = start_pos + segment_ray_dir * (i + 0.5) * step_size
                density = self.sample_world_density(p) * self.DENSITY_MULTIPLIER
                if density > 1e-3:
                    density = ti.pow(density, self.DENSITY_POW)
                    total_light = self.get_disk_emission_properties(p, ray_dir) * self.EMISSION_STRENGTH + self.get_volume_scattering(p, ray_dir)
                    step_transmittance = ti.exp(-density * step_size * self.ABSORPTION_COEFFICIENT)
                    color_out += total_light * density * transmittance_out * step_size
                    transmittance_out *= step_transmittance
        return color_out, transmittance_out
    
    @ti.func
    def get_glow_rays(self, ray_pos, sun_pos):
        vec_from_sun = ray_pos - sun_pos; radius = vec_from_sun.norm()
        inclination = ti.acos(ti.math.clamp(vec_from_sun.y / (radius + 1e-6), -1.0, 1.0))
        azimuth = ti.atan2(vec_from_sun.x, vec_from_sun.z)
        noise_coords = ti.Vector([radius * 0.1, inclination, azimuth]) * self.SUN_RAY_STRETCH
        return ti.pow(self.fbm_ridged_3d(noise_coords * self.SUN_RAY_NOISE_SCALE), self.SUN_RAY_CONTRAST)

    @ti.func
    def add_sun_glow(self, ray_pos, ray_dir, sun_pos, sun_color, transmittance):
        oc = ray_pos - sun_pos; t_closest = -oc.dot(ray_dir)
        glow_color = ti.Vector([0.0, 0.0, 0.0])
        if t_closest > -self.SUN_RADIUS * 5:
            dist_sq = (oc + t_closest * ray_dir).norm_sqr()
            smooth_glow = ti.exp(-dist_sq * self.SUN_GLOW_FALLOFF)
            glow_factor = smooth_glow * self.get_glow_rays(ray_pos, sun_pos)
            final_color = ti.math.mix(sun_color * ti.Vector([0.5, 0.6, 1.0]), sun_color, ti.math.smoothstep(0.0, 0.8, smooth_glow))
            glow_color = final_color * glow_factor * self.SUN_GLOW_STRENGTH * transmittance
        return glow_color

    @ti.func
    def trace_ray(self, ray_origin, ray_dir):
        pos, vel = ray_origin, ray_dir.normalized()
        color, transmittance = ti.Vector([0.0, 0.0, 0.0]), 1.0
        hit_object, dt, step = 0, self.DT_INITIAL, 0 # NOTE: Do not fucking touch this.
        while step < self.MAX_STEPS and hit_object == 0 and transmittance > 1e-3:
            r = pos.norm()
            if r <= self.HORIZON_RADIUS + 0.001: hit_object = 1
            elif (pos - self.SUN1_POS).norm_sqr() < self.SUN_RADIUS**2: hit_object = 1; color += self.get_sun_surface_color(pos, self.SUN1_COLOR) * transmittance * self.SUN_CORE_BRIGHTNESS
            elif (pos - self.SUN2_POS).norm_sqr() < self.SUN_RADIUS**2: hit_object = 1; color += self.get_sun_surface_color(pos, self.SUN2_COLOR) * transmittance * self.SUN_CORE_BRIGHTNESS
            elif r > self.FAR_FIELD_RADIUS: hit_object = 1; color += self.get_background_color_from_skybox(vel) * transmittance
            else:
                color += self.add_sun_glow(pos, vel, self.SUN1_POS, self.SUN1_COLOR, transmittance)
                color += self.add_sun_glow(pos, vel, self.SUN2_POS, self.SUN2_COLOR, transmittance)
                pos_new, vel_new, error = self.dopri5_step(pos, vel, dt)
                color_gas, trans_new = self.march_volume_segment(pos, pos_new, transmittance, ray_dir)
                color += color_gas; transmittance = trans_new
                if transmittance > 1e-3:
                    if error <= self.TOLERANCE: pos, vel, step = pos_new, vel_new.normalized(), step + 1
                    dt = ti.max(self.DT_MIN, ti.min(self.DT_MAX, self.SAFETY_FACTOR * dt * ti.pow(self.TOLERANCE / (error+1e-12), 1/6)))
        if hit_object == 0: color += self.get_background_color_from_skybox(vel) * transmittance
        return color

    @ti.func
    def tone_map_aces(self, color):
        A, B, C, D, E = 2.51, 0.03, 2.43, 0.59, 0.14
        return ti.max(0.0, ti.min((color * (A * color + B)) / (color * (C * color + D) + E), 1.0))

    @ti.func
    def apply_cinematic_post_processing(self, color, u, v):
        brightness = color.dot(ti.Vector([0.299, 0.587, 0.114]))
        bloom = color * self.BLOOM_STRENGTH * ti.math.smoothstep(self.BLOOM_THRESHOLD, self.BLOOM_THRESHOLD + 0.5, brightness)
        flare = ti.Vector([0.3, 0.5, 1.0]) * ti.exp(-abs(v) * 15.0) * bloom.norm() * self.ANAMORPHIC_FLARE_STRENGTH
        vignette = 1.0 - (u*u + v*v * (self.ASPECT_RATIO**2)) * self.VIGNETTE_STRENGTH
        grain = (ti.random() - 0.5) * self.GRAIN_INTENSITY
        return (color + bloom + flare) * vignette + grain

    @ti.kernel
    def render(self, cam_pos: ti.types.vector(3, ti.f32), cam_to_world: ti.types.matrix(3, 3, ti.f32), fov_local: ti.f32):  #type: ignore
        for i, j in self.pixels:
            u, v = (i - self.WIDTH*0.5)/self.HEIGHT, (j - self.HEIGHT*0.5)/self.HEIGHT
            ray_dir = cam_to_world @ ti.Vector([u, v, fov_local]).normalized()
            color = self.trace_ray(cam_pos, ray_dir)
            processed_color = self.apply_cinematic_post_processing(color * 0.5, u, v)
            self.pixels[i, j] = ti.pow(self.tone_map_aces(processed_color), 1.0 / 2.2)