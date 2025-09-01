import time
import math
import taichi as ti
import textwrap
from interstellar_renderer import InterstellarRenderer

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

try:
    from tqdm import tqdm
    _tqdm_available = True
except ImportError:
    _tqdm_available = False


def main():
    """
    Initializes the renderer and runs a scripted camera animation loop.
    For offline rendering, it provides a detailed, color-coded progress bar.
    For real-time mode, it displays the animation in a GUI.
    """
    # --- 1. RENDER CONFIGURATION ---
    # -- Choose your mode --
    # Mode 1: Real-time preview
    # SHOW_GUI = True
    # SAVE_VIDEO = False
    
    # Mode 2: Offline rendering to video (no GUI)
    SHOW_GUI = False
    SAVE_VIDEO = True

    # -- Video settings (only used if SAVE_VIDEO is True) --
    VIDEO_DURATION_SECONDS = 10
    VIDEO_FPS = 24
    OUTPUT_FILENAME = "interstellar_flight.mp4"

    # -- Render Quality --
    # Low (for quick previews): 960
    # Medium (good balance): 1280
    # High (final render): 1920
    RENDER_WIDTH = 1280

    # 2. Initialize the scene renderer with the chosen configuration.
    scene = InterstellarRenderer(
        show_gui=SHOW_GUI,
        save_video_path=OUTPUT_FILENAME if SAVE_VIDEO else None,
        video_fps=VIDEO_FPS,
        width=RENDER_WIDTH,
        use_caching=True
    )
    
    # 3. Main application loop
    if SAVE_VIDEO:
        # --- OFFLINE RENDER LOOP (with TQDM progress bar) ---
        total_frames = int(VIDEO_DURATION_SECONDS * VIDEO_FPS)
        console.print(f"\n[bold magenta]Starting offline render of {total_frames} frames to '{OUTPUT_FILENAME}'...[/bold magenta]\n")
        
        cache_hits, cache_misses = 0, 0
        total_render_time = 0.0

        progress_bar = tqdm(range(total_frames), desc="Initializing...", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") if _tqdm_available else range(total_frames)

        for frame in progress_bar:
            start_frame_time = time.time()
            
            # Calculate animation time based on frame number for deterministic output
            animation_time = frame / VIDEO_FPS
            
            # Get camera vectors for the current time
            cam_pos, cam_fwd, world_up, fov = get_camera_vectors_at_time(animation_time)
            
            # Render the frame and get cache status
            status = scene.step(cam_pos, cam_fwd, world_up, fov)
            
            frame_duration = time.time() - start_frame_time
            total_render_time += frame_duration
            
            if status == 'cache_hit':
                cache_hits += 1
            else:
                cache_misses += 1
            
            # Update the rich progress bar
            if _tqdm_available and _rich_available:
                stats_str = scene.get_system_stats_str()
                cache_str = f"Cache: [bold green]{cache_hits}[/bold green] hit, [bold red]{cache_misses}[/bold red] miss"
                progress_bar.set_description_str(Text.from_markup(f"[cyan]Rendering Frame {frame+1}/{total_frames}[/cyan]"))
                progress_bar.set_postfix_str(Text.from_markup(f"Last: {frame_duration:.2f}s | {cache_str} | {stats_str}"), refresh=True)
            elif _tqdm_available: # Fallback for no rich
                progress_bar.set_postfix_str(f"Last: {frame_duration:.2f}s, Hits: {cache_hits}, Misses: {cache_misses}")

    else:
        # --- REAL-TIME GUI LOOP ---
        start_time = time.time()
        while scene.gui and scene.gui.running:
            # Handle GUI events
            for e in scene.gui.get_events(ti.GUI.PRESS):
                if e.key == ti.GUI.ESCAPE:
                    scene.gui.running = False

            # Calculate animation time based on wall-clock time
            animation_time = time.time() - start_time

            # Get camera vectors
            cam_pos, cam_fwd, world_up, fov = get_camera_vectors_at_time(animation_time)
            
            # Render and display the frame
            scene.step(cam_pos, cam_fwd, world_up, fov)

    # 4. Clean up and finalize resources (this now handles printing the final video path)
    scene.close()

    # 5. Print final summary for offline renders
    if SAVE_VIDEO and _rich_available:
        summary_text = Text.from_markup(textwrap.dedent(f"""\
            [bold]Total Frames:[/bold]   {total_frames}
            [bold]Total Time:[/bold]     {total_render_time:.2f} seconds
            [bold]Average Time:[/bold]   {total_render_time/total_frames:.2f} s/frame
            
            [bold]Cache Hits:[/bold]     [green]{cache_hits}[/green]
            [bold]Cache Misses:[/bold]   [red]{cache_misses}[/red]
            [bold]Cache Hit Rate:[/bold] [cyan]{(cache_hits / total_frames * 100):.1f}%[/cyan]
        """))
        console.print(Panel(summary_text, title="[bold blue]Render Summary[/bold blue]", border_style="blue"))


def get_camera_vectors_at_time(animation_time):
    """
    Calculates the camera position, direction, up vector, and FOV for a given time.
    This function defines the cinematic camera path.
    """
    # --- Orbiting motion (horizontal angle) ---
    orbit_period = 120.0   # slow orbit to last 2 minutes for smooth effect
    phi = (animation_time / orbit_period) * 2.0 * math.pi + math.pi

    # --- Dolly motion (distance from center) ---
    start_radius, end_radius = 50.0, 18.0
    animation_duration = 60.0   # match the video duration
    progress = min(animation_time / animation_duration, 1.0)
    eased_progress = 0.5 * (1.0 - math.cos(progress * math.pi))
    radius = start_radius + (end_radius - start_radius) * eased_progress
    
    # --- Vertical motion (vertical angle) ---
    start_theta, end_theta = math.pi / 2.0 - 0.4, math.pi / 2.0 + 0.2
    theta = start_theta + (end_theta - start_theta) * eased_progress

    # --- FOV (zoom effect) ---
    start_fov, end_fov = 1.0, 1.3
    fov = start_fov + (end_fov - start_fov) * eased_progress

    # Calculate final camera vectors from path parameters
    cam_x = radius * math.sin(theta) * math.cos(phi)
    cam_y = radius * math.cos(theta)
    cam_z = radius * math.sin(theta) * math.sin(phi)
    
    cam_pos = ti.Vector([cam_x, cam_y, cam_z])
    cam_fwd = -cam_pos.normalized()
    world_up = ti.Vector([0.0, 1.0, 0.0])
    
    return cam_pos, cam_fwd, world_up, fov

if __name__ == "__main__":
    main()