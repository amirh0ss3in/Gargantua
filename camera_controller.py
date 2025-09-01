import time
import math
import taichi as ti
from interstellar_renderer import InterstellarRenderer

try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TaskProgressColumn
    from rich.live import Live
    _rich_available = True
    console = Console()
except ImportError:
    _rich_available = False
    class _DummyConsole:
        def print(self, text): print(text)
    console = _DummyConsole()
    # Fallback to tqdm if rich is not fully available
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
    SHOW_GUI = False
    SAVE_VIDEO = True

    VIDEO_DURATION_SECONDS = 60
    VIDEO_FPS = 60
    OUTPUT_FILENAME = "interstellar_flight.mp4"

    RENDER_WIDTH = 1980

    # 2. Initialize the scene renderer
    scene = InterstellarRenderer(
        show_gui=SHOW_GUI,
        save_video_path=OUTPUT_FILENAME if SAVE_VIDEO else None,
        video_fps=VIDEO_FPS,
        width=RENDER_WIDTH,
        use_caching=True
    )
    
    # 3. Main application loop
    if SAVE_VIDEO:
        total_frames = int(VIDEO_DURATION_SECONDS * VIDEO_FPS)
        
        cache_hits, cache_misses = 0, 0
        total_render_time = 0.0

        if _rich_available:
            # --- 1. Build all UI components before starting the Live display ---
            
            # Get the renderer's configuration panel
            # (Assumes you've added the get_init_panel() method to the renderer class)
            init_panel = scene.get_init_panel() 
            
            # Create the render job information panel
            job_text = Text.from_markup(f"""\
[bold]Total Frames:[/bold] {total_frames}
[bold]Output File:[/bold]  [cyan]'{OUTPUT_FILENAME}'[/cyan]
""")
            job_panel = Panel(job_text, title="[bold magenta]Render Job Started[/bold magenta]", border_style="magenta", expand=False)

            # Define and create the progress bar object
            progress_columns = [
                TextColumn("{task.description}"),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
                TimeRemainingColumn(),
            ]
            progress = Progress(*progress_columns, transient=True) # transient=True makes it disappear on completion

            # --- 2. Combine all components into a single Group ---
            # This group is the single "renderable" that Live will manage.
            render_group = Group(
                init_panel,
                job_panel,
                progress
            )

            # --- 3. Run the render loop inside the Live context manager ---
            # Live will now handle drawing the entire group, making it immune to resize artifacts.
            with Live(render_group, console=console, refresh_per_second=10) as live:
                task = progress.add_task("Starting...", total=total_frames)

                for frame in range(total_frames):
                    start_frame_time = time.time()
                    
                    animation_time = frame / VIDEO_FPS
                    cam_pos, cam_fwd, world_up, fov = get_camera_vectors_at_time(animation_time)
                    status = scene.step(cam_pos, cam_fwd, world_up, fov)
                    
                    frame_duration = time.time() - start_frame_time
                    total_render_time += frame_duration
                    
                    if status == 'cache_hit': cache_hits += 1
                    else: cache_misses += 1

                    stats_str = scene.get_system_stats_str()
                    cache_str = f"Cache: [green]✔ {cache_hits}[/green] [red]✖ {cache_misses}[/red]"

                    description_text = Text.from_markup(
                        f"[cyan]Rendering Frame {frame + 1}/{total_frames}[/cyan]\n"
                        f"[dim]  └─ Last: {frame_duration:.2f}s | {cache_str} | {stats_str}[/dim]"
                    )
                    
                    # Update the progress bar. Live will automatically refresh the display.
                    progress.update(task, advance=1, description=description_text)
        else:
            # Fallback to a simple loop if rich is not available
            print(f"\nStarting offline render of {total_frames} frames to '{OUTPUT_FILENAME}'...")
            for frame in range(total_frames):
                print(f"Rendering frame {frame+1}/{total_frames}...")
                animation_time = frame / VIDEO_FPS
                cam_pos, cam_fwd, world_up, fov = get_camera_vectors_at_time(animation_time)
                scene.step(cam_pos, cam_fwd, world_up, fov)

    else:
        # --- REAL-TIME GUI LOOP ---
        start_time = time.time()
        while scene.gui and scene.gui.running:
            for e in scene.gui.get_events(ti.GUI.PRESS):
                if e.key == ti.GUI.ESCAPE:
                    scene.gui.running = False

            animation_time = time.time() - start_time
            cam_pos, cam_fwd, world_up, fov = get_camera_vectors_at_time(animation_time)
            scene.step(cam_pos, cam_fwd, world_up, fov)

    # 4. Clean up and finalize resources
    scene.close()

    # 5. Print final summary for offline renders
    if SAVE_VIDEO and _rich_available:
        summary_table = Table(show_header=False, box=None, padding=(0, 2))
        summary_table.add_column(style="bold blue")
        summary_table.add_column()

        summary_table.add_row("Total Frames:", f"{total_frames}")
        summary_table.add_row("Total Time:", f"{total_render_time:.2f} seconds")
        summary_table.add_row("Average Time:", f"{(total_render_time/total_frames):.2f} s/frame" if total_frames > 0 else "N/A")
        summary_table.add_row("","") # Spacer
        summary_table.add_row("Cache Hits:", f"[green]{cache_hits}[/green]")
        summary_table.add_row("Cache Misses:", f"[red]{cache_misses}[/red]")
        summary_table.add_row("Cache Hit Rate:", f"[cyan]{(cache_hits / total_frames * 100):.1f}%[/cyan]" if total_frames > 0 else "N/A")

        console.print(Panel(summary_table, title="[bold blue]Render Summary[/bold blue]", border_style="blue"))
    elif SAVE_VIDEO:
        print("\n--- Render Summary ---")
        print(f"Total Frames: {total_frames}")
        print(f"Total Time: {total_render_time:.2f} seconds")
        print(f"Cache Hits: {cache_hits}, Misses: {cache_misses}")


def get_camera_vectors_at_time(animation_time):
    """
    Calculates the camera position, direction, up vector, and FOV for a given time.
    This function defines the cinematic camera path.
    """
    orbit_period = 120.0
    phi = (animation_time / orbit_period) * 2.0 * math.pi + math.pi

    start_radius, end_radius = 50.0, 18.0
    animation_duration = 60.0
    progress = min(animation_time / animation_duration, 1.0)
    eased_progress = 0.5 * (1.0 - math.cos(progress * math.pi))
    radius = start_radius + (end_radius - start_radius) * eased_progress
    
    start_theta, end_theta = math.pi / 2.0 - 0.4, math.pi / 2.0 + 0.2
    theta = start_theta + (end_theta - start_theta) * eased_progress

    start_fov, end_fov = 1.0, 1.3
    fov = start_fov + (end_fov - start_fov) * eased_progress

    cam_x = radius * math.sin(theta) * math.cos(phi)
    cam_y = radius * math.cos(theta)
    cam_z = radius * math.sin(theta) * math.sin(phi)
    
    cam_pos = ti.Vector([cam_x, cam_y, cam_z])
    cam_fwd = -cam_pos.normalized()
    world_up = ti.Vector([0.0, 1.0, 0.0])
    
    return cam_pos, cam_fwd, world_up, fov

if __name__ == "__main__":
    main()